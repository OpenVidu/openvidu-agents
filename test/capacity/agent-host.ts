/**
 * Agent-host side of the remote-publisher capacity probe
 * (.github/workflows/speech-processing-capacity-remote-publishers.yml).
 *
 * Runs on the machine under test, next to the OpenVidu local deployment:
 *
 *   ts-node capacity/agent-host.ts start   configure the agent provider in the local
 *                                          deployment, (re)start it and wait for the
 *                                          agent worker to register; prints the
 *                                          `AGENT HOST READY` line with the LiveKit URL
 *   ts-node capacity/agent-host.ts hold    keep the deployment up while the publishers
 *                                          run elsewhere, sampling the host load every
 *                                          HOLD_SAMPLE_SECONDS; returns when the GitHub
 *                                          Actions job named HOLD_UNTIL_JOB (same run)
 *                                          completes, or after HOLD_MAX_MINUTES
 *   ts-node capacity/agent-host.ts summary combine the samples `hold` wrote (CAPACITY_SAMPLES_FILE)
 *                                          with the Publishers job's result (GitHub API) into the
 *                                          capacity figures; appended to GITHUB_STEP_SUMMARY when set
 *   ts-node capacity/agent-host.ts stop    dump the agent's log tail and stop the deployment
 *
 * Configuration (environment):
 *   DEPLOYMENT_EDITION      community (default) or pro
 *   STT_ACCEL               cuda12 to use the -cuda12 image with GPU passthrough (empty = CPU image)
 *   CAPACITY_PROVIDER_JSON  provider entry as in the e2e specs, default: the sherpa provider with
 *                           the Nemotron 3.5 model of e2e/utils/models.ts, forced English
 *   CAPACITY_PROVIDER       sherpa (default) or vosk
 *   CAPACITY_MODEL          model directory of that provider's image (defaults: Nemotron 3.5 / vosk-model-en-us-0.22-lgraph)
 *   CAPACITY_JOB_EXECUTOR   thread|process: JOB_EXECUTOR_TYPE for the agent container (empty: the agent's default)
 *   LOCAL_DEPLOYMENT_BASE_PATH  where openvidu-local-deployment is checked out (LocalDeployment default)
 *   OPENVIDU_PRO_LICENSE    forwarded to the operator by LocalDeployment (Pro plugins need it)
 *   HOLD_UNTIL_JOB, HOLD_MAX_MINUTES (default 75), HOLD_SAMPLE_SECONDS (default 15)
 *   CAPACITY_SAMPLES_FILE   where `hold` appends its timestamped samples for `summary` (default capacity-host-load.log)
 *   GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID  for the job poll (set by GitHub Actions)
 */
import fs from "fs";
import path from "path";
import { LocalDeployment } from "../e2e/utils/local-deployment";
import { execCommand } from "../e2e/utils/helper";
import { AGENT_CONTAINER, sampleHostLoad } from "../e2e/utils/host-load";
import { SHERPA_NEMOTRON_MODEL } from "../e2e/utils/models";
import {
  HostSample,
  PROGRESS_RE,
  RESULT_MARKER,
  parseResultLine,
  parseSampleLine,
  renderMarkdown,
  renderText,
  splitTimestampedLine,
  summarize,
} from "./summary";

type Edition = "community" | "pro";

const EDITION = (process.env.DEPLOYMENT_EDITION as Edition) || "community";
const GPU = (process.env.STT_ACCEL || "").trim() === "cuda12";
/** LiveKit server container of the local deployment (both editions). */
const SERVER_CONTAINER = "openvidu";
/** thread (agent default for local providers) or process; empty leaves the agent's choice. */
const JOB_EXECUTOR = (process.env.CAPACITY_JOB_EXECUTOR || "").trim().toLowerCase();
/** The operator creates the agent container from agent-speech-processing.yaml (pulls its image). */
const OPERATOR_CONTAINER = "operator";
/** Timestamped `Host load [...]` samples written by `hold`, read by `summary`. */
const SAMPLES_FILE = process.env.CAPACITY_SAMPLES_FILE || "capacity-host-load.log";
const LOCAL_DEPLOYMENT_BASE_PATH =
  process.env.LOCAL_DEPLOYMENT_BASE_PATH ||
  path.resolve(__dirname, "../../../openvidu-local-deployment");

function log(message: string): void {
  console.log(`${new Date().toISOString()} ${message}`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function providerConfig(): Record<string, any> {
  if (process.env.CAPACITY_PROVIDER_JSON) {
    return JSON.parse(process.env.CAPACITY_PROVIDER_JSON);
  }
  // CAPACITY_PROVIDER (sherpa, the default, or vosk) and CAPACITY_MODEL (any
  // model directory of that provider's image) select what is measured. The
  // defaults are the Nemotron export of the accel for sherpa and the large
  // English model for vosk. Only the multilingual NeMo transducer takes a
  // language prompt; zipformer and vosk models ignore the option.
  const provider = process.env.CAPACITY_PROVIDER || "sherpa";
  const model =
    process.env.CAPACITY_MODEL ||
    (provider === "vosk" ? "vosk-model-en-us-0.22-lgraph" : SHERPA_NEMOTRON_MODEL);
  const config: Record<string, any> = { model, use_silero_vad: false };
  if (provider === "sherpa" && model.includes("nemotron")) {
    config.language = "en";
  }
  return { [provider]: config };
}

/** LAN_PRIVATE_IP as written by configure_lan_private_ip_linux.sh during LocalDeployment.start. */
function deploymentIp(): string {
  const env = fs.readFileSync(
    path.join(LOCAL_DEPLOYMENT_BASE_PATH, EDITION, ".env"),
    "utf8",
  );
  const match = env.match(/^LAN_PRIVATE_IP=(.+)$/m);
  if (!match || !match[1].trim()) {
    throw new Error("LAN_PRIVATE_IP not set in the local deployment .env");
  }
  return match[1].trim();
}

async function start(): Promise<void> {
  const provider = providerConfig();
  log(
    `Starting the ${EDITION} local deployment with provider ${JSON.stringify(provider)} ` +
      `(${GPU ? "GPU image, cuda12" : "CPU image"}${JOB_EXECUTOR ? `, job executor ${JOB_EXECUTOR}` : ""})`,
  );
  // The operator reads agent-speech-processing.yaml when its container starts
  // and `docker compose up -d` leaves a running operator alone, so a deployment
  // the workflow brought up before this step would keep the agent disabled.
  // Like the e2e specs, stop first so every container starts from the
  // configured files.
  LocalDeployment.stop();
  // LocalDeployment: sets the provider block, the agent image (+ GPU passthrough
  // when STT_ACCEL is set), the Pro license, runs configure_lan_private_ip_linux.sh,
  // `docker compose up -d` and waits for the agent worker to register.
  // JOB_EXECUTOR_TYPE=process runs every Room in its own process instead of
  // the agent's default single process for local providers, whose one event
  // loop saturates around 20 tracks: the probe then measures the model.
  const agentEnvironment = JOB_EXECUTOR ? { JOB_EXECUTOR_TYPE: JOB_EXECUTOR } : undefined;
  await LocalDeployment.start(EDITION, provider, undefined, "automatic", agentEnvironment);
  // LocalDeployment.start tolerates a worker that never registered (the e2e
  // specs then fail on their own assertions). Here the publishers would be
  // launched against nothing, so the operator's and the agent's logs are
  // dumped and the step fails instead.
  let agentState = "";
  try {
    agentState = execCommand(`docker inspect -f "{{.State.Status}}" ${AGENT_CONTAINER}`).trim();
  } catch {
    agentState = "missing";
  }
  const registered =
    agentState === "running" &&
    execCommand(`docker logs ${AGENT_CONTAINER} 2>&1 || true`).includes("registered worker");
  if (!registered) {
    dumpLog(OPERATOR_CONTAINER, `docker logs --tail 200 ${OPERATOR_CONTAINER}`);
    dumpLog(AGENT_CONTAINER, `docker logs --tail 200 ${AGENT_CONTAINER}`);
    throw new Error(
      `agent container ${AGENT_CONTAINER} is ${agentState} and its worker has not registered; ` +
        `see the operator log above (image pull or agent start failure)`,
    );
  }
  const ip = deploymentIp();
  // Caddy publishes LiveKit's HTTP/WS on 7880 and RTC uses 7881/tcp + 7900-7999/udp
  // on the same address (openvidu-local-deployment/<edition>/docker-compose.yaml).
  log(
    `AGENT HOST READY url=ws://${ip}:7880 rtc_tcp=${ip}:7881 rtc_udp=${ip}:7900-7999`,
  );
  log(`Idle load ${sampleHostLoad({ gpu: GPU })}`);
}

/** Status of the job named `name` in the current GitHub Actions run, via the REST API. */
async function jobStatus(
  name: string,
): Promise<{ status: string; conclusion: string | null } | null> {
  const token = process.env.GITHUB_TOKEN;
  const repository = process.env.GITHUB_REPOSITORY;
  const runId = process.env.GITHUB_RUN_ID;
  if (!token || !repository || !runId) {
    return null;
  }
  const response = await fetch(
    `https://api.github.com/repos/${repository}/actions/runs/${runId}/jobs?per_page=100`,
    {
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/vnd.github+json",
      },
    },
  );
  if (!response.ok) {
    throw new Error(
      `GitHub API ${response.status} while listing the run's jobs`,
    );
  }
  const body = (await response.json()) as {
    jobs: Array<{ name: string; status: string; conclusion: string | null }>;
  };
  const job = body.jobs.find((j) => j.name === name);
  return job ? { status: job.status, conclusion: job.conclusion } : null;
}

async function hold(): Promise<void> {
  const untilJob = process.env.HOLD_UNTIL_JOB || "";
  const maxMinutes = Number(process.env.HOLD_MAX_MINUTES || 75);
  const sampleSeconds = Number(process.env.HOLD_SAMPLE_SECONDS || 15);
  const deadline = Date.now() + maxMinutes * 60 * 1000;
  log(
    `Holding the deployment up ${untilJob ? `until job "${untilJob}" completes` : `for ${maxMinutes} min`}, ` +
      `sampling the host load every ${sampleSeconds}s`,
  );
  let lastPoll = 0;
  let serverLogSince = new Date().toISOString();
  while (Date.now() < deadline) {
    const sample = sampleHostLoad({ gpu: GPU });
    log(`Host load ${sample}`);
    fs.appendFileSync(SAMPLES_FILE, `${new Date().toISOString()} Host load ${sample}\n`);
    // New warnings and errors of the LiveKit server and of the agent since the
    // previous sample, so a refused join or a stalled agent is explained here
    // with its time. The agent's per-utterance and per-loop-tick chatter is
    // filtered out; the 300-line tail dumped at `stop` never reaches back far
    // enough on its own.
    const now = new Date().toISOString();
    for (const [container, filter] of [
      [SERVER_CONTAINER, `grep -iE "warn|error|available nodes|could not|limit|closed|timeout|reconnect|participant (left|closed|removed)" | grep -vE "webhook|pion.ice|data channel|Failed to ping"`],
      [AGENT_CONTAINER, `grep -E "WARNING|ERROR|Traceback|job (ended|failed)|received job|shutting down" | grep -vE "skipping user input|event loop blocked|pre-connect audio handler"`],
    ] as Array<[string, string]>) {
      try {
        const lines = execCommand(
          `docker logs ${container} --since ${serverLogSince} 2>&1 | ${filter} | head -25 || true`,
        ).trim();
        if (lines) {
          console.log(`--- ${container} log since ${serverLogSince} ---\n${lines}`);
        }
      } catch {
        // the container may be gone during teardown
      }
    }
    serverLogSince = now;
    try {
      const state = execCommand(
        `docker inspect -f "{{.State.Status}}" ${AGENT_CONTAINER}`,
      ).trim();
      if (state !== "running") {
        log(`Agent container is ${state}; stopping the hold`);
        return;
      }
    } catch (error: any) {
      log(`Agent container not found (${error.message}); stopping the hold`);
      return;
    }
    if (untilJob && Date.now() - lastPoll > 30000) {
      lastPoll = Date.now();
      try {
        const job = await jobStatus(untilJob);
        if (job?.status === "completed") {
          log(
            `Job "${untilJob}" completed (${job.conclusion}); releasing the deployment`,
          );
          return;
        }
      } catch (error: any) {
        log(`Job poll failed, retrying later: ${error.message}`);
      }
    }
    await sleep(sampleSeconds * 1000);
  }
  log(`Hold budget of ${maxMinutes} min exhausted; releasing the deployment`);
}

function dumpLog(container: string, command: string): void {
  try {
    console.log(`\n=== ${container} log ===\n`);
    console.log(execCommand(`${command} 2>&1 || true`));
    console.log(`\n=== end of ${container} log ===\n`);
  } catch (error: any) {
    log(`Could not read the ${container} log: ${error.message}`);
  }
}

/** GitHub REST call with the workflow token; `raw` follows the log redirect without forwarding the token. */
async function githubApi(path: string, raw = false): Promise<string> {
  const token = process.env.GITHUB_TOKEN;
  const repository = process.env.GITHUB_REPOSITORY;
  if (!token || !repository) {
    throw new Error("GITHUB_TOKEN and GITHUB_REPOSITORY are required");
  }
  const url = `https://api.github.com/repos/${repository}${path}`;
  const headers = { Authorization: `Bearer ${token}`, Accept: "application/vnd.github+json" };
  if (!raw) {
    const response = await fetch(url, { headers });
    if (!response.ok) {
      throw new Error(`GitHub API ${response.status} for ${path}`);
    }
    return response.text();
  }
  // The logs endpoint answers 302 with a signed URL that rejects extra auth headers.
  const first = await fetch(url, { headers, redirect: "manual" });
  const location = first.headers.get("location");
  const response = location ? await fetch(location) : first;
  if (!response.ok) {
    throw new Error(`GitHub API ${response.status} for ${path} (logs)`);
  }
  return response.text();
}

/**
 * The agent host knows the CPU, the publishers know the tracks: once the
 * Publishers job is complete, join both into the capacity figures. Never fails
 * the job: a missing result (the probe did not run) is reported and that is all.
 */
async function summary(): Promise<void> {
  const runId = process.env.GITHUB_RUN_ID;
  if (!runId) {
    log("GITHUB_RUN_ID is not set; nothing to summarize");
    return;
  }
  let samples: HostSample[] = [];
  try {
    samples = fs
      .readFileSync(SAMPLES_FILE, "utf8")
      .split("\n")
      .flatMap((raw) => {
        const l = splitTimestampedLine(raw);
        const s = l && parseSampleLine(l.text, l.time);
        return s ? [s] : [];
      });
  } catch (error: any) {
    log(`No samples file ${SAMPLES_FILE} (${error.message}); nothing to summarize`);
    return;
  }
  const jobs = JSON.parse(await githubApi(`/actions/runs/${runId}/jobs?per_page=100`)) as {
    jobs: Array<{ id: number; name: string; status: string; conclusion: string | null }>;
  };
  const publishers = jobs.jobs.find((j) => j.name === "Publishers");
  if (!publishers || publishers.status !== "completed") {
    log(`Publishers job ${publishers ? publishers.status : "not found"}; nothing to summarize`);
    return;
  }
  const lines = (await githubApi(`/actions/jobs/${publishers.id}/logs`, true))
    .split("\n")
    .flatMap((raw) => {
      const l = splitTimestampedLine(raw);
      return l ? [l] : [];
    });
  const resultLine = lines.find((l) => l.text.includes(RESULT_MARKER));
  const result = resultLine && parseResultLine(resultLine.text);
  if (!resultLine || !result) {
    log(`Publishers job ${publishers.conclusion} without a ${RESULT_MARKER} line; nothing to summarize`);
    return;
  }
  const lastAccepted = [...lines].reverse().find((l) => PROGRESS_RE.test(l.text));
  const rampEnd = lastAccepted ? lastAccepted.time : resultLine.time - 60_000;
  const capacity = summarize(result, samples, rampEnd, resultLine.time);
  console.log(renderText(capacity));
  if (process.env.GITHUB_STEP_SUMMARY) {
    fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, renderMarkdown(capacity));
    log("Summary written to the job summary");
  }
}

function stop(): void {
  // Without the per-utterance chatter, so the tail reaches back to the ramp.
  dumpLog(
    AGENT_CONTAINER,
    `docker logs ${AGENT_CONTAINER} 2>&1 | grep -vE "FINAL|INTERIM|PARTIAL|skipping user input|event loop blocked|\\"level\\": \\"(INFO|DEBUG)\\"" | tail -600`,
  );
  dumpLog(OPERATOR_CONTAINER, `docker logs --tail 100 ${OPERATOR_CONTAINER}`);
  // The server decides whether a new room gets a node: keep its warnings,
  // errors and node-selection lines (a join refused with HTTP 500 shows here).
  dumpLog(
    SERVER_CONTAINER,
    `docker logs ${SERVER_CONTAINER} 2>&1 | grep -iE "warn|error|available nodes|node selected|limit|could not|failed" | tail -200`,
  );
  LocalDeployment.stop();
}

async function main(): Promise<void> {
  const command = process.argv[2];
  switch (command) {
    case "start":
      await start();
      break;
    case "hold":
      await hold();
      break;
    case "summary":
      await summary();
      break;
    case "stop":
      stop();
      break;
    default:
      console.error("usage: agent-host.ts start|hold|summary|stop");
      process.exit(2);
  }
}

main().catch((error) => {
  log(`agent-host ${process.argv[2]} failed: ${error?.stack || error}`);
  process.exit(1);
});
