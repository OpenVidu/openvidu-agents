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
 *   ts-node capacity/agent-host.ts stop    dump the agent's log tail and stop the deployment
 *
 * Configuration (environment):
 *   DEPLOYMENT_EDITION      community (default) or pro
 *   STT_ACCEL               cuda12 to use the -cuda12 image with GPU passthrough (empty = CPU image)
 *   CAPACITY_PROVIDER_JSON  provider entry as in the e2e specs, default: the sherpa provider with
 *                           the Nemotron 3.5 model of e2e/utils/models.ts, forced English
 *   LOCAL_DEPLOYMENT_BASE_PATH  where openvidu-local-deployment is checked out (LocalDeployment default)
 *   OPENVIDU_PRO_LICENSE    forwarded to the operator by LocalDeployment (Pro plugins need it)
 *   HOLD_UNTIL_JOB, HOLD_MAX_MINUTES (default 75), HOLD_SAMPLE_SECONDS (default 15)
 *   GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID  for the job poll (set by GitHub Actions)
 */
import fs from "fs";
import path from "path";
import { LocalDeployment } from "../e2e/utils/local-deployment";
import { execCommand } from "../e2e/utils/helper";
import { AGENT_CONTAINER, sampleHostLoad } from "../e2e/utils/host-load";
import { SHERPA_NEMOTRON_MODEL } from "../e2e/utils/models";

type Edition = "community" | "pro";

const EDITION = (process.env.DEPLOYMENT_EDITION as Edition) || "community";
const GPU = (process.env.STT_ACCEL || "").trim() === "cuda12";
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
  return {
    sherpa: {
      model: SHERPA_NEMOTRON_MODEL,
      language: "en",
      use_silero_vad: false,
    },
  };
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
      `(${GPU ? "GPU image, cuda12" : "CPU image"})`,
  );
  // LocalDeployment: sets the provider block, the agent image (+ GPU passthrough
  // when STT_ACCEL is set), the Pro license, runs configure_lan_private_ip_linux.sh,
  // `docker compose up -d` and waits for the agent worker to register.
  await LocalDeployment.start(EDITION, provider, undefined, "automatic");
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
  while (Date.now() < deadline) {
    log(`Host load ${sampleHostLoad({ gpu: GPU })}`);
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

function stop(): void {
  try {
    console.log(`\n=== ${AGENT_CONTAINER} log tail ===\n`);
    console.log(
      execCommand(`docker logs --tail 300 ${AGENT_CONTAINER} 2>&1 || true`),
    );
    console.log(`\n=== end of ${AGENT_CONTAINER} log tail ===\n`);
  } catch (error: any) {
    log(`Could not read the agent log: ${error.message}`);
  }
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
    case "stop":
      stop();
      break;
    default:
      console.error("usage: agent-host.ts start|hold|stop");
      process.exit(2);
  }
}

main().catch((error) => {
  log(`agent-host ${process.argv[2]} failed: ${error?.stack || error}`);
  process.exit(1);
});
