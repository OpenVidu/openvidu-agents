/**
 * Headless transcription capacity probe.
 *
 * Publishers are LiveKit Node SDK participants that stream a WAV file in a
 * loop, and a track counts once the agent's final transcription of it (the
 * `lk.transcription` text stream sent on behalf of the publisher) arrives
 * within CAPACITY_VERIFY_TIMEOUT_MS. The ramp adds one publisher at a time
 * (rooms of CAPACITY_PUBLISHERS_PER_ROOM), stops after
 * CAPACITY_MAX_CONSECUTIVE_FAILURES failed adds in a row, at
 * CAPACITY_HARD_CAP tracks or when CAPACITY_RAMP_BUDGET_MS runs out, then
 * re-checks the OLDEST track for a fresh final so the result means "N tracks
 * sustained".
 *
 * The publishers of each room live in their own child process
 * (publisher-worker.ts): one Node process saturates at 15-20 publishers and
 * then stops counting finals that keep arriving, which would look like the
 * agent's limit.
 *
 * Configuration (environment):
 *   LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET   the agent host (ws://<ip>:7880, devkey/secret)
 *   CAPACITY_AUDIO_FILE                  WAV to loop (default e2e/resources/stt-test.wav, downloaded if missing)
 *   CAPACITY_HARD_CAP                    default 40
 *   CAPACITY_PUBLISHERS_PER_ROOM         default 3 (= publishers per worker process)
 *   CAPACITY_VERIFY_TIMEOUT_MS           default 60000
 *   CAPACITY_MAX_CONSECUTIVE_FAILURES    default 2
 *   CAPACITY_JOIN_ATTEMPTS               default 3: joins refused by the server are retried
 *   CAPACITY_JOIN_RETRY_DELAY_MS         default 5000, pause between join attempts
 *   CAPACITY_RAMP_BUDGET_MS              default 40 min
 *   CAPACITY_LABEL                       free text added to the result line
 *
 * Result: one `CAPACITY RESULT:` line, plus one `Capacity so far:` line per
 * accepted track with the publisher host's own load, so a saturated publisher
 * machine is visible. In GitHub Actions the result is also appended to the
 * job summary. Exit code 1 when no track was transcribed at all.
 */
import { ChildProcess, fork } from "child_process";
import fs from "fs";
import os from "os";
import path from "path";
import { downloadFile } from "../e2e/utils/helper";
import { hostBusyPercent, readHostCpuJiffies } from "../e2e/utils/host-load";
import { SAMPLE_RATE, loadWavAsMono48k, log, sleep } from "./publisher";

const LIVEKIT_URL = process.env.LIVEKIT_URL || "ws://localhost:7880";
const AUDIO_FILE =
  process.env.CAPACITY_AUDIO_FILE || path.join(__dirname, "..", "e2e", "resources", "stt-test.wav");
const AUDIO_URL = "https://s3.eu-west-1.amazonaws.com/public.openvidu.io/stt-test.wav";
const HARD_CAP_TRACKS = Number(process.env.CAPACITY_HARD_CAP || 40);
const PUBLISHERS_PER_ROOM = Number(process.env.CAPACITY_PUBLISHERS_PER_ROOM || 3);
const TRACK_VERIFY_TIMEOUT_MS = Number(process.env.CAPACITY_VERIFY_TIMEOUT_MS || 60000);
const MAX_CONSECUTIVE_FAILURES = Number(process.env.CAPACITY_MAX_CONSECUTIVE_FAILURES || 2);
const JOIN_ATTEMPTS = Number(process.env.CAPACITY_JOIN_ATTEMPTS || 3);
const JOIN_RETRY_DELAY_MS = Number(process.env.CAPACITY_JOIN_RETRY_DELAY_MS || 5000);
const RAMP_BUDGET_MS = Number(process.env.CAPACITY_RAMP_BUDGET_MS || 40 * 60 * 1000);
const LABEL = process.env.CAPACITY_LABEL || "";
const RUN_ID = Date.now().toString(36);

/** One publisher-worker.ts process, addressed with request/reply messages. */
class Worker {
  private readonly proc: ChildProcess;
  private readonly pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
  private nextId = 1;

  constructor(readonly index: number) {
    this.proc = fork(path.join(__dirname, "publisher-worker.ts"), [], {
      execArgv: ["-r", require.resolve("ts-node/register/transpile-only")],
      env: { ...process.env, CAPACITY_AUDIO_FILE: AUDIO_FILE },
      stdio: ["ignore", "inherit", "inherit", "ipc"],
    });
    this.proc.on("message", (msg: any) => {
      const waiter = this.pending.get(msg.id);
      if (!waiter) {
        return;
      }
      this.pending.delete(msg.id);
      msg.ok ? waiter.resolve(msg) : waiter.reject(new Error(msg.message));
    });
    this.proc.on("exit", (code) => {
      for (const waiter of this.pending.values()) {
        waiter.reject(new Error(`publisher worker ${this.index} exited (${code})`));
      }
      this.pending.clear();
    });
  }

  get pid(): number | undefined {
    return this.proc.pid;
  }

  request(message: Record<string, unknown>, timeoutMs = 0): Promise<any> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      if (timeoutMs > 0) {
        setTimeout(() => {
          if (this.pending.delete(id)) {
            reject(new Error(`publisher worker ${this.index}: no reply to ${message.type} in ${timeoutMs / 1000}s`));
          }
        }, timeoutMs).unref();
      }
      this.proc.send({ id, ...message });
    });
  }

  async exit(): Promise<void> {
    if (this.proc.exitCode !== null) {
      return;
    }
    await this.request({ type: "exit" }, 15_000).catch(() => undefined);
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        this.proc.kill();
        resolve();
      }, 5_000);
      this.proc.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }
}

/** CPU jiffies (user + system) of a process, from /proc; null off Linux. */
function processJiffies(pid: number): number | null {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, "utf8");
    const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    return Number(fields[11]) + Number(fields[12]);
  } catch {
    return null;
  }
}

/** The publisher host's own load since the previous call: host busy % and the probe processes' CPU %. */
class PublisherHostLoad {
  private lastJiffies = readHostCpuJiffies();
  private lastProc = new Map<number, number>();
  private lastAt = Date.now();

  constructor(private readonly pids: () => number[]) {}

  sample(): string {
    const now = Date.now();
    const elapsedS = Math.max(0.001, (now - this.lastAt) / 1000);
    let used = 0;
    const current = new Map<number, number>();
    for (const pid of this.pids()) {
      const jiffies = processJiffies(pid);
      if (jiffies === null) {
        continue;
      }
      current.set(pid, jiffies);
      const before = this.lastProc.get(pid);
      if (before !== undefined) {
        used += jiffies - before;
      }
    }
    // Jiffies at CLK_TCK = 100 Hz, the Linux default.
    const processPct = (used / 100 / elapsedS) * 100;
    const jiffiesNow = readHostCpuJiffies();
    const hostPct = hostBusyPercent(this.lastJiffies, jiffiesNow);
    this.lastJiffies = jiffiesNow;
    this.lastProc = current;
    this.lastAt = now;
    // Like the agent host's samples: busy % summed over all cores, out of cores x 100.
    const host = hostPct === null ? "n/a" : `${hostPct.toFixed(0)}% of ${os.cpus().length * 100}%`;
    return `[publisher host: ${host} busy | probe processes: ${processPct.toFixed(0)}%]`;
  }
}

interface TrackRef {
  identity: string;
  worker: Worker;
}

async function main(): Promise<void> {
  if (!fs.existsSync(AUDIO_FILE) || fs.statSync(AUDIO_FILE).size === 0) {
    fs.mkdirSync(path.dirname(AUDIO_FILE), { recursive: true });
    log(`Downloading ${AUDIO_URL} to ${AUDIO_FILE}`);
    await downloadFile(AUDIO_URL, AUDIO_FILE);
  }
  const audio = loadWavAsMono48k(AUDIO_FILE);
  log(
    `Probe ${RUN_ID}: agent host ${LIVEKIT_URL}, fixture ${path.basename(AUDIO_FILE)} ` +
      `(${(audio.length / SAMPLE_RATE).toFixed(1)} s, looped), ${PUBLISHERS_PER_ROOM} publishers per room ` +
      `(one worker process per room), verify ${TRACK_VERIFY_TIMEOUT_MS / 1000}s, hard cap ${HARD_CAP_TRACKS}, ` +
      `budget ${RAMP_BUDGET_MS / 60000} min` +
      (LABEL ? `, label "${LABEL}"` : ""),
  );

  const workers: Worker[] = [];
  const workerForRoom = (room: number): Worker => {
    while (workers.length <= room) {
      workers.push(new Worker(workers.length));
    }
    return workers[room];
  };
  const tracksRefs: TrackRef[] = [];
  const load = new PublisherHostLoad(() =>
    [process.pid, ...workers.map((w) => w.pid)].filter((p): p is number => typeof p === "number"),
  );
  load.sample();
  let tracks = 0;
  let consecutiveFailures = 0;
  let joinRetries = 0;
  let stopReason = `hard cap of ${HARD_CAP_TRACKS} tracks reached`;
  const rampDeadline = Date.now() + RAMP_BUDGET_MS;

  while (tracks < HARD_CAP_TRACKS) {
    if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
      stopReason = `${MAX_CONSECUTIVE_FAILURES} consecutive tracks failed to be transcribed`;
      break;
    }
    if (Date.now() > rampDeadline) {
      stopReason = `ramp time budget of ${RAMP_BUDGET_MS / 60000} minutes exhausted`;
      break;
    }
    const index = tracks + consecutiveFailures + 1;
    const room = Math.floor(tracks / PUBLISHERS_PER_ROOM);
    const roomName = `capacity-${RUN_ID}-room-${room}`;
    const identity = `capacity-pub-${index}`;
    const worker = workerForRoom(room);
    try {
      // A join the server refuses (HTTP 500 while it judges its node
      // unavailable for a new room, for instance) is retried after a pause:
      // a transient signaling error must not end the ramp, but every retry
      // is logged and counted in the result.
      for (let attempt = 1; ; attempt++) {
        try {
          await worker.request({ type: "add", identity, roomName }, 120_000);
          break;
        } catch (error: any) {
          if (attempt >= JOIN_ATTEMPTS) {
            throw error;
          }
          joinRetries += 1;
          log(
            `Track ${tracks + 1}: join attempt ${attempt}/${JOIN_ATTEMPTS} failed ` +
              `(${error.message}); retrying in ${JOIN_RETRY_DELAY_MS / 1000}s`,
          );
          await sleep(JOIN_RETRY_DELAY_MS);
        }
      }
      const reply = await worker.request(
        { type: "waitFinal", identity, timeoutMs: TRACK_VERIFY_TIMEOUT_MS, after: 0 },
        TRACK_VERIFY_TIMEOUT_MS + 30_000,
      );
      tracksRefs.push({ identity, worker });
      tracks += 1;
      consecutiveFailures = 0;
      log(`Capacity so far: ${tracks} simultaneous transcribed tracks ${load.sample()} first final: "${reply.text}"`);
    } catch (error: any) {
      consecutiveFailures += 1;
      log(
        `Track ${tracks + 1} failed (${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES} consecutive) ` +
          `${load.sample()}: ${error.message}`,
      );
      await worker.request({ type: "close", identity }, 30_000).catch(() => undefined);
    }
  }

  // Sustained check: the FIRST publisher must still get fresh finals now that
  // every other track is live, otherwise the count reflects accepted-then-degraded tracks.
  let sustained = false;
  if (tracksRefs.length > 0) {
    const oldest = tracksRefs[0];
    try {
      const stats = await oldest.worker.request({ type: "stats" }, 30_000);
      const finals = stats.publishers.find((p: any) => p.identity === oldest.identity)?.finals ?? 0;
      await oldest.worker.request(
        { type: "waitFinal", identity: oldest.identity, timeoutMs: TRACK_VERIFY_TIMEOUT_MS, after: finals },
        TRACK_VERIFY_TIMEOUT_MS + 30_000,
      );
      sustained = true;
    } catch {
      log("Oldest track stopped receiving transcriptions at full load");
    }
  }

  // Finals per track and the tracks still delivering in the last verify
  // window: an agent that accepted the ramp but fell behind shows as k/N below N.
  const finalsByIdentity = new Map<string, { finals: number; lastFinalAt: number }>();
  for (const worker of workers) {
    const stats = await worker.request({ type: "stats" }, 30_000).catch(() => ({ publishers: [] }));
    for (const p of stats.publishers as Array<{ identity: string; finals: number; lastFinalAt: number }>) {
      finalsByIdentity.set(p.identity, p);
    }
  }
  const now = Date.now();
  const finalsPerTrack = tracksRefs.map((t) => finalsByIdentity.get(t.identity)?.finals ?? 0);
  const liveAtEnd = tracksRefs.filter(
    (t) => now - (finalsByIdentity.get(t.identity)?.lastFinalAt ?? 0) < TRACK_VERIFY_TIMEOUT_MS,
  ).length;
  const resultLine =
    `CAPACITY RESULT: ${tracks} simultaneous transcribed tracks with headless publishers` +
    (LABEL ? ` [${LABEL}]` : "") +
    ` (agent host: ${LIVEKIT_URL}; stop reason: ${stopReason}; oldest track still transcribing ` +
    `at full load: ${sustained}; tracks with a final in the last ${TRACK_VERIFY_TIMEOUT_MS / 1000}s: ` +
    `${liveAtEnd}/${tracks}; join retries: ${joinRetries}; finals per track: ${finalsPerTrack.join(",")}) ${load.sample()}`;
  log(resultLine);
  // In GitHub Actions, the publishers' half of the picture goes to the job
  // summary; the agent host adds the CPU figures (tracks per vCPU) to its own
  // once this job is complete.
  if (process.env.GITHUB_STEP_SUMMARY) {
    fs.appendFileSync(
      process.env.GITHUB_STEP_SUMMARY,
      [
        `### Capacity probe${LABEL ? `: ${LABEL}` : ""}`,
        "",
        "| Metric | Value |",
        "| --- | --- |",
        `| Simultaneous transcribed tracks | **${tracks}** |`,
        `| Stop reason | ${stopReason} |`,
        `| Oldest track still transcribing at full load | ${sustained ? "yes" : "no"} |`,
        `| Tracks with a final in the last ${TRACK_VERIFY_TIMEOUT_MS / 1000} s | ${liveAtEnd}/${tracks} |`,
        `| Join retries | ${joinRetries} |`,
        `| Finals per track | ${finalsPerTrack.join(", ") || "none"} |`,
        `| Agent host | ${LIVEKIT_URL} |`,
        "",
        "The agent host job's summary adds the CPU behind these tracks (CPU per track, tracks per vCPU).",
        "",
      ].join("\n"),
    );
  }

  await Promise.all(workers.map((w) => w.exit()));
  if (tracks === 0) {
    process.exit(1);
  }
}

main().catch(async (error) => {
  log(`Probe failed: ${error?.stack || error}`);
  process.exit(1);
});
