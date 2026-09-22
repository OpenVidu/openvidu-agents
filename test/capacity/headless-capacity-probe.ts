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
 * Because it needs no browser, the probe runs on a machine other than the
 * agent host (see .github/workflows/speech-processing-capacity-remote-publishers.yml),
 * so the agent host's CPU, RAM and GPU are measured alone. A Node publisher
 * costs about one Opus encoder per track, so a small instance drives dozens.
 *
 * Configuration (environment):
 *   LIVEKIT_URL                          ws://<agent-host>:7880 (default ws://localhost:7880)
 *   LIVEKIT_API_KEY / LIVEKIT_API_SECRET devkey / secret (the local deployment's)
 *   CAPACITY_AUDIO_FILE                  16-bit PCM WAV to loop (default: e2e/resources/stt-test.wav, downloaded if missing)
 *   CAPACITY_HARD_CAP                    default 40
 *   CAPACITY_PUBLISHERS_PER_ROOM         default 3
 *   CAPACITY_VERIFY_TIMEOUT_MS           default 60000
 *   CAPACITY_MAX_CONSECUTIVE_FAILURES    default 2
 *   CAPACITY_JOIN_ATTEMPTS               default 3: joins refused by the server are retried
 *   CAPACITY_JOIN_RETRY_DELAY_MS         default 5000, pause between join attempts
 *   CAPACITY_RAMP_BUDGET_MS              default 40 min
 *   CAPACITY_LABEL                       free text echoed in the result line (e.g. "g4dn.xlarge cuda12")
 *
 * Result: one `CAPACITY RESULT:` line, plus one `Capacity so far:` line per
 * track with the publisher host's load (to show it is not the bottleneck).
 */
import fs from "fs";
import path from "path";
import { AccessToken } from "livekit-server-sdk";
import {
  AudioFrame,
  AudioSource,
  LocalAudioTrack,
  Room,
  TrackPublishOptions,
  TrackSource,
  dispose,
} from "@livekit/rtc-node";
import { downloadFile } from "../e2e/utils/helper";
import { hostBusyPercent, readHostCpuJiffies } from "../e2e/utils/host-load";

const LIVEKIT_URL = process.env.LIVEKIT_URL || "ws://localhost:7880";
const LIVEKIT_API_KEY = process.env.LIVEKIT_API_KEY || "devkey";
const LIVEKIT_API_SECRET = process.env.LIVEKIT_API_SECRET || "secret";
const AUDIO_FILE =
  process.env.CAPACITY_AUDIO_FILE ||
  path.join(__dirname, "..", "e2e", "resources", "stt-test.wav");
const AUDIO_URL =
  "https://s3.eu-west-1.amazonaws.com/public.openvidu.io/stt-test.wav";
const HARD_CAP_TRACKS = Number(process.env.CAPACITY_HARD_CAP || 40);
const PUBLISHERS_PER_ROOM = Number(
  process.env.CAPACITY_PUBLISHERS_PER_ROOM || 3,
);
const TRACK_VERIFY_TIMEOUT_MS = Number(
  process.env.CAPACITY_VERIFY_TIMEOUT_MS || 60000,
);
const MAX_CONSECUTIVE_FAILURES = Number(
  process.env.CAPACITY_MAX_CONSECUTIVE_FAILURES || 2,
);
const JOIN_ATTEMPTS = Number(process.env.CAPACITY_JOIN_ATTEMPTS || 3);
const JOIN_RETRY_DELAY_MS = Number(
  process.env.CAPACITY_JOIN_RETRY_DELAY_MS || 5000,
);
const RAMP_BUDGET_MS = Number(
  process.env.CAPACITY_RAMP_BUDGET_MS || 40 * 60 * 1000,
);
const LABEL = process.env.CAPACITY_LABEL || "";
const RUN_ID = Date.now().toString(36);

// The SDK encodes Opus at 48 kHz; frames of 100 ms keep the FFI call rate low
// (10/s per track) and the SDK's queue (~1 s) paces the playout.
const SAMPLE_RATE = 48000;
const FRAME_MS = 100;
const FRAME_SAMPLES = (SAMPLE_RATE * FRAME_MS) / 1000;

const TRANSCRIPTION_TOPIC = "lk.transcription";

function log(message: string): void {
  console.log(`${new Date().toISOString()} ${message}`);
}

/** Decode a 16-bit PCM RIFF/WAVE file to mono 48 kHz samples. */
function loadWavAsMono48k(file: string): Int16Array {
  const buf = fs.readFileSync(file);
  if (
    buf.toString("ascii", 0, 4) !== "RIFF" ||
    buf.toString("ascii", 8, 12) !== "WAVE"
  ) {
    throw new Error(`${file} is not a RIFF/WAVE file`);
  }
  let offset = 12;
  let channels = 0;
  let rate = 0;
  let bits = 0;
  let data: Buffer | null = null;
  while (offset + 8 <= buf.length) {
    const id = buf.toString("ascii", offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    const body = offset + 8;
    if (id === "fmt ") {
      const format = buf.readUInt16LE(body);
      if (format !== 1) {
        throw new Error(
          `${file}: only PCM WAV is supported (format ${format})`,
        );
      }
      channels = buf.readUInt16LE(body + 2);
      rate = buf.readUInt32LE(body + 4);
      bits = buf.readUInt16LE(body + 14);
    } else if (id === "data") {
      data = buf.subarray(body, Math.min(body + size, buf.length));
    }
    offset = body + size + (size % 2);
  }
  if (!data || !channels || !rate || bits !== 16) {
    throw new Error(
      `${file}: unsupported WAV layout (channels=${channels}, rate=${rate}, bits=${bits})`,
    );
  }
  const frames = Math.floor(data.length / 2 / channels);
  const mono = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      sum += data.readInt16LE((i * channels + c) * 2);
    }
    mono[i] = sum / channels;
  }
  if (rate === SAMPLE_RATE) {
    return Int16Array.from(mono, (v) => Math.round(v));
  }
  // Linear interpolation is enough for speech going into an Opus encoder.
  const outLength = Math.floor((frames * SAMPLE_RATE) / rate);
  const out = new Int16Array(outLength);
  const step = rate / SAMPLE_RATE;
  for (let i = 0; i < outLength; i++) {
    const pos = i * step;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, frames - 1);
    const frac = pos - i0;
    out[i] = Math.round(mono[i0] * (1 - frac) + mono[i1] * frac);
  }
  return out;
}

async function buildToken(identity: string, roomName: string): Promise<string> {
  const at = new AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, {
    identity,
    name: identity,
    ttl: "3h",
  });
  at.addGrant({
    room: roomName,
    roomJoin: true,
    canPublish: true,
    canPublishData: true,
    // Publisher-only: transcriptions arrive as text streams, no media
    // subscription is needed.
    canSubscribe: false,
  });
  return at.toJwt();
}

class Publisher {
  readonly room = new Room();
  private source: AudioSource | null = null;
  private track: LocalAudioTrack | null = null;
  private feeding: Promise<void> | null = null;
  private stopped = false;
  private position = 0;
  finals = 0;
  lastFinalAt = 0;
  lastFinalText = "";

  constructor(
    readonly identity: string,
    readonly roomName: string,
    private readonly audio: Int16Array,
  ) {}

  async connect(): Promise<void> {
    // Finals of this participant only: the agent sends the text stream on
    // behalf of the transcribed participant, so the sender identity is ours.
    this.room.registerTextStreamHandler(
      TRANSCRIPTION_TOPIC,
      async (reader, participantInfo) => {
        const text = await reader.readAll();
        const isFinal =
          reader.info.attributes?.["lk.transcription_final"] === "true";
        if (
          participantInfo.identity !== this.identity ||
          !isFinal ||
          !text.trim()
        ) {
          return;
        }
        this.finals += 1;
        this.lastFinalAt = Date.now();
        this.lastFinalText = text.trim();
      },
    );
    const token = await buildToken(this.identity, this.roomName);
    await this.room.connect(LIVEKIT_URL, token, {
      autoSubscribe: false,
      dynacast: false,
    });
    if (!this.room.localParticipant) {
      throw new Error(
        `${this.identity}: connected without a local participant`,
      );
    }
    this.source = new AudioSource(SAMPLE_RATE, 1);
    this.track = LocalAudioTrack.createAudioTrack(
      `${this.identity}-audio`,
      this.source,
    );
    const options = new TrackPublishOptions();
    options.source = TrackSource.SOURCE_MICROPHONE;
    await this.room.localParticipant.publishTrack(this.track, options);
    this.feeding = this.feed();
  }

  /** Push the looped WAV; captureFrame blocks while the SDK's queue is full, which paces us. */
  private async feed(): Promise<void> {
    const source = this.source!;
    try {
      while (!this.stopped) {
        const chunk = new Int16Array(FRAME_SAMPLES);
        for (let i = 0; i < FRAME_SAMPLES; i++) {
          chunk[i] = this.audio[this.position];
          this.position = (this.position + 1) % this.audio.length;
        }
        await source.captureFrame(
          new AudioFrame(chunk, SAMPLE_RATE, 1, FRAME_SAMPLES),
        );
      }
    } catch (error: any) {
      if (!this.stopped) {
        log(`${this.identity}: audio feed stopped: ${error.message}`);
      }
    }
  }

  /** Resolve once a final for this participant arrives, or throw at the deadline. */
  async waitForOwnFinal(timeoutMs: number, after = 0): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (this.finals > after) {
        return;
      }
      await sleep(250);
    }
    throw new Error(
      `${this.identity}: no final transcription within ${timeoutMs / 1000}s ` +
        `(${this.finals} finals so far in room ${this.roomName})`,
    );
  }

  async close(): Promise<void> {
    this.stopped = true;
    try {
      await this.source?.close();
    } catch {
      // ignore
    }
    if (this.feeding) {
      await this.feeding.catch(() => undefined);
    }
    try {
      await this.room.disconnect();
    } catch {
      // ignore
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** The publisher host's own load since the previous call: host busy % and this process' CPU %. */
class PublisherHostLoad {
  private lastJiffies = readHostCpuJiffies();
  private lastCpu = process.cpuUsage();
  private lastAt = Date.now();

  sample(): string {
    const now = Date.now();
    const jiffies = readHostCpuJiffies();
    const cpu = process.cpuUsage(this.lastCpu);
    const elapsedUs = Math.max(1, (now - this.lastAt) * 1000);
    const processPct = ((cpu.user + cpu.system) / elapsedUs) * 100;
    const hostPct = hostBusyPercent(this.lastJiffies, jiffies);
    this.lastJiffies = jiffies;
    this.lastCpu = process.cpuUsage();
    this.lastAt = now;
    const host = hostPct === null ? "n/a" : `${hostPct.toFixed(0)}%`;
    return `[publisher host: ${host} busy | probe process: ${processPct.toFixed(0)}%]`;
  }
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
      `(${(audio.length / SAMPLE_RATE).toFixed(1)} s, looped), ${PUBLISHERS_PER_ROOM} publishers per room, ` +
      `verify ${TRACK_VERIFY_TIMEOUT_MS / 1000}s, hard cap ${HARD_CAP_TRACKS}, budget ${RAMP_BUDGET_MS / 60000} min` +
      (LABEL ? `, label "${LABEL}"` : ""),
  );

  const publishers: Publisher[] = [];
  const load = new PublisherHostLoad();
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
    const index = publishers.length + 1;
    const roomName = `capacity-${RUN_ID}-room-${Math.floor(tracks / PUBLISHERS_PER_ROOM)}`;
    let publisher = new Publisher(`capacity-pub-${index}`, roomName, audio);
    try {
      // A join the server refuses (HTTP 500 while it judges its node
      // unavailable for a new room, for instance) is retried after a pause:
      // a transient signaling error must not end the ramp, but every retry
      // is logged and counted in the result.
      for (let attempt = 1; ; attempt++) {
        try {
          await publisher.connect();
          break;
        } catch (error: any) {
          await publisher.close();
          if (attempt >= JOIN_ATTEMPTS) {
            throw error;
          }
          joinRetries += 1;
          log(
            `Track ${tracks + 1}: join attempt ${attempt}/${JOIN_ATTEMPTS} failed ` +
              `(${error.message}); retrying in ${JOIN_RETRY_DELAY_MS / 1000}s`,
          );
          await sleep(JOIN_RETRY_DELAY_MS);
          publisher = new Publisher(`capacity-pub-${index}`, roomName, audio);
        }
      }
      await publisher.waitForOwnFinal(TRACK_VERIFY_TIMEOUT_MS);
      publishers.push(publisher);
      tracks += 1;
      consecutiveFailures = 0;
      log(
        `Capacity so far: ${tracks} simultaneous transcribed tracks ${load.sample()} first final: "${publisher.lastFinalText}"`,
      );
    } catch (error: any) {
      consecutiveFailures += 1;
      log(
        `Track ${tracks + 1} failed (${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES} consecutive) ` +
          `${load.sample()}: ${error.message}`,
      );
      await publisher.close();
    }
  }

  // Sustained check: the FIRST publisher must still get fresh finals now that
  // every other track is live, otherwise the count reflects accepted-then-degraded tracks.
  let sustained = false;
  if (publishers.length > 0) {
    const oldest = publishers[0];
    try {
      await oldest.waitForOwnFinal(TRACK_VERIFY_TIMEOUT_MS, oldest.finals);
      sustained = true;
    } catch {
      log("Oldest track stopped receiving transcriptions at full load");
    }
  }

  const finalsPerTrack = publishers.map((p) => p.finals);
  // Tracks that produced a final during the last verify window: an agent that
  // accepted the ramp but fell behind shows here as k/N well below N.
  const liveAtEnd = publishers.filter(
    (p) => Date.now() - p.lastFinalAt < TRACK_VERIFY_TIMEOUT_MS,
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

  await Promise.all(publishers.map((p) => p.close()));
  await dispose();
  if (tracks === 0) {
    process.exit(1);
  }
}

main().catch(async (error) => {
  log(`Probe failed: ${error?.stack || error}`);
  try {
    await dispose();
  } finally {
    process.exit(1);
  }
});
