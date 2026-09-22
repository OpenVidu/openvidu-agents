/**
 * One headless LiveKit publisher of the capacity probe: joins a room with a
 * publish-only token, streams a looped WAV as a 48 kHz Opus track and counts
 * the final transcriptions the agent sends back for it. Hosted by
 * `publisher-worker.ts` (one process per room), driven by
 * `headless-capacity-probe.ts`.
 */
import fs from "fs";
import { AccessToken } from "livekit-server-sdk";
import {
  AudioFrame,
  AudioSource,
  LocalAudioTrack,
  Room,
  TrackPublishOptions,
  TrackSource,
} from "@livekit/rtc-node";

// The SDK encodes Opus at 48 kHz; frames of 100 ms keep the FFI call rate low
// (10/s per track) and the SDK's queue (~1 s) paces the playout.
export const SAMPLE_RATE = 48000;
export const FRAME_MS = 100;
export const FRAME_SAMPLES = (SAMPLE_RATE * FRAME_MS) / 1000;
export const TRANSCRIPTION_TOPIC = "lk.transcription";

export function log(message: string): void {
  console.log(`${new Date().toISOString()} ${message}`);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Decode a 16-bit PCM RIFF/WAVE file to mono 48 kHz samples. */
export function loadWavAsMono48k(file: string): Int16Array {
  const buf = fs.readFileSync(file);
  if (buf.toString("ascii", 0, 4) !== "RIFF" || buf.toString("ascii", 8, 12) !== "WAVE") {
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
        throw new Error(`${file}: only PCM WAV is supported (format ${format})`);
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
    throw new Error(`${file}: unsupported WAV layout (channels=${channels}, rate=${rate}, bits=${bits})`);
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

export interface LiveKitCredentials {
  url: string;
  apiKey: string;
  apiSecret: string;
}

export async function buildToken(identity: string, roomName: string, creds: LiveKitCredentials): Promise<string> {
  const at = new AccessToken(creds.apiKey, creds.apiSecret, { identity, name: identity, ttl: "3h" });
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

export class Publisher {
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
    private readonly creds: LiveKitCredentials,
  ) {}

  async connect(): Promise<void> {
    // Finals of this participant only: the agent sends the text stream on
    // behalf of the transcribed participant, so the sender identity is ours.
    // Interim results are not read at all: with some providers they are far
    // more frequent than finals and reading them is what saturated a single
    // publisher process.
    this.room.registerTextStreamHandler(TRANSCRIPTION_TOPIC, async (reader, participantInfo) => {
      const isFinal = reader.info.attributes?.["lk.transcription_final"] === "true";
      if (participantInfo.identity !== this.identity || !isFinal) {
        return;
      }
      const text = await reader.readAll();
      if (!text.trim()) {
        return;
      }
      this.finals += 1;
      this.lastFinalAt = Date.now();
      this.lastFinalText = text.trim();
    });
    const token = await buildToken(this.identity, this.roomName, this.creds);
    await this.room.connect(this.creds.url, token, { autoSubscribe: false, dynacast: false });
    if (!this.room.localParticipant) {
      throw new Error(`${this.identity}: connected without a local participant`);
    }
    this.source = new AudioSource(SAMPLE_RATE, 1);
    this.track = LocalAudioTrack.createAudioTrack(`${this.identity}-audio`, this.source);
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
        await source.captureFrame(new AudioFrame(chunk, SAMPLE_RATE, 1, FRAME_SAMPLES));
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
