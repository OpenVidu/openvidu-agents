/**
 * Child process of the capacity probe hosting the publishers of one room.
 * Forked by headless-capacity-probe.ts (which sets LIVEKIT_URL,
 * LIVEKIT_API_KEY, LIVEKIT_API_SECRET and CAPACITY_AUDIO_FILE in its
 * environment) and driven over IPC. Every request carries an `id` that the
 * reply echoes, with `ok` and either the payload or a `message`.
 *
 * A single Node process saturates at 15-20 publishers (the text-stream
 * callbacks starve behind the audio pushes, and the agent's finals stop being
 * counted although they keep arriving), so the probe spreads rooms over these
 * workers and the publisher host's cores.
 */
import { dispose } from "@livekit/rtc-node";
import { Publisher, loadWavAsMono48k, log } from "./publisher";

type Request =
  | { id: number; type: "add"; identity: string; roomName: string }
  | { id: number; type: "waitFinal"; identity: string; timeoutMs: number; after: number }
  | { id: number; type: "stats" }
  | { id: number; type: "close"; identity?: string }
  | { id: number; type: "exit" };

const creds = {
  url: process.env.LIVEKIT_URL || "ws://localhost:7880",
  apiKey: process.env.LIVEKIT_API_KEY || "devkey",
  apiSecret: process.env.LIVEKIT_API_SECRET || "secret",
};
const audioFile = process.env.CAPACITY_AUDIO_FILE;
if (!audioFile) {
  throw new Error("publisher-worker: CAPACITY_AUDIO_FILE is not set");
}
const audio = loadWavAsMono48k(audioFile);
const publishers = new Map<string, Publisher>();

function reply(message: Record<string, unknown>): void {
  process.send?.(message);
}

async function closePublisher(identity: string): Promise<void> {
  const publisher = publishers.get(identity);
  if (publisher) {
    publishers.delete(identity);
    await publisher.close();
  }
}

async function handle(req: Request): Promise<Record<string, unknown>> {
  switch (req.type) {
    case "add": {
      // A retried join gets a fresh Publisher; the failed one is closed first.
      await closePublisher(req.identity);
      const publisher = new Publisher(req.identity, req.roomName, audio, creds);
      publishers.set(req.identity, publisher);
      try {
        await publisher.connect();
      } catch (error) {
        await closePublisher(req.identity);
        throw error;
      }
      return {};
    }
    case "waitFinal": {
      const publisher = publishers.get(req.identity);
      if (!publisher) {
        throw new Error(`${req.identity}: unknown publisher`);
      }
      await publisher.waitForOwnFinal(req.timeoutMs, req.after);
      return { finals: publisher.finals, text: publisher.lastFinalText };
    }
    case "stats":
      return {
        publishers: [...publishers.values()].map((p) => ({
          identity: p.identity,
          finals: p.finals,
          lastFinalAt: p.lastFinalAt,
        })),
      };
    case "close":
      if (req.identity) {
        await closePublisher(req.identity);
      } else {
        await Promise.all([...publishers.keys()].map(closePublisher));
      }
      return {};
    case "exit":
      await Promise.all([...publishers.keys()].map(closePublisher));
      await dispose();
      setTimeout(() => process.exit(0), 50);
      return {};
  }
}

process.on("message", (req: Request) => {
  handle(req)
    .then((payload) => reply({ id: req.id, ok: true, ...payload }))
    .catch((error: any) => reply({ id: req.id, ok: false, message: error?.message ?? String(error) }));
});

process.on("disconnect", () => {
  // The orchestrator is gone: do not leave publishers streaming.
  Promise.all([...publishers.keys()].map(closePublisher))
    .catch(() => undefined)
    .finally(() => process.exit(0));
});

log(`publisher worker ${process.pid} ready (${(audio.length / 48000).toFixed(1)} s of audio)`);
