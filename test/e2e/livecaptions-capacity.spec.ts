import { test, expect, Page, Locator } from "@playwright/test";
import fs from "fs";
import os from "os";
import { LocalDeployment } from "./utils/local-deployment";
import { TESTAPP_URL } from "./config";
import {
  downloadFile,
  execCommand,
  sleep,
  waitForEvent,
  waitForEventContentToStartWith,
} from "./utils/helper";
import { SHERPA_NEMOTRON_MODEL } from "./utils/models";

/**
 * Capacity probe: how many SIMULTANEOUS transcribed audio tracks can this
 * server sustain with the given STT provider (the sherpa provider serving the
 * Nemotron 3.5 model, see utils/models.ts)?
 *
 * The test ramps up one publisher-only audio participant at a time (packed
 * into rooms of PUBLISHERS_PER_ROOM). A track only counts if its OWN final
 * transcription arrives within TRACK_VERIFY_TIMEOUT_MS — so the reported
 * number is "tracks the agent actually transcribes", not "tracks published".
 * The ramp stops at the first sign of saturation (MAX_CONSECUTIVE_FAILURES
 * failed adds in a row), at HARD_CAP_TRACKS, or when RAMP_BUDGET_MS runs out;
 * then the OLDEST track is re-checked for a fresh transcription so the result
 * means "N tracks sustained", not "N accepted and then degraded".
 *
 * This is a measurement, not a pass/fail quality gate: the only assertion is
 * that at least one track was transcribed. The measured capacity is printed
 * to the log (grep for "CAPACITY RESULT").
 */

const PROVIDER: Record<string, any> = {
  sherpa: {
    model: SHERPA_NEMOTRON_MODEL,
    language: "en",
    use_silero_vad: false,
  },
};

// Safety ceiling so a surprisingly capable box cannot run the ramp forever.
const HARD_CAP_TRACKS = 40;
// Stop ramping when this many consecutive participant adds fail.
const MAX_CONSECUTIVE_FAILURES = 2;
// A new track must receive its own final transcription within this window.
const TRACK_VERIFY_TIMEOUT_MS = 60000;
// Total time allowed for the ramp itself (excluding deployment start/stop).
// A track takes 30-60 s to add and verify; 40 tracks need up to 40 min.
const RAMP_BUDGET_MS = 40 * 60 * 1000;
const PUBLISHERS_PER_ROOM = 3;
const PARTICIPANT_ACTION_TIMEOUT_MS = 10000;

const AGENT_CONTAINER = "agent-speech-processing";

// Whether nvidia-smi works on this host; probed once, cached.
let vramAvailable: boolean | null = null;

/** Parse docker's "12.34%" into a number (per-core convention: 400% = 4 cores). */
function parseDockerPercent(value: string): number {
  const n = parseFloat(value.replace("%", ""));
  return Number.isFinite(n) ? n : 0;
}

/**
 * Host CPU time in jiffies from /proc/stat (Linux only): busy and total. Two
 * snapshots around a sampling window give the host-wide utilization, which
 * includes what runs outside containers: the Playwright browser publishing
 * every track, and the runner itself.
 */
function readHostCpuJiffies(): { busy: number; total: number } | null {
  try {
    const line = fs.readFileSync("/proc/stat", "utf8").split("\n")[0];
    const fields = line.trim().split(/\s+/).slice(1).map(Number);
    if (fields.length < 5) {
      return null;
    }
    const total = fields.reduce((a: number, b: number) => a + b, 0);
    const idle = fields[3] + fields[4]; // idle + iowait
    return { busy: total - idle, total };
  } catch {
    return null;
  }
}

/**
 * One-line snapshot of where the machine's resources go as the ramp
 * progresses, to identify WHICH resource saturates first:
 * - the agent container (CPU% in docker's per-core convention, 400% = 4 cores
 *   fully busy, and RAM);
 * - every other container, as a total plus the three largest consumers;
 * - the host as a whole, busy % of all cores over the sampling window. The
 *   difference to the containers' sum is the browser running the publishers
 *   plus the runner, which share the box with the deployment in CI;
 * - on GPU runs, VRAM and GPU utilization from nvidia-smi (host-wide).
 */
function sampleAgentLoad(): string {
  const parts: string[] = [];
  const before = readHostCpuJiffies();
  try {
    // One docker stats call over every container. It takes ~2 s (docker
    // samples twice to derive CPU%), which doubles as the host sampling window.
    const rows = execCommand(
      'docker stats --no-stream --format "{{.Name}} {{.CPUPerc}} {{.MemUsage}}"',
    )
      .trim()
      .split("\n")
      .map((line) => line.trim().split(/\s+/))
      .filter((fields) => fields.length >= 2);
    const agent = rows.find((fields) => fields[0] === AGENT_CONTAINER);
    if (agent) {
      parts.push(`agent CPU: ${agent[1]}`, `RAM: ${agent.slice(2).join(" ")}`);
    } else {
      parts.push("agent stats unavailable (container not found)");
    }
    const others = rows
      .filter((fields) => fields[0] !== AGENT_CONTAINER)
      .map((fields) => ({
        name: fields[0],
        cpu: parseDockerPercent(fields[1]),
      }))
      .sort((a, b) => b.cpu - a.cpu);
    const othersTotal = others.reduce((sum, o) => sum + o.cpu, 0);
    const top = others
      .slice(0, 3)
      .map((o) => `${o.name} ${o.cpu.toFixed(0)}%`)
      .join(", ");
    parts.push(`other containers: ${othersTotal.toFixed(0)}% (${top})`);
  } catch (error: any) {
    parts.push(`docker stats unavailable (${error.message})`);
  }
  const after = readHostCpuJiffies();
  if (before && after && after.total > before.total) {
    const cores = os.cpus().length;
    const busy =
      ((after.busy - before.busy) / (after.total - before.total)) * 100 * cores;
    parts.push(`host: ${busy.toFixed(0)}% of ${cores * 100}% busy`);
  }
  if (process.env.STT_ACCEL && vramAvailable !== false) {
    try {
      const [vram, util] = execCommand(
        "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits",
      )
        .trim()
        .split("\n")[0]
        .split(",")
        .map((v) => v.trim());
      vramAvailable = true;
      parts.push(`VRAM: ${vram} MiB`, `GPU util: ${util}%`);
    } catch {
      vramAvailable = false;
    }
  }
  return `[${parts.join(" | ")}]`;
}

test.beforeAll(async () => {
  const fs = require("fs");
  const path = require("path");
  const audioFilePath = path.join(__dirname, "resources", "stt-test.wav");
  if (!fs.existsSync(audioFilePath) || fs.statSync(audioFilePath).size === 0) {
    await downloadFile(
      "https://s3.eu-west-1.amazonaws.com/public.openvidu.io/stt-test.wav",
      audioFilePath,
    );
  }
  LocalDeployment.stop();
});

test.describe("Transcribed tracks capacity probe", () => {
  const providerName = Object.keys(PROVIDER)[0];

  test.beforeEach(async () => {
    LocalDeployment.stop();
    await LocalDeployment.start("community", PROVIDER, undefined, "automatic");
  });

  test.afterEach(async ({}, testInfo) => {
    if (testInfo.status === "failed") {
      try {
        console.log("\n=== Docker logs for failed test ===\n");
        console.log(execCommand("docker logs agent-speech-processing"));
        console.log("\n=== End of Docker logs ===\n");
      } catch (error: any) {
        console.log("Failed to get docker logs:", error.message);
      }
    }
    LocalDeployment.stop();
  });

  test(`maximum simultaneous transcribed tracks with ${providerName}`, async ({
    page,
  }) => {
    test.setTimeout(RAMP_BUDGET_MS + 15 * 60 * 1000);

    await page.goto(TESTAPP_URL);
    // The testapp re-renders constantly with many instances; never let a
    // single UI action inherit the whole test budget.
    page.setDefaultTimeout(PARTICIPANT_ACTION_TIMEOUT_MS);

    let tracks = 0;
    let consecutiveFailures = 0;
    let stopReason = `hard cap of ${HARD_CAP_TRACKS} tracks reached`;
    const rampDeadline = Date.now() + RAMP_BUDGET_MS;

    console.log(`Baseline (0 tracks) ${sampleAgentLoad()}`);

    while (tracks < HARD_CAP_TRACKS) {
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        stopReason = `${MAX_CONSECUTIVE_FAILURES} consecutive tracks failed to be transcribed`;
        break;
      }
      if (Date.now() > rampDeadline) {
        stopReason = `ramp time budget of ${RAMP_BUDGET_MS / 60000} minutes exhausted`;
        break;
      }

      const roomName = `capacity-room-${Math.floor(tracks / PUBLISHERS_PER_ROOM)}`;
      let uid: number | undefined;
      try {
        uid = await addParticipantToRoom(page, roomName);
        await connectParticipant(page, uid);
        tracks++;
        consecutiveFailures = 0;
        console.log(
          `Capacity so far: ${tracks} simultaneous transcribed tracks ${sampleAgentLoad()}`,
        );
      } catch (error: any) {
        consecutiveFailures++;
        console.warn(
          `Track ${tracks + 1} failed (${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES} consecutive) ${sampleAgentLoad()}: ${error.message}`,
        );
        // Destroy the half-created instance so it cannot linger publishing an
        // unverified track that would skew the measurement.
        if (uid !== undefined) {
          try {
            await removeInstance(page, uid);
          } catch {
            // ignore
          }
        }
      }
    }

    // Sustained check: the FIRST participant must still be getting fresh
    // transcriptions now that the ramp is done, otherwise the count reflects
    // accepted-then-degraded tracks rather than sustained capacity.
    let sustained = false;
    if (tracks > 0) {
      // Count the first participant's OWN final transcriptions accumulated so
      // far, then require one more to arrive: proof that the oldest track is
      // still being transcribed with every other track live. Panel content is
      // rendered in the DOM even while collapsed, so :has-text() sees it.
      const ownFinalsOfFirst = page.locator(
        `#openvidu-instance-0 mat-accordion mat-expansion-panel` +
          `:has-text("finalTranscription"):has-text("TestParticipant0 said: ")`,
      );
      const before = await ownFinalsOfFirst.count();
      const deadline = Date.now() + TRACK_VERIFY_TIMEOUT_MS;
      while (!sustained && Date.now() < deadline) {
        sustained = (await ownFinalsOfFirst.count()) > before;
        await sleep(1);
      }
      if (!sustained) {
        console.warn(
          "Oldest track stopped receiving transcriptions at full load",
        );
      }
    }

    console.log(
      `CAPACITY RESULT: ${tracks} simultaneous transcribed tracks with ` +
        `${providerName} (model: ${PROVIDER[providerName].model}; stop reason: ${stopReason}; ` +
        `oldest track still transcribing at full load: ${sustained}) ${sampleAgentLoad()}`,
    );

    expect(tracks).toBeGreaterThan(0);
  });
});

async function getLastInstanceUid(page: Page): Promise<number> {
  const id = await page
    .locator("app-openvidu-instance")
    .last()
    .getAttribute("id");
  const uid = Number(id?.replace("openvidu-instance-", ""));
  if (!Number.isInteger(uid)) {
    throw new Error(`Could not determine uid of the new instance (id="${id}")`);
  }
  return uid;
}

/**
 * Add a brand-new participant instance configured as publisher-only,
 * audio-only with minimal event rendering (so the page stays responsive at
 * high instance counts), and return its stable uid.
 */
async function addParticipantToRoom(
  page: Page,
  roomName: string,
): Promise<number> {
  await page.click("#add-user-btn");
  const uid = await getLastInstanceUid(page);
  console.log(`Adding participant ${uid} to room "${roomName}"...`);

  await page.click(`#openvidu-instance-${uid} .subscriber-checkbox`);
  await page.click(`#room-options-btn-${uid}`);
  await page.waitForSelector("#video-capture-false", { state: "visible" });
  await page.click("#video-capture-false");
  await page.click("#close-dialog-btn");

  await page.click(`#room-events-btn-${uid}`);
  const turnOff = async (toggle: Locator) => {
    if ((await toggle.getAttribute("aria-checked")) === "true") {
      await toggle.dispatchEvent("click");
    }
  };
  await turnOff(
    page.getByRole("switch", { name: "Render interim transcription events" }),
  );
  const transcriptionReceived = page.getByRole("switch", {
    name: "transcriptionReceived",
    exact: true,
  });
  await turnOff(transcriptionReceived.nth(1));
  await turnOff(transcriptionReceived.nth(2));
  await page.click("#close-dialog-btn");

  await page
    .locator(`#openvidu-instance-${uid} #room-name-input-${uid}`)
    .fill(roomName, { timeout: PARTICIPANT_ACTION_TIMEOUT_MS });

  return uid;
}

/**
 * Connect a participant and verify its track is actually transcribed: resolves
 * only once the participant receives its OWN final transcription.
 */
async function connectParticipant(page: Page, uid: number): Promise<void> {
  const deadline = Date.now() + PARTICIPANT_ACTION_TIMEOUT_MS;
  while (
    !(await page
      .locator(`#openvidu-instance-${uid} .disconnect-btn`)
      .isVisible())
  ) {
    if (Date.now() > deadline) {
      throw new Error(
        `Participant ${uid} did not start connecting within ${PARTICIPANT_ACTION_TIMEOUT_MS}ms`,
      );
    }
    await page
      .locator(`#openvidu-instance-${uid} .connect-btn`)
      .dispatchEvent("click");
    await sleep(0.5);
  }
  await waitForEvent(page, "localTrackPublished", 1, uid, 60000);
  // A participant also receives OTHER participants' transcriptions, so the
  // verification must match this participant's own content prefix.
  await waitForEventContentToStartWith(
    page,
    "finalTranscription",
    `TestParticipant${uid} said: `,
    1,
    uid,
    TRACK_VERIFY_TIMEOUT_MS,
  );
  console.log(`Participant ${uid} connected and transcribed.`);
}

async function removeInstance(page: Page, uid: number): Promise<void> {
  const instance = page.locator(`#openvidu-instance-${uid}`);
  const deadline = Date.now() + PARTICIPANT_ACTION_TIMEOUT_MS;
  while ((await instance.count()) > 0) {
    if (Date.now() > deadline) {
      throw new Error(
        `Instance ${uid} was not removed within ${PARTICIPANT_ACTION_TIMEOUT_MS}ms`,
      );
    }
    try {
      await page
        .locator(`#openvidu-instance-${uid} .remove-instance-btn`)
        .dispatchEvent("click", {}, { timeout: 2000 });
    } catch {
      // Remove button not present/clickable yet (e.g. mid-render); retry.
    }
    await sleep(0.5);
  }
}
