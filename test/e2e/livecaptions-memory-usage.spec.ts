import { test, Page, Locator } from "@playwright/test";
import { RoomServiceClient } from "livekit-server-sdk";
import { LocalDeployment } from "./utils/local-deployment";
import {
  TESTAPP_URL,
  LIVEKIT_URL_HTTP,
  LIVEKIT_API_KEY,
  LIVEKIT_API_SECRET,
} from "./config";
import {
  downloadFile,
  execCommand,
  sleep,
  waitForEvent,
  waitForEventContentToStartWith,
} from "./utils/helper";

// LiveKit server API, used to force-tear-down rooms and participants during the
// leak-check teardown (see Step 11).
const roomServiceClient = new RoomServiceClient(
  LIVEKIT_URL_HTTP,
  LIVEKIT_API_KEY,
  LIVEKIT_API_SECRET,
);

interface SttProviderConfig {
  [providerName: string]: any;
  maxMemoryMB: number;
  maxTracks?: number;
  // Teardown memory cap. Either an absolute value in MiB (a number) or a
  // percentage string (e.g. "10%") applied on top of the idle baseline
  // captured at the start of the test (i.e. the container may sit up to that
  // percentage above idle after teardown).
  maxMemoryAfterTeardownMB: number | string;
}

const LOCAL_STT_PROVIDERS: SttProviderConfig[] = [
  {
    vosk: {
      model: "vosk-model-en-us-0.22-lgraph",
      use_silero_vad: false,
    },
    maxMemoryMB: 1500,
    maxTracks: 8,
    maxMemoryAfterTeardownMB: 350,
  },
  {
    sherpa: {
      model: "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
      use_silero_vad: false,
    },
    maxMemoryMB: 600,
    maxTracks: 12,
    maxMemoryAfterTeardownMB: 400,
  },
  {
    vosk: {
      model: "vosk-model-en-us-0.22-lgraph",
      use_silero_vad: true,
    },
    maxMemoryMB: 2000,
    maxTracks: 8,
    maxMemoryAfterTeardownMB: 350,
  },
  {
    sherpa: {
      model: "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
      use_silero_vad: true,
    },
    maxMemoryMB: 1600,
    maxTracks: 12,
    maxMemoryAfterTeardownMB: 600,
  },
];

const CLOUD_STT_PROVIDERS: SttProviderConfig[] = [
  {
    azure: {
      speech_key: process.env.AZURE_SPEECH_KEY,
      speech_region: process.env.AZURE_SPEECH_REGION,
    },
    // Cloud providers offload transcription, so the local container footprint
    // is small. These limits may need calibration against real measurements.
    maxMemoryMB: 400,
    maxTracks: 12,
    maxMemoryAfterTeardownMB: "25%",
  },
  {
    aws: {
      aws_access_key_id: process.env.AWS_ACCESS_KEY_ID,
      aws_secret_access_key: process.env.AWS_SECRET_ACCESS_KEY,
      aws_default_region: process.env.AWS_DEFAULT_REGION,
    },
    maxMemoryMB: 400,
    maxTracks: 12,
    maxMemoryAfterTeardownMB: "25%",
  },
];

// Configuration
const PARTICIPANT_ACTION_TIMEOUT_MS = 10000;
const TOTAL_ROOMS = 5;
const TOTAL_PUBLISHERS_PER_ROOM = 3;
const MEMORY_STABILITY_CHECK_INTERVAL_MS = 1500;
const MEMORY_STABILITY_THRESHOLD_PERCENT = 5; // Memory is stable if change is less than this percent
const MEMORY_STABILITY_CHECKS = 3; // Number of consecutive stable checks required

const SOAK_DURATION_MS = 60 * 3 * 1000;
const SOAK_MEMORY_LOG_INTERVAL_MS = 20 * 1000; // sample container memory 3 times a minute
const SOAK_OPERATION_PAUSE_SECONDS = 0.5; // small pause between churn operations

// Leak detection (teardown phase): after destroying everything, CPU should
// return close to its idle baseline, and memory must drop below each provider's
// maxMemoryAfterTeardownMB cap (an absolute MiB value, or a percentage above the
// idle baseline). Allocators rarely release memory back to the OS exactly, so
// the cap sits above the idle baseline.
const LEAK_SETTLE_MAX_WAIT_MS = 120000; // Max time to wait for return to baseline
const LEAK_SETTLE_CHECK_INTERVAL_MS = 2000;
const LEAK_SETTLE_STABLE_CHECKS = 3; // Consecutive checks at baseline required
const CPU_IDLE_TOLERANCE_PERCENT = 5; // Allowed CPU above baseline after teardown

interface MemoryStats {
  usedBytes: number;
  limitBytes: number;
  percentUsed: number;
}

/**
 * Get memory stats for a Docker container using docker stats API
 */
function getContainerMemoryStats(containerName: string): MemoryStats {
  try {
    // Use docker stats with --no-stream to get a single snapshot
    // Format: {{.MemUsage}} gives us "X MiB / Y MiB" format
    const output = execCommand(
      `docker stats ${containerName} --no-stream --format "{{.MemUsage}}"`,
    );

    // Parse output like "123.4MiB / 7.773GiB" or "1.234GiB / 7.773GiB"
    const match = output.match(
      /^([\d.]+)([A-Za-z]+)\s*\/\s*([\d.]+)([A-Za-z]+)$/,
    );
    if (!match) {
      throw new Error(`Unexpected memory format: ${output}`);
    }

    const usedValue = parseFloat(match[1]);
    const usedUnit = match[2].toLowerCase();
    const limitValue = parseFloat(match[3]);
    const limitUnit = match[4].toLowerCase();

    const usedBytes = convertToBytes(usedValue, usedUnit);
    const limitBytes = convertToBytes(limitValue, limitUnit);

    return {
      usedBytes,
      limitBytes,
      percentUsed: (usedBytes / limitBytes) * 100,
    };
  } catch (error: any) {
    console.error(
      `Error getting memory stats for ${containerName}:`,
      error.message,
    );
    throw error;
  }
}

/**
 * Get the current CPU usage percentage for a Docker container
 */
function getContainerCpuStats(containerName: string): number {
  try {
    const output = execCommand(
      `docker stats ${containerName} --no-stream --format "{{.CPUPerc}}"`,
    );

    // Parse output like "12.34%"
    const match = output.match(/^([\d.]+)%$/);
    if (!match) {
      throw new Error(`Unexpected CPU format: ${output}`);
    }

    return parseFloat(match[1]);
  } catch (error: any) {
    console.error(
      `Error getting CPU stats for ${containerName}:`,
      error.message,
    );
    throw error;
  }
}

/**
 * Convert memory value to bytes based on unit
 */
function convertToBytes(value: number, unit: string): number {
  const units: Record<string, number> = {
    b: 1,
    kib: 1024,
    kb: 1000,
    mib: 1024 * 1024,
    mb: 1000 * 1000,
    gib: 1024 * 1024 * 1024,
    gb: 1000 * 1000 * 1000,
  };

  const multiplier = units[unit];
  if (!multiplier) {
    throw new Error(`Unknown memory unit: ${unit}`);
  }

  return value * multiplier;
}

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
  } else if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
  } else if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(2)} KiB`;
  }
  return `${bytes} B`;
}

/**
 * Resolve the teardown memory threshold (in bytes) from a provider's
 * maxMemoryAfterTeardownMB setting.
 *
 * - A number is treated as an absolute cap in MiB.
 * - A percentage string like "10%" is applied on top of the idle baseline
 *   captured at the start of the test: the container may sit up to that
 *   percentage above its idle memory after teardown.
 */
function resolveTeardownThresholdBytes(
  maxMemoryAfterTeardownMB: number | string,
  baselineMemoryBytes: number,
): number {
  if (typeof maxMemoryAfterTeardownMB === "number") {
    return maxMemoryAfterTeardownMB * 1024 * 1024;
  }

  const match = maxMemoryAfterTeardownMB.trim().match(/^([\d.]+)%$/);
  if (!match) {
    throw new Error(
      `Invalid maxMemoryAfterTeardownMB value "${maxMemoryAfterTeardownMB}": ` +
        `expected a number (absolute MiB) or a percentage string like "10%".`,
    );
  }

  const percent = parseFloat(match[1]);
  return baselineMemoryBytes * (1 + percent / 100);
}

/**
 * Wait for memory to stabilize and return the baseline
 */
async function waitForStableMemory(
  containerName: string,
  maxWaitMs: number = 60000,
): Promise<number> {
  console.log(
    `Waiting for memory to stabilize in container ${containerName}...`,
  );

  const deadline = Date.now() + maxWaitMs;
  let stableCount = 0;
  let previousMemory = 0;

  while (Date.now() < deadline) {
    const stats = getContainerMemoryStats(containerName);
    const currentMemory = stats.usedBytes;

    if (previousMemory > 0) {
      const changePercent =
        (Math.abs(currentMemory - previousMemory) / previousMemory) * 100;

      if (changePercent < MEMORY_STABILITY_THRESHOLD_PERCENT) {
        stableCount++;
        console.log(
          `Memory stable check ${stableCount}/${MEMORY_STABILITY_CHECKS}: ${formatBytes(currentMemory)} (change: ${changePercent.toFixed(2)}%)`,
        );

        if (stableCount >= MEMORY_STABILITY_CHECKS) {
          console.log(`Memory stabilized at ${formatBytes(currentMemory)}`);
          return currentMemory;
        }
      } else {
        stableCount = 0;
        console.log(
          `Memory not stable: ${formatBytes(currentMemory)} (change: ${changePercent.toFixed(2)}%)`,
        );
      }
    }

    previousMemory = currentMemory;
    await sleep(MEMORY_STABILITY_CHECK_INTERVAL_MS / 1000);
  }

  // Return current memory even if not fully stable
  const finalStats = getContainerMemoryStats(containerName);
  console.warn(
    `Memory did not fully stabilize within ${maxWaitMs}ms. Using current value: ${formatBytes(finalStats.usedBytes)}`,
  );
  return finalStats.usedBytes;
}

interface LeakCheckResult {
  memoryBytes: number;
  cpuPercent: number;
  returnedToBaseline: boolean;
}

/**
 * After teardown, wait until the container's CPU drops back to its idle
 * baseline and its memory drops below the provider's explicit teardown cap.
 * Used to detect leaks.
 */
async function waitForResourcesToReturnToBaseline(
  containerName: string,
  baselineMemoryBytes: number,
  baselineCpuPercent: number,
  memoryThreshold: number,
  maxWaitMs: number = LEAK_SETTLE_MAX_WAIT_MS,
): Promise<LeakCheckResult> {
  const cpuThreshold = baselineCpuPercent + CPU_IDLE_TOLERANCE_PERCENT;

  console.log(
    `Waiting for resources to return to baseline (memory <= ${formatBytes(memoryThreshold)}, CPU <= ${cpuThreshold.toFixed(2)}%)...`,
  );

  const deadline = Date.now() + maxWaitMs;
  let stableCount = 0;
  let currentMemory = baselineMemoryBytes;
  let currentCpu = baselineCpuPercent;

  while (Date.now() < deadline) {
    currentMemory = getContainerMemoryStats(containerName).usedBytes;
    currentCpu = getContainerCpuStats(containerName);

    const memoryOk = currentMemory <= memoryThreshold;
    const cpuOk = currentCpu <= cpuThreshold;

    if (memoryOk && cpuOk) {
      stableCount++;
      console.log(
        `At baseline check ${stableCount}/${LEAK_SETTLE_STABLE_CHECKS}: memory ${formatBytes(currentMemory)}, CPU ${currentCpu.toFixed(2)}%`,
      );
      if (stableCount >= LEAK_SETTLE_STABLE_CHECKS) {
        return {
          memoryBytes: currentMemory,
          cpuPercent: currentCpu,
          returnedToBaseline: true,
        };
      }
    } else {
      stableCount = 0;
      console.log(
        `Not at baseline yet: memory ${formatBytes(currentMemory)} (${memoryOk ? "ok" : "high"}), CPU ${currentCpu.toFixed(2)}% (${cpuOk ? "ok" : "high"})`,
      );
    }

    await sleep(LEAK_SETTLE_CHECK_INTERVAL_MS / 1000);
  }

  console.warn(
    `Resources did not return to baseline within ${maxWaitMs}ms. Memory: ${formatBytes(currentMemory)}, CPU: ${currentCpu.toFixed(2)}%`,
  );
  return {
    memoryBytes: currentMemory,
    cpuPercent: currentCpu,
    returnedToBaseline: false,
  };
}

/**
 * Read the stable uid of the most recently added instance (the last
 * app-openvidu-instance element), parsed from its "openvidu-instance-<uid>" id.
 */
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
 * Add a brand-new participant instance, configure it (publisher-only, audio-only,
 * minimal event rendering) for the given room, and return its stable uid.
 */
async function addParticipantToRoom(
  page: Page,
  roomName: string,
): Promise<number> {
  // Add a new user instance and read its stable uid
  await page.click("#add-user-btn");
  const uid = await getLastInstanceUid(page);
  console.log(`Adding participant ${uid} to room "${roomName}"...`);

  // Configure as publisher-only, audio-only
  await page.click(`#openvidu-instance-${uid} .subscriber-checkbox`);
  await page.click(`#room-options-btn-${uid}`);
  await page.waitForSelector("#video-capture-false", { state: "visible" });
  await page.click("#video-capture-false");
  await page.click("#close-dialog-btn");

  // Open the "Room events" dialog of this instance to tweak event toggles
  await page.click(`#room-events-btn-${uid}`);
  const turnOff = async (toggle: Locator) => {
    if ((await toggle.getAttribute("aria-checked")) === "true") {
      await toggle.dispatchEvent("click");
    }
  };

  // Do not render interim transcription events
  await turnOff(
    page.getByRole("switch", { name: "Render interim transcription events" }),
  );

  // Keep "transcriptionReceived" only for the RoomEvent list
  const transcriptionReceived = page.getByRole("switch", {
    name: "transcriptionReceived",
    exact: true,
  });
  await turnOff(transcriptionReceived.nth(1));
  await turnOff(transcriptionReceived.nth(2));

  await page.click("#close-dialog-btn");

  // Set the room name for this participant
  await setParticipantRoom(page, uid, roomName);

  return uid;
}

/**
 * Change the room an existing participant instance will join on its next
 * connect. Used by the soak phase so each reconnect targets a brand-new room.
 */
async function setParticipantRoom(
  page: Page,
  instanceIndex: number,
  roomName: string,
): Promise<void> {
  const roomNameInput = page.locator(
    `#openvidu-instance-${instanceIndex} #room-name-input-${instanceIndex}`,
  );
  await roomNameInput.fill(roomName, {
    timeout: PARTICIPANT_ACTION_TIMEOUT_MS,
  });
}

/**
 * Whether a participant is currently connected, judged from the live UI state
 */
async function isParticipantConnected(
  page: Page,
  instanceIndex: number,
): Promise<boolean> {
  return await page
    .locator(`#openvidu-instance-${instanceIndex} .disconnect-btn`)
    .isVisible();
}

/**
 * Trigger a button inside an instance panel with a direct DOM click event.
 */
async function dispatchInstanceButtonClick(
  page: Page,
  instanceIndex: number,
  buttonClass: string,
): Promise<void> {
  await page
    .locator(`#openvidu-instance-${instanceIndex} ${buttonClass}`)
    .dispatchEvent("click");
}

/**
 * Connect a participant by its instance index
 */
async function connectParticipant(
  page: Page,
  instanceIndex: number,
): Promise<void> {
  // Dispatch the connect click, retrying until the UI reflects the connected
  // state, in case the saturated page dropped the first event.
  const deadline = Date.now() + PARTICIPANT_ACTION_TIMEOUT_MS;
  while (!(await isParticipantConnected(page, instanceIndex))) {
    if (Date.now() > deadline) {
      throw new Error(
        `Participant ${instanceIndex} did not start connecting within ${PARTICIPANT_ACTION_TIMEOUT_MS}ms`,
      );
    }
    await dispatchInstanceButtonClick(page, instanceIndex, ".connect-btn");
    await sleep(0.5);
  }
  await waitForEvent(page, "localTrackPublished", 1, instanceIndex, 60000);
  await waitForEventContentToStartWith(
    page,
    "finalTranscription",
    `TestParticipant${instanceIndex} said: `,
    1,
    instanceIndex,
    60000,
  );
  console.log(
    `Participant ${instanceIndex} connected, publishing audio track and receiving own transcriptions.`,
  );
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
  console.log(`Participant ${uid} removed.`);
}

test.beforeAll(async () => {
  const fs = require("fs");
  const path = require("path");
  const audioFilePath = path.join(__dirname, "resources", "stt-test.wav");

  // Download test audio file if needed
  if (!fs.existsSync(audioFilePath) || fs.statSync(audioFilePath).size === 0) {
    await downloadFile(
      "https://s3.eu-west-1.amazonaws.com/public.openvidu.io/stt-test.wav",
      audioFilePath,
    );
  }
  LocalDeployment.stop();
});

/**
 * Register the memory-usage test for a single STT provider (local or cloud).
 *
 * @param provider       the STT provider configuration to soak.
 * @param soakDurationMs how long the chaos soak (Step 2) runs. Defaults to the
 *                       short SOAK_DURATION_MS; the long-running variant passes
 *                       LONG_SOAK_DURATION_MS.
 */
function registerProviderMemoryTest(
  provider: SttProviderConfig,
  soakDurationMs: number = SOAK_DURATION_MS,
) {
  const providerName = Object.keys(provider).find(
    (key) => key !== "maxMemoryMB" && key !== "maxTracks",
  ) as string;

  const providerConfig = provider[providerName];
  const useVad = providerConfig?.use_silero_vad;
  const testLabel =
    typeof useVad === "boolean"
      ? `${providerName} (VAD ${useVad ? "enabled" : "disabled"})`
      : providerName;

  test.describe(testLabel, () => {
    test.beforeEach(async () => {
      LocalDeployment.stop();
      await LocalDeployment.start(
        "community",
        provider,
        undefined,
        "automatic",
      );
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

    test(`memory usage should not exceed provider limit with ${providerName}`, async ({
      page,
    }) => {
      // The chaos soak runs for soakDurationMs, far longer than the
      // global per-test timeout, so extend it (with margin for setup/teardown).
      test.setTimeout(soakDurationMs + 30 * 60 * 1000);

      const containerName = "agent-speech-processing";

      // Step 1: Wait for baseline memory to stabilize and capture idle CPU
      console.log("Step 1: Waiting for baseline memory to stabilize...");
      const baselineMemory = await waitForStableMemory(containerName);
      const baselineCpu = getContainerCpuStats(containerName);
      console.log(
        `Idle baseline: memory ${formatBytes(baselineMemory)}, CPU ${baselineCpu.toFixed(2)}%`,
      );

      // Cap on simultaneous tracks (= simultaneous connected participants).
      const maxTracks =
        provider.maxTracks ?? TOTAL_ROOMS * TOTAL_PUBLISHERS_PER_ROOM;

      await page.goto(TESTAPP_URL);

      // Step 2: Chaos soak. For soakDurationMs, randomly add and remove participants,
      // never exceeding maxTracks simultaneous tracks. The result is a shifting mix of rooms
      // and participants coming and going. The agent container memory is sampled regularly
      console.log(
        `Step 2: Starting ${(soakDurationMs / 60000).toFixed(1)}min chaos soak (max ${maxTracks} simultaneous tracks)...`,
      );

      interface ActiveParticipant {
        uid: number;
        roomName: string;
      }
      const active: ActiveParticipant[] = [];
      let roomCounter = 0; // source of brand-new room names
      const soakStartTime = Date.now();
      let lastMemoryLogTime = 0;

      const logSoakMemory = () => {
        const memoryBytes = getContainerMemoryStats(containerName).usedBytes;
        const elapsedMin = ((Date.now() - soakStartTime) / 60000).toFixed(1);
        const activeRooms = new Set(active.map((a) => a.roomName)).size;
        console.log(
          `[soak ${elapsedMin}min] agent memory: ${formatBytes(memoryBytes)} | tracks: ${active.length}/${maxTracks} | active rooms: ${activeRooms} | rooms created: ${roomCounter}`,
        );
        lastMemoryLogTime = Date.now();
      };

      while (Date.now() - soakStartTime < soakDurationMs) {
        if (Date.now() - lastMemoryLogTime >= SOAK_MEMORY_LOG_INTERVAL_MS) {
          logSoakMemory();
        }

        // Choose the next action, keeping 0 <= active <= maxTracks. Bias
        // slightly towards adding so the population stays lively.
        const canAdd = active.length < maxTracks;
        const canRemove = active.length > 0;
        const add = canAdd && (!canRemove || Math.random() < 0.6);

        if (add) {
          // Sometimes join an existing active room (making it a multi-participant
          // room), sometimes create a brand-new one.
          const activeRooms = [...new Set(active.map((a) => a.roomName))];
          const roomName =
            activeRooms.length > 0 && Math.random() < 0.5
              ? activeRooms[Math.floor(Math.random() * activeRooms.length)]
              : `chaos-room-${roomCounter++}`;
          let uid: number | undefined;
          try {
            uid = await addParticipantToRoom(page, roomName);
            await connectParticipant(page, uid);
            active.push({ uid, roomName });
            console.log(
              `  + participant ${uid} -> room "${roomName}" (${active.length}/${maxTracks} tracks)`,
            );
          } catch (error: any) {
            console.error(`Error adding participant: ${error.message}`);
            // Best-effort: destroy a half-created instance so it cannot linger
            // holding a track (which would break the maxTracks cap).
            if (uid !== undefined) {
              try {
                await removeInstance(page, uid);
              } catch {
                // ignore
              }
            }
          }
        } else if (canRemove) {
          // Remove a random active participant, destroying its instance.
          const index = Math.floor(Math.random() * active.length);
          const victim = active[index];
          try {
            await removeInstance(page, victim.uid);
            active.splice(index, 1);
            console.log(
              `  - participant ${victim.uid} from room "${victim.roomName}" (${active.length}/${maxTracks} tracks)`,
            );
          } catch (error: any) {
            // Leave it tracked so we don't undercount tracks; retried later.
            console.error(
              `Error removing participant ${victim.uid}: ${error.message}`,
            );
          }
        }

        await sleep(SOAK_OPERATION_PAUSE_SECONDS);
      }

      logSoakMemory();
      console.log(
        `Chaos soak finished after ${((Date.now() - soakStartTime) / 60000).toFixed(1)} minutes; ${roomCounter} rooms created.`,
      );

      // Step 3: Tear everything down from the server side via the LiveKit server API
      console.log(
        "Step 3: Removing all participants via the LiveKit server API...",
      );
      const rooms = await roomServiceClient.listRooms();
      for (const room of rooms) {
        const participants = await roomServiceClient.listParticipants(
          room.name,
        );
        for (const participant of participants) {
          try {
            await roomServiceClient.removeParticipant(
              room.name,
              participant.identity,
            );
            console.log(
              `  Removed participant "${participant.identity}" from room "${room.name}"`,
            );
          } catch (error: any) {
            console.error(
              `Error removing participant "${participant.identity}" from room "${room.name}": ${error.message}`,
            );
          }
        }
      }

      // Ensure no room stays active: delete any that are still listed
      const remainingRooms = await roomServiceClient.listRooms();
      for (const room of remainingRooms) {
        try {
          await roomServiceClient.deleteRoom(room.name);
          console.log(`  Deleted room "${room.name}"`);
        } catch (error: any) {
          console.error(`Error deleting room "${room.name}": ${error.message}`);
        }
      }

      active.length = 0;
      console.log("All participants and rooms removed.");

      // Step 4: Wait for the agent container to return to its idle baseline.
      // If memory or CPU never drops back, it indicates a leak.
      console.log(
        "Step 4: Waiting for agent memory and CPU to return to idle baseline (leak check)...",
      );
      const memoryThresholdBytes = resolveTeardownThresholdBytes(
        provider.maxMemoryAfterTeardownMB,
        baselineMemory,
      );
      const settled = await waitForResourcesToReturnToBaseline(
        containerName,
        baselineMemory,
        baselineCpu,
        memoryThresholdBytes,
        LEAK_SETTLE_MAX_WAIT_MS,
      );

      console.log(
        `Idle baseline: memory ${formatBytes(baselineMemory)}, CPU ${baselineCpu.toFixed(2)}%`,
      );
      console.log(
        `After teardown: memory ${formatBytes(settled.memoryBytes)}, CPU ${settled.cpuPercent.toFixed(2)}%`,
      );

      if (!settled.returnedToBaseline) {
        throw new Error(
          `Possible memory/CPU leak with ${providerName}! ` +
            `Resources did not return to idle baseline after destroying all tracks, participants, agents and rooms.\n` +
            `Idle baseline: memory ${formatBytes(baselineMemory)}, CPU ${baselineCpu.toFixed(2)}%\n` +
            `After teardown: memory ${formatBytes(settled.memoryBytes)}, CPU ${settled.cpuPercent.toFixed(2)}%\n` +
            `Allowed: memory up to ${formatBytes(memoryThresholdBytes)}` +
            (typeof provider.maxMemoryAfterTeardownMB === "string"
              ? ` (${provider.maxMemoryAfterTeardownMB} above idle baseline)`
              : "") +
            `, CPU up to baseline + ${CPU_IDLE_TOLERANCE_PERCENT}%`,
        );
      }

      console.log(
        `✓ No leak detected: agent returned to idle baseline for ${providerName}`,
      );
    });
  });
}

test.describe("Memory usage tests for local STT providers", () => {
  LOCAL_STT_PROVIDERS.forEach((provider) => {
    registerProviderMemoryTest(provider);
  });
});

test.describe("Memory usage tests for cloud STT providers", () => {
  CLOUD_STT_PROVIDERS.forEach((provider) => {
    registerProviderMemoryTest(provider);
  });
});

// const LONG_SOAK_DURATION_MS = 60 * 60 * 1000; // 1 hour
// const SOAK_PROVIDER = LOCAL_STT_PROVIDERS.find(
//   (p) => p.vosk?.use_silero_vad === false,
// )!;
// test.describe(`Memory leak soak test (long-running, ${(LONG_SOAK_DURATION_MS / 60000).toFixed(0)} min)`, () => {
//   registerProviderMemoryTest(SOAK_PROVIDER, LONG_SOAK_DURATION_MS);
// });
