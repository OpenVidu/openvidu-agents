import { test, Page } from "@playwright/test";
import { LocalDeployment } from "./utils/local-deployment";
import { TESTAPP_URL } from "./config";
import {
  downloadFile,
  execCommand,
  sleep,
  waitForEvent,
  waitForEventContentToStartWith,
} from "./utils/helper";

const LOCAL_STT_PROVIDERS = [
  {
    vosk: {
      model: "vosk-model-en-us-0.22-lgraph",
      use_silero_vad: false,
    },
    maxMemoryMB: 1500,
    maxTracks: 5, // Vosk has a limit of 5 simultaneous tracks
  },
  {
    sherpa: {
      model: "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
      use_silero_vad: false,
    },
    maxMemoryMB: 500,
  },
];

// Configuration
const ROOM_CREATION_TIMESPAN_SECONDS = 25; // Time window to create all rooms
const TOTAL_ROOMS = 4;
const TOTAL_PUBLISHERS_PER_ROOM = 2;
const MEMORY_CHECK_WAIT_SECONDS = 5; // Additional wait after room creation
const MEMORY_STABILITY_CHECK_INTERVAL_MS = 1500;
const MEMORY_STABILITY_THRESHOLD_PERCENT = 5; // Memory is stable if change is less than this percent
const MEMORY_STABILITY_CHECKS = 3; // Number of consecutive stable checks required

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

/**
 * Generate random delays for room/participant creation within a timespan
 */
function generateRandomDelays(
  count: number,
  timespanSeconds: number,
): number[] {
  const delays: number[] = [];
  for (let i = 0; i < count; i++) {
    delays.push(Math.random() * timespanSeconds * 1000);
  }
  return delays.sort((a, b) => a - b);
}

/**
 * Add a participant to the page and configure it for a specific room
 * Returns the instance index of the created participant
 */
async function addParticipantToRoom(
  page: Page,
  instanceIndex: number,
  roomName: string,
): Promise<void> {
  console.log(`Adding participant ${instanceIndex} to room "${roomName}"...`);

  // Add a new user instance
  await page.click("#add-user-btn");

  // Configure as publisher-only, audio-only
  await page.click(`#openvidu-instance-${instanceIndex} .subscriber-checkbox`);
  await page.click(`#room-options-btn-${instanceIndex}`);
  await page.waitForSelector("#video-capture-false", { state: "visible" });
  await page.click("#video-capture-false");
  await page.click("#close-dialog-btn");

  // Set the room name for this participant
  const roomNameInput = page.locator(
    `#openvidu-instance-${instanceIndex} #room-name-input-${instanceIndex}`,
  );
  await roomNameInput.clear();
  await roomNameInput.fill(roomName);
}

/**
 * Connect a participant by its instance index
 */
async function connectParticipant(
  page: Page,
  instanceIndex: number,
): Promise<void> {
  const connectButton = page.locator(
    `#openvidu-instance-${instanceIndex} .connect-btn`,
  );
  await connectButton.click();
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

test.describe("Memory usage tests for local STT providers", () => {
  LOCAL_STT_PROVIDERS.forEach((provider) => {
    const providerName = Object.keys(provider)[0] as string;

    test.describe(providerName, () => {
      test.beforeEach(async () => {
        LocalDeployment.stop();
        await LocalDeployment.start("community", provider);
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
        const containerName = "agent-speech-processing";
        const maxMemoryMB = provider.maxMemoryMB;

        // Step 1: Wait for baseline memory to stabilize
        console.log("Step 1: Waiting for baseline memory to stabilize...");
        const baselineMemory = await waitForStableMemory(containerName);
        console.log(`Baseline memory: ${formatBytes(baselineMemory)}`);

        // Step 2: Generate random schedule for participant creation
        // Respect maxTracks limit if defined for this provider
        const maxTracks = provider.maxTracks;
        const defaultTotalTracks = TOTAL_ROOMS * TOTAL_PUBLISHERS_PER_ROOM;
        const totalTracks = maxTracks
          ? Math.min(maxTracks, defaultTotalTracks)
          : defaultTotalTracks;

        console.log(
          `Step 2: Generating random schedule for ${totalTracks} tracks over ${ROOM_CREATION_TIMESPAN_SECONDS} seconds...`,
        );

        interface ParticipantTask {
          instanceIndex: number;
          roomName: string;
          delayMs: number;
        }

        const tasks: ParticipantTask[] = [];
        let instanceIndex = 0;
        let tracksCreated = 0;

        // Generate tasks for each room with random publishers, respecting maxTracks limit
        for (let roomIndex = 0; roomIndex < TOTAL_ROOMS; roomIndex++) {
          if (tracksCreated >= totalTracks) break;

          const roomName = `memory-test-room-${roomIndex}`;

          for (let p = 0; p < TOTAL_PUBLISHERS_PER_ROOM; p++) {
            if (tracksCreated >= totalTracks) break;

            tasks.push({
              instanceIndex: instanceIndex++,
              roomName,
              delayMs: 0, // Will be assigned below
            });
            tracksCreated++;
          }
        }

        // Assign random delays to all participant tasks
        const delays = generateRandomDelays(
          tasks.length,
          ROOM_CREATION_TIMESPAN_SECONDS,
        );
        tasks.forEach((task, i) => {
          task.delayMs = delays[i];
        });

        // Sort by delay for sequential execution
        tasks.sort((a, b) => a.delayMs - b.delayMs);

        // Normalize delays so the first participant connects at time 0
        const firstDelay = tasks[0].delayMs;
        tasks.forEach((task) => {
          task.delayMs -= firstDelay;
        });

        console.log("Participant creation schedule:");
        tasks.forEach((task) => {
          console.log(
            `  Participant ${task.instanceIndex} in room "${task.roomName}" at ${(task.delayMs / 1000).toFixed(2)}s`,
          );
        });

        // Navigate to testapp
        await page.goto(TESTAPP_URL);

        // Step 3: Create all participants first (add them to the page)
        console.log("Step 3: Adding all participants to the page...");
        for (const task of tasks) {
          await addParticipantToRoom(page, task.instanceIndex, task.roomName);
        }

        // Step 4: Connect participants according to random schedule
        console.log("Step 4: Connecting participants according to schedule...");
        const startTime = Date.now();

        for (const task of tasks) {
          const elapsedMs = Date.now() - startTime;
          const waitMs = Math.max(0, task.delayMs - elapsedMs);

          if (waitMs > 0) {
            await sleep(waitMs / 1000);
          }

          try {
            await connectParticipant(page, task.instanceIndex);
          } catch (error: any) {
            console.error(
              `Error connecting participant ${task.instanceIndex}: ${error.message}`,
            );
          }
        }

        // Wait for remaining time in the timespan
        const totalElapsed = Date.now() - startTime;
        const remainingTimespan =
          ROOM_CREATION_TIMESPAN_SECONDS * 1000 - totalElapsed;
        if (remainingTimespan > 0) {
          console.log(
            `Waiting ${(remainingTimespan / 1000).toFixed(2)}s for timespan to complete...`,
          );
          await sleep(remainingTimespan / 1000);
        }

        // Step 5: Wait additional time for memory to settle
        console.log(
          `Step 5: Waiting ${MEMORY_CHECK_WAIT_SECONDS} additional seconds for memory to settle...`,
        );
        await sleep(MEMORY_CHECK_WAIT_SECONDS);

        // Step 6: Check final memory usage
        console.log("Step 6: Checking final memory usage...");
        const finalStats = getContainerMemoryStats(containerName);
        const finalMemory = finalStats.usedBytes;
        const finalMemoryMB = finalMemory / (1024 * 1024);
        const maxMemoryBytes = maxMemoryMB * 1024 * 1024;

        console.log(`Baseline memory: ${formatBytes(baselineMemory)}`);
        console.log(`Final memory: ${formatBytes(finalMemory)}`);
        console.log(`Maximum allowed for ${providerName}: ${maxMemoryMB} MiB`);

        // Assert memory usage does not exceed absolute limit
        if (finalMemory > maxMemoryBytes) {
          throw new Error(
            `Memory usage exceeded ${maxMemoryMB} MiB limit for ${providerName}!\n` +
              `Baseline: ${formatBytes(baselineMemory)}\n` +
              `Final: ${formatBytes(finalMemory)} (${finalMemoryMB.toFixed(2)} MiB)\n` +
              `Limit: ${maxMemoryMB} MiB`,
          );
        }

        console.log(
          `✓ Memory usage is within acceptable limits (${finalMemoryMB.toFixed(2)} MiB < ${maxMemoryMB} MiB)`,
        );
      });
    });
  });
});
