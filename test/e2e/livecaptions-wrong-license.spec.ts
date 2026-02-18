import { test, expect } from "@playwright/test";
import { execCommand, sleep } from "./utils/helper";
import { LocalDeployment } from "./utils/local-deployment";

/**
 * Check if a container is running and healthy
 */
function isContainerRunning(containerName: string): boolean {
  try {
    const status = execCommand(
      `docker inspect ${containerName} -f {{.State.Status}}`,
    ).trim();
    return status === "running";
  } catch (error) {
    return false;
  }
}

/**
 * Get container exit code
 */
function getContainerExitCode(containerName: string): number {
  try {
    const exitCode = execCommand(
      `docker inspect ${containerName} -f {{.State.ExitCode}}`,
    ).trim();
    return parseInt(exitCode, 10);
  } catch (error) {
    return -1;
  }
}

/**
 * Get container logs
 */
function getContainerLogs(containerName: string): string {
  try {
    return execCommand(`docker logs ${containerName} 2>&1`);
  } catch (error: any) {
    return `Failed to get logs: ${error.message}`;
  }
}

/**
 * Wait for container to enter any failure state
 * Returns the failure state found, or null if timeout
 */
async function waitForContainerFailure(
  containerName: string,
  timeoutSeconds: number = 60,
): Promise<string | null> {
  const startTime = Date.now();
  const timeoutMs = timeoutSeconds * 1000;
  const failureStates = ["restarting", "dead"];

  while (Date.now() - startTime < timeoutMs) {
    try {
      const status = execCommand(
        `docker inspect ${containerName} -f {{.State.Status}}`,
      ).trim();

      // Check for explicit failure states
      if (failureStates.includes(status)) {
        console.log(
          `Container ${containerName} entered failure state: ${status}`,
        );
        return status;
      }

      // Check for exited state with non-zero exit code
      if (status === "exited") {
        const exitCode = getContainerExitCode(containerName);
        if (exitCode !== 0) {
          console.log(
            `Container ${containerName} exited with non-zero exit code: ${exitCode}`,
          );
          return "exited";
        }
      }
    } catch (error) {
      // Container might not exist yet
    }
    await sleep(1);
  }
  return null;
}

test.describe("Wrong license tests", () => {
  test.afterEach(() => {
    LocalDeployment.stop();
  });

  test("agent-speech-processing container should fail with wrong license", async () => {
    // Stop any existing deployment
    LocalDeployment.stop();

    // Configure with a wrong license and start deployment
    const wrongLicense = "WRONG_LICENSE_12345";
    const sherpaProvider = {
      sherpa: {
        model: "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
        use_silero_vad: false,
      },
    };
    await LocalDeployment.start("pro", sherpaProvider, wrongLicense);

    // Wait a bit for containers to start
    await sleep(5);

    // Check that agent-speech-processing container enters any failure state
    // Possible failure states: restarting (failed + auto-restart), exited (non-zero), dead
    const containerName = "agent-speech-processing";

    console.log(
      `Waiting for container ${containerName} to enter a failure state...`,
    );
    const failureState = await waitForContainerFailure(containerName, 60);
    expect(failureState).not.toBeNull();
    console.log(`Container failed with state: ${failureState}`);

    // Verify logs contain license error message
    // Loop until the container logs are NOT empty
    let logs = "";
    const maxLogWaitSeconds = 20;
    const logStartTime = Date.now();
    while (
      logs.trim() === "" &&
      Date.now() - logStartTime < maxLogWaitSeconds * 1000
    ) {
      logs = getContainerLogs(containerName);
      if (logs.trim() === "") {
        await sleep(0.3);
      }
    }
    if (logs.trim() === "") {
      throw new Error(
        `No logs found for container ${containerName} after waiting for ${maxLogWaitSeconds} seconds`,
      );
    }
    expect(logs).toContain(`License ${wrongLicense} does not exist`);
  });
});
