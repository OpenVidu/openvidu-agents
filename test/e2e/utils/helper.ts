import { execSync, spawn } from "child_process";
import { Page, ElementHandle } from "@playwright/test";

export const execCommand = (command: string): string => {
  try {
    return execSync(command).toString().trim();
  } catch (error) {
    console.error(`Error executing command: ${command}`);
    console.error(error);
    throw error;
  }
};

export const execCommandInBackground = (
  command: string,
  args: string[],
): number | undefined => {
  const child = spawn(command, args, { detached: true });

  child.stdout.on("data", (data) => {
    console.log(`stdout (${command}): ${data}`);
  });
  child.stderr.on("data", (data) => {
    console.log(`stderr (${command}): ${data}`);
  });
  child.on("close", (code) => {
    console.log(`child process (${command}) exited with code ${code}`);
  });
  child.on("error", (error) => {
    console.error(`child process (${command}) error: ${error}`);
    throw error;
  });

  return child.pid;
};

export const killProcess = (pid: number) => {
  process.kill(pid);
};

export const sleep = async (seconds: number) => {
  return new Promise((resolve) => {
    setTimeout(resolve, seconds * 1000);
  });
};

export const downloadFile = async (url: string, destination: string) => {
  const path = require("path");
  const https = require("https");
  const http = require("http");
  const fs = require("fs");

  const dir = path.dirname(destination);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  return new Promise((resolve, reject) => {
    const downloadWithRedirect = (downloadUrl: string, maxRedirects = 5) => {
      const client = downloadUrl.startsWith("https:") ? https : http;

      const request = client.get(downloadUrl, (response: any) => {
        // Handle redirects
        if (
          response.statusCode >= 300 &&
          response.statusCode < 400 &&
          response.headers.location
        ) {
          if (maxRedirects > 0) {
            console.log(`Redirecting to: ${response.headers.location}`);
            return downloadWithRedirect(
              response.headers.location,
              maxRedirects - 1,
            );
          } else {
            return reject(new Error("Too many redirects"));
          }
        }

        // Check for successful response
        if (response.statusCode !== 200) {
          return reject(
            new Error(`HTTP ${response.statusCode}: ${response.statusMessage}`),
          );
        }

        // Check content length
        const contentLength = parseInt(
          response.headers["content-length"] || "0",
        );
        console.log(`Downloading file, expected size: ${contentLength} bytes`);

        const file = fs.createWriteStream(destination);
        let downloadedBytes = 0;

        response.on("data", (chunk: any) => {
          downloadedBytes += chunk.length;
        });

        response.pipe(file);

        file.on("finish", () => {
          file.close((err: any) => {
            if (err) {
              return reject(err);
            }

            console.log(`File downloaded successfully to ${destination}`);
            console.log(`Downloaded ${downloadedBytes} bytes`);

            // Verify file size
            const stats = fs.statSync(destination);
            if (stats.size === 0) {
              return reject(new Error("Downloaded file is empty"));
            }

            // Give permissions to the file
            fs.chmod(destination, 0o644, (chmodErr: any) => {
              if (chmodErr) {
                console.error(
                  `Error setting permissions for ${destination}:`,
                  chmodErr,
                );
                return reject(chmodErr);
              }
              console.log(`Permissions set for ${destination}`);
              resolve(true);
            });
          });
        });

        file.on("error", (err: any) => {
          fs.unlink(destination, () => reject(err));
        });
      });

      request.on("error", (err: any) => {
        reject(err);
      });

      request.setTimeout(30000, () => {
        request.destroy();
        reject(new Error("Download timeout"));
      });
    };

    downloadWithRedirect(url);
  });
};

/**
 * Get the text content from an event panel element
 */
export async function getEventText(
  elementHandle: ElementHandle,
): Promise<string> {
  try {
    const text = await elementHandle.evaluate((el) => {
      const contentElement = (el as Element).querySelector(
        ".mat-expansion-panel-body .event-content",
      );
      return contentElement ? contentElement.textContent : null;
    });
    return text ? text.trim() : "";
  } catch (error: any) {
    console.error(`Error getting text for event:`, error.message);
    return "";
  }
}

/**
 * Wait for a specific event to appear in the UI for a given user instance
 *
 * <mat-expansion-panel>
 *   <mat-expansion-panel-header>
 *     <span class="mat-content"> EVENT_NAME </span>
 *   </mat-expansion-panel-header>
 *   <div class="mat-expansion-panel-content-wrapper">
 *     <div class="mat-expansion-panel-content">
 *       <div class="mat-expansion-panel-body">
 *         <div class="event-content">EVENT_CONTENT</div>
 *       </div>
 *     </div>
 *   </div>
 * </mat-expansion-panel>
 */
export async function waitForEvent(
  page: Page,
  eventName: string,
  numEvents: number,
  user: number,
  timeout = 5000,
): Promise<ElementHandle[]> {
  const selector = `#openvidu-instance-${user} mat-accordion mat-expansion-panel mat-expansion-panel-header span.mat-content:has-text("${eventName}")`;
  const locator = page.locator(selector);
  try {
    await locator.nth(numEvents - 1).waitFor({ timeout, state: "visible" });
  } catch (error: any) {
    console.error(
      `Timeout waiting for ${eventName} events (${numEvents}) in user ${user}:`,
      error.message,
    );
    if (!page.isClosed()) {
      try {
        const screenshot = await page.screenshot();
        const base64Image = screenshot.toString("base64");
        console.log(
          `Screenshot at timeout:\ndata:image/png;base64,${base64Image}\n`,
        );
      } catch (screenshotError: any) {
        console.error(`Failed to capture screenshot:`, screenshotError.message);
      }
    } else {
      console.warn(`Page already closed; skipping screenshot`);
    }
    throw error;
  }

  const headerHandles = await locator.elementHandles();
  const panelElements: ElementHandle[] = [];

  for (let i = 0; i < Math.min(numEvents, headerHandles.length); i++) {
    const panelHandle = await headerHandles[i].evaluateHandle((header) =>
      (header as Element).closest("mat-expansion-panel"),
    );
    const elementHandle = panelHandle.asElement();
    if (elementHandle) {
      panelElements.push(elementHandle);
    } else {
      await panelHandle.dispose();
    }
  }

  return panelElements;
}

/**
 * Wait for an event with specific content that starts with a given string
 */
export async function waitForEventContentToStartWith(
  page: Page,
  eventName: string,
  eventContent: string,
  numEvents: number,
  user: number,
  timeout = 5000,
): Promise<ElementHandle[]> {
  const selector = `#openvidu-instance-${user} mat-accordion mat-expansion-panel mat-expansion-panel-header span.mat-content:has-text("${eventName}")`;
  const locator = page.locator(selector);
  const deadline = Date.now() + timeout;

  while (Date.now() < deadline) {
    if (page.isClosed()) {
      console.warn(`Page is closed, exiting waitForEventContentToStartWith`);
      break;
    }

    const remaining = Math.max(deadline - Date.now(), 0);
    const headerCount = await locator.count();
    if (headerCount < numEvents) {
      if (remaining === 0) {
        break;
      }
      await page.waitForTimeout(Math.min(remaining, 200));
      continue;
    }

    const headerHandles = await locator.elementHandles();
    const matchingPanels: ElementHandle[] = [];

    for (const headerHandle of headerHandles) {
      const panelHandle = await headerHandle.evaluateHandle((header) =>
        (header as Element).closest("mat-expansion-panel"),
      );
      await headerHandle.dispose();
      const elementHandle = panelHandle.asElement();
      if (!elementHandle) {
        await panelHandle.dispose();
        continue;
      }

      const text = await getEventText(elementHandle);

      if (text.startsWith(eventContent)) {
        console.log(
          `Found ${eventName} event for user ${user} starting with "${eventContent}": ${text}`,
        );
        matchingPanels.push(elementHandle);
        if (matchingPanels.length === numEvents) {
          return matchingPanels;
        }
      } else {
        await elementHandle.dispose();
      }
    }

    for (const panel of matchingPanels) {
      await panel.dispose();
    }

    await page.waitForTimeout(
      Math.min(Math.max(deadline - Date.now(), 0), 200),
    );
  }

  if (!page.isClosed()) {
    try {
      const screenshot = await page.screenshot();
      const base64Image = screenshot.toString("base64");
      console.log(
        `Screenshot at timeout for ${eventName} starting with "${eventContent}":\ndata:image/png;base64,${base64Image}\n`,
      );
    } catch (screenshotError: any) {
      console.error(
        `Failed to capture screenshot for ${eventName}:`,
        screenshotError.message,
      );
    }
  } else {
    console.warn("Page already closed; skipping screenshot");
  }

  throw new Error(
    `Timeout waiting for ${eventName} event contents (${numEvents}) for user ${user} starting with "${eventContent}"`,
  );
}

/**
 * Count total events of a specific type for a user
 */
export async function countTotalEvents(
  page: Page,
  eventName: string,
  user: number,
): Promise<number> {
  const selector = `#openvidu-instance-${user} mat-accordion mat-expansion-panel mat-expansion-panel-header span.mat-content:has-text("${eventName}")`;
  try {
    const elements = await page.$$(selector);
    return elements.length;
  } catch (error: any) {
    console.error(`Error counting events for ${eventName}:`, error.message);
    return 0;
  }
}
