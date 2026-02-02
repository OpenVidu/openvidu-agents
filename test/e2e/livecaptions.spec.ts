import { test, expect, Page, ElementHandle } from "@playwright/test";
import { LocalDeployment } from "./utils/local-deployment";
import { TESTAPP_URL } from "./config";
import { downloadFile, execCommand } from "./utils/helper";

const STT_AI_PROVIDERS = [
  {
    azure_openai: {
      azure_api_key: process.env.AZURE_OPENAI_API_KEY,
      azure_endpoint: process.env.AZURE_OPENAI_ENDPOINT,
      api_version: "2025-04-01-preview",
    },
  },
  {
    aws: {
      aws_access_key_id: process.env.AWS_ACCESS_KEY_ID,
      aws_secret_access_key: process.env.AWS_SECRET_ACCESS_KEY,
      aws_default_region: process.env.AWS_DEFAULT_REGION,
    },
  },
  {
    google: {
      credentials_info: process.env.GOOGLE_CREDENTIALS_INFO,
    },
  },
  {
    sarvam: {
      api_key: process.env.SARVAM_API_KEY,
    },
  },
  {
    azure: {
      speech_key: process.env.AZURE_SPEECH_KEY,
      speech_region: process.env.AZURE_SPEECH_REGION,
    },
  },
  {
    assemblyai: {
      api_key: process.env.ASSEMBLYAI_API_KEY,
    },
  },
  {
    gladia: {
      api_key: process.env.GLADIA_API_KEY,
    },
  },
  {
    deepgram: {
      api_key: process.env.DEEPGRAM_API_KEY,
    },
  },
  {
    groq: {
      api_key: process.env.GROQ_API_KEY,
    },
  },
  {
    openai: {
      api_key: process.env.OPENAI_API_KEY,
    },
  },
  {
    speechmatics: {
      api_key: process.env.SPEECHMATICS_API_KEY,
      max_concurrent_transcriptions: 2, // https://docs.speechmatics.com/speech-to-text/realtime/limits
    },
  },
  {
    mistralai: {
      api_key: process.env.MISTRALAI_API_KEY,
    },
  },
  {
    cartesia: {
      api_key: process.env.CARTESIA_API_KEY,
    },
  },
  {
    soniox: {
      api_key: process.env.SONIOX_API_KEY,
    },
  },
  {
    nvidia: {
      api_key: process.env.NVIDIA_API_KEY,
    },
  },
  {
    vosk: {
      model: "vosk-model-en-us-0.22-lgraph",
    },
  },
  // REASON THIS PROVIDER CAN'T BE AUTOMATICALLY TESTED: it is broken as it needs library httpx,
  // which the google plugin also requires with a different version
  // {
  //   spitch: {
  //     api_key: process.env.SPITCH_API_KEY,
  //   },
  // },
  // REASON THIS PROVIDER CAN'T BE AUTOMATICALLY TESTED: no free credits or tier available
  // {
  //   fal: {
  //     api_key:
  //       process.env.FAL_API_KEY,
  //   },
  // },
  // REASON THIS PROVIDER CAN'T BE AUTOMATICALLY TESTED: : Clova platform has a very difficult signup process, requiring a phone number based in only a handful of countries
  // {
  //   clova: {
  //     api_key:
  //       process.env.CLOVA_API_KEY,
  //   },
  // },
] as const;

const AUDIO_TRANSCRIPTIONS = [
  "The stale smell of old beer lingers.",
  "It takes heat to bring out the odor.",
  "A cold dip restores health and zest.",
  "A salt pickle tastes fine with ham.",
  "Tacos al pastor are my favorite.",
  "A zestful food is the hot cross bun.",
];

test.beforeAll(async () => {
  const fs = require("fs");
  const path = require("path");
  const audioFilePath = path.join(
    __dirname,
    "resources",
    "stt-test-with-silence.wav",
  );
  // If file does not exist or is empty, download it
  if (!fs.existsSync(audioFilePath) || fs.statSync(audioFilePath).size === 0) {
    await downloadFile(
      "https://s3.eu-west-1.amazonaws.com/public.openvidu.io/stt-test-with-silence.wav",
      audioFilePath,
    );
  }
  LocalDeployment.stop();
});

function describeProviderTests(
  groupTitle: string,
  registerTests: ({
    provider,
    providerName,
  }: {
    provider: (typeof STT_AI_PROVIDERS)[number];
    providerName: string;
  }) => void,
) {
  // Reuse provider-level setup/teardown across different test groups.
  test.describe(groupTitle, () => {
    STT_AI_PROVIDERS.forEach((provider) => {
      const providerName = Object.keys(provider)[0] as string;

      test.describe(providerName, () => {
        test.beforeEach(async () => {
          LocalDeployment.stop();
          await LocalDeployment.start(provider);
        });

        test.afterEach(async ({}, testInfo) => {
          // Capture logs before stopping if test failed
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

        registerTests({ provider, providerName });
      });
    });
  });
}

describeProviderTests("Single user STT tests", ({ providerName }) => {
  test(`testing simple STT with ${providerName}`, async ({ page }) => {
    console.log(`Running simple test with provider: ${providerName}`);
    await page.goto(TESTAPP_URL);
    await page.click("#add-user-btn");
    await page.click(".connect-btn");
    const interimEvents = await waitForEvent(
      page,
      "interimTranscription",
      1,
      0,
      20000,
    );
    console.log(`Interim transcription received from provider ${providerName}`);
    const TIMEOUT_FINAL = providerName === "vosk" ? 50000 : 20000;
    const finalEvents = await waitForEvent(
      page,
      "finalTranscription",
      1,
      0,
      TIMEOUT_FINAL,
    );
    console.log(`Final transcription received from provider ${providerName}`);
    const totalInterimEvents = await countTotalEvents(
      page,
      "interimTranscription",
      0,
    );
    const totalFinalEvents = await countTotalEvents(
      page,
      "finalTranscription",
      0,
    );
    if (totalInterimEvents === totalFinalEvents) {
      console.warn(
        `ATTENTION! Same number of interim and final events (${totalFinalEvents}) for provider ${providerName}.`,
      );
    }
    let firstInterimEventText = await getEventText(interimEvents[0]);
    firstInterimEventText = firstInterimEventText.replace(
      /^TestParticipant0 is saying: /i,
      "",
    );
    let lastFinalEventText = await getEventText(
      finalEvents[finalEvents.length - 1],
    );
    lastFinalEventText = lastFinalEventText.replace(
      /^TestParticipant0 said: /i,
      "",
    );
    if (firstInterimEventText === lastFinalEventText) {
      console.warn(
        `ATTENTION! First interim and last final transcription are identical ("${lastFinalEventText}") for provider ${providerName}.\nNo real interim transcriptions supported by provider ${providerName}`,
      );
    } else {
      console.log(`Final transcription: "${lastFinalEventText}"`);
    }
    checkLevenshteinDistance(providerName, lastFinalEventText);
  });
});

describeProviderTests("Multi-user STT tests", ({ providerName }) => {
  test(`testing multiple users STT with ${providerName}`, async ({ page }) => {
    console.log(`Running multi-user test with provider: ${providerName}`);

    const providerConfig = STT_AI_PROVIDERS.find(
      (p) => Object.keys(p)[0] === providerName,
    ) as any;
    const maxConcurrentTranscriptions =
      providerConfig[providerName].max_concurrent_transcriptions;
    const NUM_USERS = maxConcurrentTranscriptions
      ? Math.min(4, maxConcurrentTranscriptions)
      : 4;

    await page.goto(TESTAPP_URL);
    for (let i = 0; i < NUM_USERS; i++) {
      await page.click("#add-user-btn");
      await page.click(`#openvidu-instance-${i} .subscriber-checkbox`);
      await page.click(`#room-options-btn-${i}`);
      await page.click("#video-capture-false");
      await page.click("#close-dialog-btn");
    }
    for (let i = 0; i < NUM_USERS; i++) {
      const connectButtons = await page.$$(".connect-btn");
      await connectButtons[i].click();
    }
    // Check that each user has received at least one final transcription event for every other user (and itself)
    const promises = [];
    const TIMEOUT = providerName === "vosk" ? 50000 : 20000;
    for (let user = 0; user < NUM_USERS; user++) {
      for (let otherUser = 0; otherUser < NUM_USERS; otherUser++) {
        promises.push(
          waitForEventContentToStartWith(
            page,
            "finalTranscription",
            `TestParticipant${otherUser} said: `,
            1,
            user,
            TIMEOUT,
          ),
        );
      }
    }
    console.log(`Waiting for ${promises.length} final transcription events`);
    const elements = await Promise.all(promises);
    console.log(
      `All final transcription events received for provider ${providerName}`,
    );
    for (const el of elements) {
      const firstEl = el[0];
      const text = await getEventText(firstEl);
      const strippedText = text.replace(/^TestParticipant\d+ said: /i, "");
      checkLevenshteinDistance(providerName, strippedText);
    }
  });
});

/*
<mat-expansion-panel>
  <mat-expansion-panel-header>
    <span class="mat-content"> EVENT_NAME </span>
  </mat-expansion-panel-header>
  <div class="mat-expansion-panel-content-wrapper">
    <div class="mat-expansion-panel-content">
      <div class="mat-expansion-panel-body">
        <div class="event-content">EVENT_CONTENT</div>
      </div>
    </div>
  </div>
</mat-expansion-panel>
*/
async function waitForEvent(
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

// Same as waitForEvent but also checks for specific content inside the event
async function waitForEventContentToStartWith(
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

async function getEventText(elementHandle: ElementHandle): Promise<string> {
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

function checkLevenshteinDistance(
  providerName: string,
  transcribedText: string,
) {
  // Compare only first sentence of the transcription.
  transcribedText = transcribedText.split(".")[0];
  let expectedText = AUDIO_TRANSCRIPTIONS[0];
  let LD = getLevenshteinDistance(expectedText, transcribedText);
  if (LD > 5) {
    throw new Error(
      `Levenshtein distance (${LD}) exceeds maximum allowed between expected and transcribed text.\nExpected: "${expectedText}"\nTranscribed: "${transcribedText}"`,
    );
  }
  console.log(`Levenshtein distance is ${LD}`);
}

function getLevenshteinDistance(
  expectedText: string,
  transcribedText: string,
): number {
  // Use fast-levenshtein NPM library
  const levenshtein = require("fast-levenshtein");
  // Remove all punctuation, trailing and leading spaces, and convert to lowercase
  expectedText = expectedText
    .replace(/[^\w\s]|_/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
  transcribedText = transcribedText
    .replace(/[^\w\s]|_/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
  return levenshtein.get(transcribedText, expectedText);
}

async function countTotalEvents(
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
