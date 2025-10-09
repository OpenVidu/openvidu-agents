import { test, expect, Page } from "@playwright/test";
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
    },
  },
  {
    mistralai: {
      api_key: process.env.MISTRALAI_API_KEY,
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

test.beforeAll(async () => {
  const fs = require("fs");
  const path = require("path");
  const audioFilePath = path.join(__dirname, "resources", "stt-test.wav");
  // If file does not exist or is empty, download it
  if (!fs.existsSync(audioFilePath) || fs.statSync(audioFilePath).size === 0) {
    await downloadFile(
      "https://github.com/OpenVidu/openvidu/raw/v2/openvidu-test-e2e/docker/stt-test.wav",
      audioFilePath
    );
  }
  LocalDeployment.stop();
});

STT_AI_PROVIDERS.forEach((provider) => {
  const providerName = Object.keys(provider)[0];

  test.describe(() => {
    test.beforeEach(async ({ page }) => {
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

    test(`testing with ${providerName}`, async ({ page }) => {
      console.log(`Running test with provider: ${providerName}`);
      await page.goto(TESTAPP_URL);
      await page.click("#add-user-btn");
      await page.click(".connect-btn");
      await waitForEvent(page, "interimTranscription", 1, 0, 60000);
      console.log(
        `Interim transcription received from provider ${providerName}`
      );
      await waitForEvent(page, "finalTranscription", 1, 0, 60000);
      console.log(`Final transcription received from provider ${providerName}`);
      const totalInterimEvents = await countTotalEvents(
        page,
        "interimTranscription",
        0
      );
      const totalFinalEvents = await countTotalEvents(
        page,
        "finalTranscription",
        0
      );
      if (totalInterimEvents === totalFinalEvents) {
        console.warn(
          `ATTENTION! No real interim transcriptions supported by provider ${providerName}`
        );
      }
    });
  });
});

async function waitForEvent(
  page: Page,
  eventName: string,
  numEvents: number,
  user: number,
  timeout = 5000
): Promise<any> {
  return new Promise(async (resolve, reject) => {
    const selector = `#openvidu-instance-${user} mat-accordion mat-expansion-panel mat-expansion-panel-header span.mat-content:has-text("${eventName}")`;
    let eventsCount = 0;

    const checkEvents = async () => {
      let elements;
      try {
        elements = await page.$$(selector);
      } catch (error: any) {
        console.error(
          `Error selecting elements for event ${eventName}:`,
          error.message
        );
        return reject(error);
      }
      eventsCount = elements.length;

      if (eventsCount >= numEvents) {
        resolve(elements);
      } else {
        setTimeout(checkEvents, 100); // Check again after 100ms
      }
    };

    await checkEvents();

    // Reject if timeout is reached
    setTimeout(() => {
      if (eventsCount < numEvents) {
        reject(new Error(`Timeout waiting for ${eventName} events`));
      }
    }, timeout);
  });
}

async function countTotalEvents(
  page: Page,
  eventName: string,
  user: number
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
