import { test, expect } from "@playwright/test";
import { LocalDeployment } from "./utils/local-deployment";
import { TESTAPP_URL } from "./config";
import { downloadFile, execCommand } from "./utils/helper";

const STT_AI_PROVIDERS = [
  // {
  //   azure_openai: {
  //     azure_api_key: process.env.AZURE_OPENAI_API_KEY,
  //     azure_endpoint: process.env.AZURE_OPENAI_ENDPOINT,
  //   },
  // },
  {
    aws: {
      aws_access_key_id: process.env.AWS_ACCESS_KEY_ID,
      aws_secret_access_key: process.env.AWS_SECRET_ACCESS_KEY,
      aws_default_region: process.env.AWS_DEFAULT_REGION,
    },
  },
  // {
  //   google: {
  //     credentials_info: process.env.GOOGLE_CREDENTIALS_INFO,
  //   },
  // },
  // {
  //   sarvam: {
  //     api_key: process.env.SARVAM_API_KEY,
  //   },
  // },
  // {
  //   azure: {
  //     speech_key: process.env.AZURE_SPEECH_KEY,
  //     speech_region: process.env.AZURE_SPEECH_REGION,
  //   },
  // },
  // {
  //   assemblyai: {
  //     api_key: process.env.ASSEMBLYAI_API_KEY,
  //   },
  // },
  // {
  //   gladia: {
  //     api_key: process.env.GLADIA_API_KEY,
  //   },
  // },
  // {
  //   deepgram: {
  //     api_key: process.env.DEEPGRAM_API_KEY,
  //   },
  // },
  // {
  //   groq: {
  //     api_key: process.env.GROQ_API_KEY,
  //   },
  // },
  // {
  //   openai: {
  //     api_key: process.env.OPENAI_API_KEY,
  //   },
  // },
  // REASON THIS PROVIDER CAN'T BE AUTOMATICALLY TESTED: no free credits or tier available
  // {
  //   fal: {
  //     api_key:
  //       process.env.FAL_API_KEY,
  //   },
  // },
  // REASON THIS PROVIDER CAN'T BE AUTOMATICALLY TESTED: it fails with error message "Speechmatics connection closed unexpectedly"
  // {
  //   speechmatics: {
  //     api_key:
  //       process.env.SPEECHMATICS_API_KEY,
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
      await LocalDeployment.start(provider);
      await page.goto(TESTAPP_URL);
    });

    test.afterEach(async () => {
      LocalDeployment.stop();
    });

    test(`testing with ${providerName}`, async ({ page }) => {
      await page.click("#add-user-btn");
      await page.click(".connect-btn");
      try {
        await waitForEvent(page, "transcriptionReceived", 1, 0, 45000);
      } catch (error) {
        console.log(execCommand("docker logs agent-speech-processing"));
        throw error;
      }
    });
  });
});

function waitForEvent(
  page: any,
  eventName: string,
  numEvents: number,
  user: number,
  timeout = 5000
): Promise<any> {
  return new Promise((resolve, reject) => {
    const selector = `#openvidu-instance-${user} mat-accordion mat-expansion-panel mat-expansion-panel-header span.mat-content:has-text("${eventName}")`;
    let eventsCount = 0;

    const checkEvents = async () => {
      const elements = await page.$$(selector);
      eventsCount = elements.length;

      if (eventsCount >= numEvents) {
        resolve(elements);
      } else {
        setTimeout(checkEvents, 100); // Check again after 100ms
      }
    };

    checkEvents();

    // Reject if timeout is reached
    setTimeout(() => {
      if (eventsCount < numEvents) {
        reject(new Error(`Timeout waiting for ${eventName} events`));
      }
    }, timeout);
  });
}
