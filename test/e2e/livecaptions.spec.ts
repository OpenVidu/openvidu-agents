import { test, expect, Page, ElementHandle } from "@playwright/test";
import { LocalDeployment } from "./utils/local-deployment";
import { TESTAPP_URL } from "./config";
import {
  downloadFile,
  execCommand,
  waitForEvent,
  waitForEventContentToStartWith,
  getEventText,
  countTotalEvents,
} from "./utils/helper";

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
    spitch: {
      api_key: process.env.SPITCH_API_KEY,
    },
  },
  {
    vosk: {
      model: "vosk-model-en-us-0.22-lgraph",
      use_silero_vad: false,
    },
  },
  {
    vosk: {
      model: "vosk-model-en-us-0.22-lgraph",
      use_silero_vad: true,
    },
  },
  {
    sherpa: {
      model: "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
      use_silero_vad: false,
    },
  },
  {
    sherpa: {
      model: "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
      use_silero_vad: true,
    },
  },
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
    STT_AI_PROVIDERS.forEach((provider, index) => {
      const providerName = Object.keys(provider)[0] as string;

      // Generate unique test name for duplicate providers
      const sameNameCount = STT_AI_PROVIDERS.slice(0, index).filter(
        (p) => Object.keys(p)[0] === providerName,
      ).length;
      const uniqueTestName =
        sameNameCount > 0
          ? `${providerName}-${sameNameCount + 1}`
          : providerName;

      test.describe(uniqueTestName, () => {
        test.beforeEach(async () => {
          LocalDeployment.stop();
          await LocalDeployment.start("pro", provider);
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
