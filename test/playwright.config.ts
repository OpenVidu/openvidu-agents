import { defineConfig } from "@playwright/test";
import { RUN_MODE } from "./e2e/config";

const commonLaunchArgs = [
  "--use-fake-ui-for-media-stream",
  "--use-fake-device-for-media-stream",
  "--allow-file-access-from-files",
  "--no-sandbox",
  "--disable-dev-shm-usage",
];

export default defineConfig({
  testDir: "./e2e",
  timeout: 300000,
  retries: 0,
  workers: 1,
  fullyParallel: false,
  use: {
    headless: RUN_MODE === "CI",
    viewport: { width: 1280, height: 720 },
    ignoreHTTPSErrors: true,
    permissions: ["camera", "microphone"],
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "livecaptions",
      testMatch: /livecaptions\.spec\.ts$/,
      use: {
        launchOptions: {
          args: [
            ...commonLaunchArgs,
            "--use-file-for-fake-audio-capture=e2e/resources/stt-test-with-silence.wav",
          ],
        },
      },
    },
    {
      name: "livecaptions-memory-usage",
      testMatch: /livecaptions-memory-usage\.spec\.ts$/,
      use: {
        launchOptions: {
          args: [
            ...commonLaunchArgs,
            "--use-file-for-fake-audio-capture=e2e/resources/stt-test.wav",
          ],
        },
      },
    },
    {
      name: "livecaptions-wrong-license",
      testMatch: /livecaptions-wrong-license\.spec\.ts$/,
      use: {
        launchOptions: {
          args: commonLaunchArgs,
        },
      },
    },
  ],
  reporter:
    RUN_MODE == "CI"
      ? [
          ["list"],
          [
            "playwright-ctrf-json-reporter",
            {
              outputFile: "ctrf-report.json", // Optional: Output file name. Defaults to 'ctrf-report.json'.
              outputDir: "test-results", // Optional: Output directory path. Defaults to '.' (project root).
            },
          ],
        ]
      : [["list"]],
});
