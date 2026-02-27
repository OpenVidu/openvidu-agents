import { test, expect, Page } from "@playwright/test";
import {
  AccessToken,
  AgentDispatchClient,
  type AgentDispatch,
} from "livekit-server-sdk";
import { Room, dispose } from "@livekit/rtc-node";
import { RoomAgentDispatch, RoomConfiguration } from "@livekit/protocol";
import { LocalDeployment } from "./utils/local-deployment";
import { TESTAPP_URL } from "./config";
import { downloadFile, execCommand, waitForEvent } from "./utils/helper";

const PROVIDER = {
  vosk: { model: "vosk-model-en-us-0.22-lgraph", use_silero_vad: false },
};
const LIVEKIT_URL_HTTP = "http://localhost:7880";
const LIVEKIT_URL_RTC = "ws://localhost:7880";
const LIVEKIT_API_KEY = "devkey";
const LIVEKIT_API_SECRET = "secret";
const AGENT_NAME = "speech-processing";

const dispatchClient = new AgentDispatchClient(
  LIVEKIT_URL_HTTP,
  LIVEKIT_API_KEY,
  LIVEKIT_API_SECRET,
);

function roomName(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
}

async function connectPublisher(page: Page, room: string) {
  await page.goto(TESTAPP_URL);
  await page.click("#add-user-btn");
  await page.fill("#room-name-input-0", room);
  await page.click("#openvidu-instance-0 .subscriber-checkbox");
  await page.click("#room-options-btn-0");
  await page.click("#video-capture-false");
  await page.click("#close-dialog-btn");
  await page.click(".connect-btn");
  await waitForEvent(page, "localTrackPublished", 1, 0, 60000);
}

async function finalTranscriptionCount(page: Page): Promise<number> {
  return page
    .locator(
      '#openvidu-instance-0 mat-accordion mat-expansion-panel mat-expansion-panel-header span.mat-content:has-text("finalTranscription")',
    )
    .count();
}

async function waitForDispatches(room: string, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const dispatches = (await dispatchClient.listDispatch(room)) as AgentDispatch[];
    if (dispatches.length > 0) {
      return dispatches;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return [] as AgentDispatch[];
}

async function connectWithTokenDispatch(room: string) {
  const at = new AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, {
    identity: `token-participant-${Date.now()}`,
  });
  at.addGrant({ roomJoin: true, room });
  at.roomConfig = new RoomConfiguration({
    agents: [new RoomAgentDispatch({ agentName: AGENT_NAME })],
  });

  const token = await at.toJwt();

  const rtcRoom = new Room();
  try {
    await rtcRoom.connect(LIVEKIT_URL_RTC, token, {
      autoSubscribe: false,
      dynacast: false,
    });
    await new Promise((resolve) => setTimeout(resolve, 2000));
    await rtcRoom.disconnect();
  } finally {
    await dispose();
  }
}

test.beforeAll(async () => {
  const fs = require("fs");
  const path = require("path");
  const audioFilePath = path.join(
    __dirname,
    "resources",
    "stt-test-with-silence.wav",
  );
  if (!fs.existsSync(audioFilePath) || fs.statSync(audioFilePath).size === 0) {
    await downloadFile(
      "https://s3.eu-west-1.amazonaws.com/public.openvidu.io/stt-test-with-silence.wav",
      audioFilePath,
    );
  }
  LocalDeployment.stop();
});

test.describe("live_captions.processing dispatch", () => {
  test.afterEach(async ({}, testInfo) => {
    if (testInfo.status === "failed") {
      try {
        console.log(execCommand("docker logs agent-speech-processing"));
      } catch (error: any) {
        console.log("Failed to get docker logs:", error.message);
      }
    }
    LocalDeployment.stop();
  });

  test("automatic: captions are received without explicit dispatch", async ({
    page,
  }) => {
    const room = roomName("auto");
    await LocalDeployment.start("community", PROVIDER, undefined, "automatic");
    await connectPublisher(page, room);
    await waitForEvent(page, "finalTranscription", 1, 0, 60000);
  });

  test("manual: no captions without dispatch", async ({ page }) => {
    const room = roomName("manual-none");
    await LocalDeployment.start("community", PROVIDER, undefined, "manual");
    await connectPublisher(page, room);
    await page.waitForTimeout(12000);
    expect(await finalTranscriptionCount(page)).toBe(0);
  });

  test("manual + API dispatch: captions are received", async ({ page }) => {
    const room = roomName("manual-api");
    await LocalDeployment.start("community", PROVIDER, undefined, "manual");
    await dispatchClient.createDispatch(room, AGENT_NAME);
    await connectPublisher(page, room);
    await waitForEvent(page, "finalTranscription", 1, 0, 60000);
  });

  test("manual + participant token dispatch: dispatch is created on join", async ({
    page,
  }) => {
    const room = roomName("manual-token");
    await LocalDeployment.start("community", PROVIDER, undefined, "manual");
    await connectWithTokenDispatch(room);
    const dispatches = await waitForDispatches(room, 15000);
    expect(dispatches.length).toBeGreaterThan(0);
    expect(dispatches.some((dispatch) => dispatch.agentName === AGENT_NAME)).toBeTruthy();
  });
});
