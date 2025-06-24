import { execCommand, sleep } from "./helper";

import fs from "fs";
import yaml from "yaml";

const LOCAL_DEPLOYMENT_PATH = "../../openvidu-local-deployment/community";
const DOCKER_COMPOSE_FILE = `${LOCAL_DEPLOYMENT_PATH}/docker-compose.yaml`;
const AGENT_SPEECH_PROCESSING_FILE = `${LOCAL_DEPLOYMENT_PATH}/agent-speech-processing.yaml`;

export class LocalDeployment {
  static async start(provider: any) {
    console.log(
      "Configuring local deployment with agent-speech-processing live_captions provider " +
        Object.keys(provider)[0]
    );

    const providerName = Object.keys(provider)[0];

    // Parse with specific options to preserve comments
    const yamlContent = fs.readFileSync(AGENT_SPEECH_PROCESSING_FILE, "utf8");
    const doc = yaml.parseDocument(yamlContent, {
      keepSourceTokens: true,
      prettyErrors: true,
    });

    doc.set("enabled", true);
    doc.setIn(["live_captions", "provider"], providerName);
    const providerConfig = provider[providerName];
    for (const [key, value] of Object.entries(providerConfig)) {
      if (value !== null && value !== undefined && value !== "") {
        doc.setIn(["live_captions", providerName, key], value);
      }
    }

    const output = doc.toString({
      lineWidth: 0,
      minContentWidth: 0,
      indent: 2,
      blockQuote: true,
      collectionStyle: "any",
      directives: null,
      flowCollectionPadding: true,
      indentSeq: true,
      simpleKeys: false,
      singleQuote: null,
      verifyAliasOrder: true,
    });
    fs.writeFileSync(AGENT_SPEECH_PROCESSING_FILE, output, "utf8");

    console.log("Restarting local deployment...");
    execCommand(`docker compose -f ${DOCKER_COMPOSE_FILE} up -d`);
    let statusCode: string;

    // Check that container "ready-check" exited with code 0
    do {
      await sleep(1);
      statusCode = execCommand(
        "docker inspect ready-check -f {{.State.Status}}:{{.State.ExitCode}}"
      );
    } while (statusCode !== "exited:0");
    console.log("Local deployment started");
  }

  static stop() {
    console.log("Stopping local deployment...");
    execCommand(`docker compose -f ${DOCKER_COMPOSE_FILE} down -v`);
  }
}
