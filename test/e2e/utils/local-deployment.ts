import { execCommand, sleep } from "./helper";

import fs from "fs";
import yaml from "yaml";

type DeploymentEdition = "community" | "pro";

const DEFAULT_EDITION: DeploymentEdition =
  (process.env.DEPLOYMENT_EDITION as DeploymentEdition) || "community";
const LOCAL_DEPLOYMENT_BASE_PATH =
  process.env.LOCAL_DEPLOYMENT_BASE_PATH || "../../openvidu-local-deployment";

export class LocalDeployment {
  private static edition: DeploymentEdition = DEFAULT_EDITION;

  private static getLocalDeploymentPath(): string {
    return `${LOCAL_DEPLOYMENT_BASE_PATH}/${this.edition}`;
  }

  private static getDockerComposeFile(): string {
    return `${this.getLocalDeploymentPath()}/docker-compose.yaml`;
  }

  private static getAgentSpeechProcessingFile(): string {
    return `${this.getLocalDeploymentPath()}/agent-speech-processing.yaml`;
  }

  static async start(edition: DeploymentEdition, provider: any) {
    this.edition = edition;
    console.log(
      "Configuring local deployment with agent-speech-processing live_captions provider " +
        Object.keys(provider)[0],
    );
    console.log(`Using deployment edition: ${this.edition}`);
    console.log(`Deployment path: ${this.getLocalDeploymentPath()}`);

    this.configureProvider(provider);

    console.log("Restarting local deployment...");
    const dockerComposeFile = this.getDockerComposeFile();
    execCommand(`docker compose -f ${dockerComposeFile} up -d`);
    let statusCode: string;

    // Check that container "ready-check" exited with code 0
    do {
      await sleep(1);
      statusCode = execCommand(
        "docker inspect ready-check -f {{.State.Status}}:{{.State.ExitCode}}",
      );
    } while (statusCode !== "exited:0");
    console.log("Local deployment started");
  }

  static stop() {
    console.log("Stopping local deployment...");
    const dockerComposeFile = this.getDockerComposeFile();
    execCommand(`docker compose -f ${dockerComposeFile} down -v`);
  }

  private static configureProvider(provider: any) {
    const providerName = Object.keys(provider)[0];

    // Parse with specific options to preserve comments
    const agentSpeechProcessingFile = this.getAgentSpeechProcessingFile();
    const yamlContent = fs.readFileSync(agentSpeechProcessingFile, "utf8");
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
    if (providerName === "vosk") {
      doc.set("docker_image", "openvidu/agent-speech-processing-vosk:main");
    } else if (providerName === "sherpa") {
      doc.set("docker_image", "openvidu/agent-speech-processing-sherpa:main");
      this.setOpenViduProLicenseInOperatorService();
    } else {
      doc.set("docker_image", "openvidu/agent-speech-processing-cloud:main");
    }

    this.writeYamlFile(agentSpeechProcessingFile, doc);
  }

  private static setOpenViduProLicenseInOperatorService() {
    const dockerComposeFile = this.getDockerComposeFile();
    const yamlContent = fs.readFileSync(dockerComposeFile, "utf8");
    const doc = yaml.parseDocument(yamlContent, {
      keepSourceTokens: true,
      prettyErrors: true,
    });
    const operatorService = doc.getIn(["services", "operator"]) as any;
    if (operatorService) {
      let env = operatorService.get("environment");

      if (yaml.isSeq(env)) {
        const licenseEntry = "OPENVIDU_PRO_LICENSE=${OPENVIDU_PRO_LICENSE:-}";

        // Find and replace if exists, otherwise add
        const index = env.items.findIndex((item) => {
          const value = yaml.isScalar(item) ? String(item.value) : String(item);
          return value.startsWith("OPENVIDU_PRO_LICENSE=");
        });

        if (index >= 0) {
          env.set(index, licenseEntry);
        } else {
          env.add(licenseEntry);
        }
      } else {
        // No environment defined, create as mapping
        env = new yaml.YAMLMap();
        env.set("OPENVIDU_PRO_LICENSE", "${OPENVIDU_PRO_LICENSE:-}");
        operatorService.set("environment", env);
      }
    }
    this.writeYamlFile(dockerComposeFile, doc);
  }

  private static writeYamlFile(
    filePath: string,
    document: yaml.Document.Parsed,
  ) {
    const output = document.toString({
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
    fs.writeFileSync(filePath, output, "utf8");
  }
}
