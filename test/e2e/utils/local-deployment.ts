import { execCommand, sleep } from "./helper";

import fs from "fs";
import path from "path";
import yaml from "yaml";

type DeploymentEdition = "community" | "pro";
type Processing = "automatic" | "manual";

const DEFAULT_EDITION: DeploymentEdition =
  (process.env.DEPLOYMENT_EDITION as DeploymentEdition) || "community";
const LOCAL_DEPLOYMENT_BASE_PATH =
  process.env.LOCAL_DEPLOYMENT_BASE_PATH ||
  path.resolve(__dirname, "../../../../openvidu-local-deployment");

const ALL_EDITIONS: DeploymentEdition[] = ["community", "pro"];

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

  private static configureLanPrivateIp() {
    let scriptCommand: string;
    switch (process.platform) {
      case "linux":
        scriptCommand = "sh configure_lan_private_ip_linux.sh";
        break;
      case "darwin":
        scriptCommand = "sh configure_lan_private_ip_macos.sh";
        break;
      case "win32":
        scriptCommand = "configure_lan_private_ip_windows.bat";
        break;
      default:
        throw new Error(
          `Unsupported platform '${process.platform}' for LAN private IP configuration`,
        );
    }
    console.log(`Configuring LAN private IP (${scriptCommand})...`);
    execCommand(scriptCommand, { cwd: this.getLocalDeploymentPath() });
  }

  static async start(
    edition: DeploymentEdition,
    provider: any,
    customLicense?: string,
    processing?: Processing,
  ) {
    this.edition = edition;
    console.log(
      "Configuring local deployment with agent-speech-processing live_captions provider " +
        Object.keys(provider)[0],
    );
    console.log(`Using deployment edition: ${this.edition}`);
    console.log(`Deployment path: ${this.getLocalDeploymentPath()}`);

    this.configureLanPrivateIp();
    this.configureProvider(provider, customLicense, processing);

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

    // Wait for the worker to register (nemotron takes longer)
    await this.waitForAgentWorkerRegistered();

    console.log("Local deployment started");
  }

  private static async waitForAgentWorkerRegistered(timeoutMs = 180000) {
    const container = "agent-speech-processing";
    const start = Date.now();
    console.log(`Waiting for '${container}' worker to register...`);
    while (Date.now() - start < timeoutMs) {
      let logs = "";
      try {
        // `|| true` so a not-yet-spawned container doesn't throw; keep polling.
        logs = execCommand(`docker logs ${container} 2>&1 || true`);
      } catch {
        // Ignore transient docker errors and retry.
      }
      if (logs.includes("registered worker")) {
        const secs = ((Date.now() - start) / 1000).toFixed(1);
        console.log(`Agent '${container}' worker registered after ${secs}s`);
        return;
      }
      await sleep(1);
    }
    console.warn(
      `Agent '${container}' worker did not register within ${
        timeoutMs / 1000
      }s; proceeding anyway (the test may fail because the agent is not ready)`,
    );
  }

  /**
   * Stop the active OpenVidu local deployment, regardless of its edition.
   */
  static stop() {
    console.log("Stopping local deployment...");
    const previousEdition = this.edition;
    try {
      for (const edition of ALL_EDITIONS) {
        this.edition = edition;
        const dockerComposeFile = this.getDockerComposeFile();
        if (fs.existsSync(dockerComposeFile)) {
          execCommand(`docker compose -f ${dockerComposeFile} down -v`);
        }
      }
    } finally {
      this.edition = previousEdition;
    }
  }

  private static configureProvider(
    provider: any,
    customLicense?: string,
    processing?: Processing,
  ) {
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
    if (processing) {
      doc.setIn(["live_captions", "processing"], processing);
    }
    const providerConfig = provider[providerName];
    for (const [key, value] of Object.entries(providerConfig)) {
      if (value !== null && value !== undefined && value !== "") {
        doc.setIn(["live_captions", providerName, key], value);
      }
    }
    // GPU acceleration: when STT_ACCEL is set, the GPU-capable local providers (sherpa, nemotron) use their "-cudaXX"
    // image, request GPU passthrough to the agent container via the `docker_options.gpus`, and run their CUDA runtime.
    // Empty/unset => CPU images (default, unchanged behavior)
    const accel = (process.env.STT_ACCEL || "").trim(); // "" | "cuda11" | "cuda12"
    const gpu = accel === "cuda11" || accel === "cuda12";

    if (providerName === "vosk") {
      // vosk has no GPU build (CPU-only).
      doc.set("docker_image", "openvidu/agent-speech-processing-vosk:main");
    } else if (providerName === "sherpa") {
      const suffix = gpu ? `-${accel}` : "";
      doc.set(
        "docker_image",
        `openvidu/agent-speech-processing-sherpa${suffix}:main`,
      );
      if (gpu) {
        doc.setIn(["docker_options", "gpus"], "all");
        doc.setIn(["live_captions", "sherpa", "provider"], "cuda");
      }
      this.setOpenViduProLicenseInOperatorService(customLicense);
    } else if (providerName === "nemotron") {
      const suffix = gpu ? "-cuda12" : "";
      doc.set(
        "docker_image",
        `openvidu/agent-speech-processing-nemotron${suffix}:main`,
      );
      if (gpu) {
        doc.setIn(["docker_options", "gpus"], "all");
        doc.setIn(["live_captions", "nemotron", "device"], "cuda");
      }
      this.setOpenViduProLicenseInOperatorService(customLicense);
    } else {
      doc.set("docker_image", "openvidu/agent-speech-processing-cloud:main");
    }

    this.writeYamlFile(agentSpeechProcessingFile, doc);
  }

  private static setOpenViduProLicenseInOperatorService(
    customLicense?: string,
  ) {
    const dockerComposeFile = this.getDockerComposeFile();
    const yamlContent = fs.readFileSync(dockerComposeFile, "utf8");
    const doc = yaml.parseDocument(yamlContent, {
      keepSourceTokens: true,
      prettyErrors: true,
    });
    const operatorService = doc.getIn(["services", "operator"]) as any;
    if (operatorService) {
      let env = operatorService.get("environment");
      const licenseValue = customLicense || "${OPENVIDU_PRO_LICENSE:-}";

      if (yaml.isSeq(env)) {
        const licenseEntry = `OPENVIDU_PRO_LICENSE=${licenseValue}`;

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
      } else if (yaml.isMap(env)) {
        // Environment is a mapping: {KEY: value, ...}
        env.set("OPENVIDU_PRO_LICENSE", licenseValue);
      } else {
        // No environment defined, create as mapping
        env = new yaml.YAMLMap();
        env.set("OPENVIDU_PRO_LICENSE", licenseValue);
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
