import fs from "fs";
import os from "os";
import { execCommand } from "./helper";

/** Name of the agent container the operator launches in the local deployment. */
export const AGENT_CONTAINER = "agent-speech-processing";

// Whether nvidia-smi works on this host; probed once, cached.
let gpuAvailable: boolean | null = null;

/** Parse docker's "12.34%" into a number (per-core convention: 400% = 4 cores). */
export function parseDockerPercent(value: string): number {
  const n = parseFloat(value.replace("%", ""));
  return Number.isFinite(n) ? n : 0;
}

/**
 * Host CPU time in jiffies from /proc/stat (Linux only): busy and total. Two
 * snapshots around a sampling window give the host-wide utilization, which
 * includes what runs outside containers: a browser publishing tracks, the
 * runner itself, this very process.
 */
export function readHostCpuJiffies(): { busy: number; total: number } | null {
  try {
    const line = fs.readFileSync("/proc/stat", "utf8").split("\n")[0];
    const fields = line.trim().split(/\s+/).slice(1).map(Number);
    if (fields.length < 5) {
      return null;
    }
    const total = fields.reduce((a: number, b: number) => a + b, 0);
    const idle = fields[3] + fields[4]; // idle + iowait
    return { busy: total - idle, total };
  } catch {
    return null;
  }
}

/** Host-wide busy percentage (docker's per-core convention) between two snapshots. */
export function hostBusyPercent(
  before: { busy: number; total: number } | null,
  after: { busy: number; total: number } | null,
): number | null {
  if (!before || !after || after.total <= before.total) {
    return null;
  }
  return (
    ((after.busy - before.busy) / (after.total - before.total)) *
    100 *
    os.cpus().length
  );
}

/**
 * One-line snapshot of where the machine's resources go, to identify WHICH
 * resource saturates first as a load ramp progresses:
 * - the agent container (CPU% in docker's per-core convention, 400% = 4 cores
 *   fully busy, and RAM);
 * - every other container, as a total plus the three largest consumers;
 * - the host as a whole, busy % of all cores over the sampling window. The
 *   difference to the containers' sum is whatever runs outside docker (the
 *   GitHub runner itself); on a dedicated agent host it should be close to zero;
 * - on GPU hosts (`gpu`, default: STT_ACCEL set), VRAM and GPU utilization
 *   from nvidia-smi (host-wide).
 */
export function sampleHostLoad(options: { gpu?: boolean } = {}): string {
  const gpu = options.gpu ?? Boolean(process.env.STT_ACCEL);
  const parts: string[] = [];
  const before = readHostCpuJiffies();
  try {
    // One docker stats call over every container. It takes ~2 s (docker
    // samples twice to derive CPU%), which doubles as the host sampling window.
    const rows = execCommand(
      'docker stats --no-stream --format "{{.Name}} {{.CPUPerc}} {{.MemUsage}}"',
    )
      .trim()
      .split("\n")
      .map((line) => line.trim().split(/\s+/))
      .filter((fields) => fields.length >= 2);
    const agent = rows.find((fields) => fields[0] === AGENT_CONTAINER);
    if (agent) {
      parts.push(`agent CPU: ${agent[1]}`, `RAM: ${agent.slice(2).join(" ")}`);
    } else {
      parts.push("agent stats unavailable (container not found)");
    }
    const others = rows
      .filter((fields) => fields[0] !== AGENT_CONTAINER)
      .map((fields) => ({
        name: fields[0],
        cpu: parseDockerPercent(fields[1]),
      }))
      .sort((a, b) => b.cpu - a.cpu);
    const othersTotal = others.reduce((sum, o) => sum + o.cpu, 0);
    const top = others
      .slice(0, 3)
      .map((o) => `${o.name} ${o.cpu.toFixed(0)}%`)
      .join(", ");
    parts.push(`other containers: ${othersTotal.toFixed(0)}% (${top})`);
  } catch (error: any) {
    parts.push(`docker stats unavailable (${error.message})`);
  }
  const busy = hostBusyPercent(before, readHostCpuJiffies());
  if (busy !== null) {
    parts.push(`host: ${busy.toFixed(0)}% of ${os.cpus().length * 100}% busy`);
  }
  if (gpu && gpuAvailable !== false) {
    try {
      const [vram, util] = execCommand(
        "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits",
      )
        .trim()
        .split("\n")[0]
        .split(",")
        .map((v) => v.trim());
      gpuAvailable = true;
      parts.push(`VRAM: ${vram} MiB`, `GPU util: ${util}%`);
    } catch {
      gpuAvailable = false;
    }
  }
  return `[${parts.join(" | ")}]`;
}
