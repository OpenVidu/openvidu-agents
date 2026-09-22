/**
 * Shared parsing and arithmetic for the capacity numbers: what the headless
 * probe prints (`CAPACITY RESULT: ...`, `Capacity so far: ...`) and what the
 * agent host samples (`Host load [...]`), turned into the figures the docs'
 * capacity table uses. Used by `summarize-run.ts` (a finished run, through the
 * GitHub CLI) and by `agent-host.ts summary` (the running workflow, from the
 * samples it just collected and the Publishers job log).
 */

export interface ProbeResult {
  tracks: number;
  label: string;
  agentHost: string;
  stopReason: string;
  /** null when the result line predates the sustained check */
  oldestSustained: boolean | null;
  /** "k/N" or "" when the result line predates the live-at-end figure */
  liveAtEnd: string;
  joinRetries: number;
  finalsPerTrack: number[];
  /** publisher host busy at the end as a fraction of its cores (0-1), from the result line's load suffix; -1 when the line lacks the core total */
  publisherHostBusy: number;
  resultLine: string;
}

export interface HostSample {
  time: number;
  agentCpu: number;
  ram: string;
  hostBusy: number;
  cores: number;
  vram?: number;
  gpuUtil?: number;
}

export interface CapacitySummary {
  result: ProbeResult;
  cores: number;
  plateauSamples: number;
  agentCpu: number;
  ram: string;
  perTrackVcpu: number;
  tracksPerVcpu: number;
  tracksPer8Vcpu: number;
  gpu: boolean;
  gpuPeak?: number;
  gpuAtPlateau?: number;
  vram?: number;
  degraded: boolean;
  /** the publishers' own host was near saturation: the count is a probe limit, not the agent's */
  publisherLimited: boolean;
}

export const RESULT_MARKER = "CAPACITY RESULT:";
export const PROGRESS_RE = /Capacity so far: (\d+) simultaneous/;
const SAMPLE_RE =
  /Host load \[agent CPU: ([\d.]+)% \| RAM: ([^|]+?) \| other containers: [^|]+ \| host: (\d+)% of (\d+)% busy(?: \| VRAM: (\d+) MiB \| GPU util: (\d+)%)?\]/;

export function parseResultLine(text: string): ProbeResult | null {
  const m = text.match(/CAPACITY RESULT: (\d+) simultaneous transcribed tracks with headless publishers(?: \[([^\]]*)\])? \((.*)\)/);
  if (!m) {
    return null;
  }
  const details = m[3];
  const detail = (key: string): string | undefined =>
    details.match(new RegExp(`${key}: ([^;)]*)`))?.[1]?.trim();
  const sustained = detail("oldest track still transcribing at full load");
  const finals = detail("finals per track") ?? "";
  return {
    tracks: Number(m[1]),
    label: m[2] ?? "",
    agentHost: detail("agent host") ?? "",
    stopReason: detail("stop reason") ?? "",
    oldestSustained: sustained === undefined ? null : sustained === "true",
    liveAtEnd: details.match(/tracks with a final in the last \d+s: (\d+\/\d+)/)?.[1] ?? "",
    joinRetries: Number(detail("join retries") ?? 0),
    finalsPerTrack: finals ? finals.split(",").map(Number).filter((n) => !Number.isNaN(n)) : [],
    publisherHostBusy: (() => {
      const b = text.match(/publisher host: (\d+)% of (\d+)% busy/);
      return b ? Number(b[1]) / Number(b[2]) : -1;
    })(),
    resultLine: text.slice(text.indexOf(RESULT_MARKER)),
  };
}

export function parseSampleLine(text: string, time: number): HostSample | null {
  const m = text.match(SAMPLE_RE);
  if (!m) {
    return null;
  }
  return {
    time,
    agentCpu: Number(m[1]),
    ram: m[2].trim(),
    hostBusy: Number(m[3]),
    cores: Number(m[4]) / 100,
    vram: m[5] ? Number(m[5]) : undefined,
    gpuUtil: m[6] ? Number(m[6]) : undefined,
  };
}

/** `<ISO time> <text>` lines, as GitHub's job logs and the agent host's sample file are written. */
export function splitTimestampedLine(raw: string): { time: number; text: string } | null {
  const m = raw.match(/^(\S+Z) (.*)$/);
  return m ? { time: Date.parse(m[1]), text: m[2] } : null;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Combine the probe's result with the agent host's samples. The agent CPU is
 * the median of the samples taken while every accepted track was live: from
 * the last accepted track (`rampEnd`) to the result (`resultTime`).
 */
export function summarize(
  result: ProbeResult,
  samples: HostSample[],
  rampEnd: number,
  resultTime: number,
): CapacitySummary {
  if (samples.length === 0) {
    throw new Error("no Host load samples");
  }
  let plateau = samples.filter((s) => s.time >= rampEnd && s.time <= resultTime + 5_000);
  if (plateau.length === 0) {
    plateau = samples.filter((s) => s.time <= resultTime + 5_000).slice(-3);
  }
  if (plateau.length === 0) {
    plateau = samples.slice(-3);
  }
  const gpu = plateau.some((s) => s.gpuUtil !== undefined);
  const agentCpu = median(plateau.map((s) => s.agentCpu));
  const cores = plateau[0].cores;
  const tracks = Math.max(result.tracks, 1);
  const perTrackVcpu = agentCpu / 100 / tracks;
  const [live, total] = result.liveAtEnd.split("/").map(Number);
  const degraded =
    result.oldestSustained === false || (Number.isFinite(live) && Number.isFinite(total) && live < total);
  return {
    result,
    cores,
    plateauSamples: plateau.length,
    agentCpu,
    ram: plateau[plateau.length - 1].ram,
    perTrackVcpu,
    tracksPerVcpu: perTrackVcpu > 0 ? 1 / perTrackVcpu : 0,
    tracksPer8Vcpu: Math.round((result.tracks * 8) / cores),
    gpu,
    gpuPeak: gpu ? Math.max(...samples.filter((s) => s.time <= resultTime + 5_000).map((s) => s.gpuUtil ?? 0)) : undefined,
    gpuAtPlateau: gpu ? Math.max(...plateau.map((s) => s.gpuUtil ?? 0)) : undefined,
    vram: gpu ? Math.max(...plateau.map((s) => s.vram ?? 0)) : undefined,
    degraded,
    publisherLimited: result.publisherHostBusy >= 0.85,
  };
}

const round = (n: number, step: number): string => (Math.round(n / step) * step).toFixed(2).replace(/\.?0+$/, "");

export function renderText(s: CapacitySummary): string {
  const r = s.result;
  const lines = [
    `Capacity: ${r.tracks} simultaneous transcribed tracks${r.label ? ` [${r.label}]` : ""}`,
    `  stop reason: ${r.stopReason}; oldest track still transcribing: ${r.oldestSustained ?? "n/a"}; live at end: ${r.liveAtEnd || "n/a"}; join retries: ${r.joinRetries}`,
    `  agent host: ${s.cores} vCPUs; agent CPU at full load (median of ${s.plateauSamples} samples): ${s.agentCpu.toFixed(0)}%; RAM ${s.ram}` +
      (s.gpu ? `; GPU util ${s.gpuAtPlateau}% at full load, peak ${s.gpuPeak}% during the ramp; VRAM ${s.vram} MiB` : ""),
    `  CPU per track: ~${s.perTrackVcpu.toFixed(2)} vCPU  |  tracks per vCPU: ~${s.tracksPerVcpu.toFixed(1)}`,
  ];
  if (s.degraded) {
    lines.push(
      `  WARNING: accepted-then-degraded ramp; ${r.tracks} is what the agent accepted, the sustained capacity is lower (last count with every track live).`,
    );
  }
  if (s.publisherLimited) {
    lines.push(
      `  WARNING: the publishers' host was ${Math.round(r.publisherHostBusy * 100)}% busy at the end; the count may be the probe's limit, use a bigger publisher instance.`,
    );
  }
  lines.push(
    s.gpu
      ? `  tracks per 8 vCPUs: not linear on a GPU host; measured ${r.tracks} with ${s.cores} vCPUs and one GPU (${(s.gpuPeak ?? 0) >= 90 ? "the GPU is the limit" : "the GPU has headroom; the admission rule or the CPU stopped the ramp"})`
      : `  tracks per 8 vCPUs: ~${s.tracksPer8Vcpu} (linear in cores: ${r.tracks} x 8 / ${s.cores})`,
  );
  return lines.join("\n");
}

/** Markdown for the GitHub job summary, ending with the row skeleton of the docs table. */
export function renderMarkdown(s: CapacitySummary): string {
  const r = s.result;
  const rows: Array<[string, string]> = [
    ["Simultaneous transcribed tracks", `**${r.tracks}**${s.degraded ? " (accepted, then degraded: the sustained figure is lower)" : ""}`],
    ["Stop reason", r.stopReason],
    ["Oldest track still transcribing at full load", r.oldestSustained === null ? "n/a" : r.oldestSustained ? "yes" : "no"],
    ["Tracks with a final in the last minute", r.liveAtEnd || "n/a"],
    ["Join retries", String(r.joinRetries)],
    ["Agent host", `${s.cores} vCPUs${s.gpu ? " + GPU" : ""}`],
    ["Agent CPU at full load", `${s.agentCpu.toFixed(0)} % (median of ${s.plateauSamples} samples)`],
    ["Agent RAM at full load", s.ram],
    ["CPU per track", `~${s.perTrackVcpu.toFixed(2)} vCPU`],
    ["Tracks per vCPU", `~${s.tracksPerVcpu.toFixed(1)}`],
    [
      "Tracks per 8 vCPUs",
      s.gpu
        ? `${r.tracks} measured with ${s.cores} vCPUs and one GPU (not linear: ${(s.gpuPeak ?? 0) >= 90 ? "the GPU is the limit" : "the GPU has headroom"})`
        : `~${s.tracksPer8Vcpu}`,
    ],
  ];
  if (s.publisherLimited) {
    rows.push(["Publishers host", `${Math.round(r.publisherHostBusy * 100)} % busy at the end: the count may be the probe's limit, not the agent's`]);
  }
  if (s.gpu) {
    rows.push(["GPU utilization", `${s.gpuAtPlateau} % at full load, ${s.gpuPeak} % peak`], ["VRAM", `${s.vram} MiB`]);
  }
  const docsRow =
    `| ${r.label || "<model>"} | <image> | <languages> | <quality> | ~${round(s.perTrackVcpu, 0.05)} vCPU${s.gpu ? ", plus GPU time" : ""} | ` +
    `${s.gpu ? `~${r.tracks} with ${s.cores} vCPUs and one GPU` : `~${s.tracksPer8Vcpu}`} |`;
  return [
    `### Capacity${r.label ? `: ${r.label}` : ""}`,
    "",
    "| Metric | Value |",
    "| --- | --- |",
    ...rows.map(([k, v]) => `| ${k} | ${v} |`),
    "",
    "Row for the docs capacity table (fill in image, languages and quality):",
    "",
    "```",
    docsRow,
    "```",
    "",
  ].join("\n");
}
