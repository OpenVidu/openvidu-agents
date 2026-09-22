/**
 * Turn one run of the "Speech Processing agent capacity, remote publishers"
 * workflow into the numbers the docs' capacity table uses.
 *
 *   npm run capacity:summarize -- <run-id> [--repo OpenVidu/openvidu-agents]
 *
 * Needs the GitHub CLI (`gh`) logged in. Reads the Publishers job log for the
 * `CAPACITY RESULT` line and the Agent host job log for the `Host load [...]`
 * samples taken while the ramp was complete (between the last accepted track
 * and the result), then prints:
 *   - the measured track count and how it ended,
 *   - the agent's CPU at that load and the derived cost per track,
 *   - the "tracks per 8 vCPUs" figure (linear in cores on CPU hosts; on GPU
 *     hosts the GPU is the limit, so the measured count is reported as is),
 *   - a Markdown row skeleton for openvidu.io's live-captions page.
 */
import { execSync } from "child_process";

const args = process.argv.slice(2);
const runId = args.find((a) => /^\d+$/.test(a));
const repoIndex = args.indexOf("--repo");
const repo = repoIndex >= 0 ? args[repoIndex + 1] : "OpenVidu/openvidu-agents";
if (!runId) {
  console.error("usage: summarize-run.ts <run-id> [--repo owner/name]");
  process.exit(2);
}

function gh(command: string): string {
  return execSync(`gh ${command}`, { encoding: "utf8", maxBuffer: 256 * 1024 * 1024 });
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

const jobs: Array<{ name: string; databaseId: number; conclusion: string }> = JSON.parse(
  gh(`run view ${runId} -R ${repo} --json jobs`),
).jobs;
const jobId = (name: string): number => {
  const job = jobs.find((j) => j.name === name);
  if (!job) {
    throw new Error(`job "${name}" not found in run ${runId} (jobs: ${jobs.map((j) => j.name).join(", ")})`);
  }
  return job.databaseId;
};
const publishersLog = gh(`api repos/${repo}/actions/jobs/${jobId("Publishers")}/logs`);
const agentHostLog = gh(`api repos/${repo}/actions/jobs/${jobId("Agent host")}/logs`);

// GitHub prefixes every log line with its own timestamp; the scripts add a
// second ISO timestamp of their own. Both are read here.
type Line = { ghTime: number; text: string };
const lines = (log: string): Line[] =>
  log.split("\n").flatMap((raw) => {
    const m = raw.match(/^(\S+Z) (.*)$/);
    return m ? [{ ghTime: Date.parse(m[1]), text: m[2] }] : [];
  });

const pub = lines(publishersLog);
const result = pub.find((l) => l.text.includes("CAPACITY RESULT:"));
if (!result) {
  console.error(`run ${runId}: no CAPACITY RESULT line in the Publishers job (did the probe run?)`);
  process.exit(1);
}
const tracks = Number(result.text.match(/CAPACITY RESULT: (\d+) simultaneous/)![1]);
const label = result.text.match(/headless publishers \[([^\]]*)\]/)?.[1] ?? "";
const detail = (key: string): string =>
  result.text.match(new RegExp(`${key}: ([^;)]+)`))?.[1]?.trim() ?? "?";
const stopReason = detail("stop reason");
const oldestSustained = detail("oldest track still transcribing at full load");
const liveAtEnd = result.text.match(/tracks with a final in the last \d+s: (\d+\/\d+)/)?.[1] ?? "?";
const joinRetries = detail("join retries");
const lastAccepted = [...pub].reverse().find((l) => /Capacity so far: \d+ simultaneous/.test(l.text));
const rampEnd = lastAccepted ? lastAccepted.ghTime : result.ghTime - 60_000;

// Host load samples of the agent host while every accepted track was live.
const sampleRe =
  /Host load \[agent CPU: ([\d.]+)% \| RAM: ([^|]+?) \| other containers: [^|]+ \| host: (\d+)% of (\d+)% busy(?: \| VRAM: (\d+) MiB \| GPU util: (\d+)%)?\]/;
type Sample = { time: number; agentCpu: number; ram: string; hostBusy: number; cores: number; vram?: number; gpuUtil?: number };
const samples: Sample[] = lines(agentHostLog).flatMap((l) => {
  const m = l.text.match(sampleRe);
  if (!m) return [];
  return [{
    time: l.ghTime,
    agentCpu: Number(m[1]),
    ram: m[2].trim(),
    hostBusy: Number(m[3]),
    cores: Number(m[4]) / 100,
    vram: m[5] ? Number(m[5]) : undefined,
    gpuUtil: m[6] ? Number(m[6]) : undefined,
  }];
});
if (samples.length === 0) {
  console.error(`run ${runId}: no "Host load" samples in the Agent host job`);
  process.exit(1);
}
let plateau = samples.filter((s) => s.time >= rampEnd && s.time <= result.ghTime + 5_000);
if (plateau.length === 0) {
  plateau = samples.filter((s) => s.time <= result.ghTime).slice(-3);
}
const gpu = plateau.some((s) => s.gpuUtil !== undefined);
// Peak GPU utilization over the whole ramp: saturation shows while tracks are
// still being added, before the plateau samples.
const gpuPeak = gpu
  ? Math.max(...samples.filter((s) => s.time <= result.ghTime + 5_000).map((s) => s.gpuUtil ?? 0))
  : undefined;
const degraded = oldestSustained !== "true" || (/^(\d+)\/(\d+)$/.test(liveAtEnd) && liveAtEnd.split("/")[0] !== liveAtEnd.split("/")[1]);
const agentCpu = median(plateau.map((s) => s.agentCpu));
const cores = plateau[0].cores;
const perTrack = agentCpu / 100 / tracks;
const tracksPer8 = Math.round((tracks * 8) / cores);
const ram = plateau[plateau.length - 1].ram;
const gpuUtil = gpu ? Math.max(...plateau.map((s) => s.gpuUtil ?? 0)) : undefined;
const vram = gpu ? Math.max(...plateau.map((s) => s.vram ?? 0)) : undefined;

console.log(`Run ${runId} (${repo}) — ${label || "no label"}`);
console.log(`  tracks: ${tracks}  (stop reason: ${stopReason}; oldest track still transcribing: ${oldestSustained}; live at end: ${liveAtEnd}; join retries: ${joinRetries})`);
console.log(`  agent host: ${cores} vCPUs; agent CPU at full load (median of ${plateau.length} samples): ${agentCpu.toFixed(0)}%; RAM ${ram}` +
  (gpu ? `; GPU util up to ${gpuUtil}%; VRAM ${vram} MiB` : ""));
console.log(`  CPU per track: ~${perTrack.toFixed(2)} vCPU`);
if (degraded) {
  console.log(`  WARNING: accepted-then-degraded ramp (oldest track sustained: ${oldestSustained}, live at end: ${liveAtEnd}); ` +
    `${tracks} is the count the agent accepted, the sustained capacity is lower. Report the last count at which every track was still live.`);
}
if (gpu) {
  console.log(`  tracks per 8 vCPUs: not linear on a GPU host; measured ${tracks} with ${cores} vCPUs and one GPU (GPU util peaked at ${gpuPeak}% during the ramp: ` +
    `${(gpuPeak ?? 0) >= 90 ? "the GPU is the limit" : "the GPU has headroom, the admission rule or the CPU stopped the ramp"})`);
} else {
  console.log(`  tracks per 8 vCPUs: ~${tracksPer8} (linear in cores: ${tracks} × 8 / ${cores})`);
}
console.log("\nMarkdown row for docs/docs/ai/live-captions.md (fill in the quality column):");
console.log(
  `| ${label || "<model>"} | <image> | <languages> | <quality> | ~${(Math.round(perTrack * 20) / 20).toFixed(2).replace(/0$/, "")} vCPU${gpu ? ", plus GPU time" : ""} | ` +
    `${gpu ? `~${tracks} with ${cores} vCPUs and one GPU` : `~${tracksPer8}`} |`,
);
