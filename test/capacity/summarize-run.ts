/**
 * Turn one finished run of the "Speech Processing agent capacity, remote
 * publishers" workflow into the numbers the docs' capacity table uses.
 *
 *   npm run capacity:summarize -- <run-id> [--repo OpenVidu/openvidu-agents]
 *
 * Needs the GitHub CLI (`gh`) logged in. Reads the Publishers job log for the
 * `CAPACITY RESULT` line and the Agent host job log for the `Host load [...]`
 * samples, and prints the same summary the workflow writes to its job summary
 * (tracks and how the ramp ended, agent CPU at full load, CPU per track,
 * tracks per vCPU and per 8 vCPUs, a Markdown row for the docs table).
 */
import { execSync } from "child_process";
import {
  HostSample,
  PROGRESS_RE,
  RESULT_MARKER,
  parseResultLine,
  parseSampleLine,
  renderMarkdown,
  renderText,
  splitTimestampedLine,
  summarize,
} from "./summary";

const args = process.argv.slice(2);
const runId = args.find((a) => /^\d+$/.test(a));
const repoIndex = args.indexOf("--repo");
const repo = repoIndex >= 0 ? args[repoIndex + 1] : "OpenVidu/openvidu-agents";
if (!runId) {
  console.error("usage: summarize-run.ts <run-id> [--repo owner/name] [--markdown]");
  process.exit(2);
}

function gh(command: string): string {
  return execSync(`gh ${command}`, { encoding: "utf8", maxBuffer: 256 * 1024 * 1024 });
}

const jobs: Array<{ name: string; databaseId: number }> = JSON.parse(
  gh(`run view ${runId} -R ${repo} --json jobs`),
).jobs;
const jobId = (name: string): number => {
  const job = jobs.find((j) => j.name === name);
  if (!job) {
    throw new Error(`job "${name}" not found in run ${runId} (jobs: ${jobs.map((j) => j.name).join(", ")})`);
  }
  return job.databaseId;
};
const publishersLines = gh(`api repos/${repo}/actions/jobs/${jobId("Publishers")}/logs`)
  .split("\n")
  .flatMap((raw) => {
    const l = splitTimestampedLine(raw);
    return l ? [l] : [];
  });
const resultLine = publishersLines.find((l) => l.text.includes(RESULT_MARKER));
const result = resultLine && parseResultLine(resultLine.text);
if (!resultLine || !result) {
  console.error(`run ${runId}: no ${RESULT_MARKER} line in the Publishers job (did the probe run?)`);
  process.exit(1);
}
const lastAccepted = [...publishersLines].reverse().find((l) => PROGRESS_RE.test(l.text));
const rampEnd = lastAccepted ? lastAccepted.time : resultLine.time - 60_000;

const samples: HostSample[] = gh(`api repos/${repo}/actions/jobs/${jobId("Agent host")}/logs`)
  .split("\n")
  .flatMap((raw) => {
    const l = splitTimestampedLine(raw);
    const s = l && parseSampleLine(l.text, l.time);
    return s ? [s] : [];
  });
if (samples.length === 0) {
  console.error(`run ${runId}: no "Host load" samples in the Agent host job`);
  process.exit(1);
}

const summary = summarize(result, samples, rampEnd, resultLine.time);
console.log(`Run ${runId} (${repo})`);
console.log(renderText(summary));
if (args.includes("--markdown")) {
  console.log("\n" + renderMarkdown(summary));
}
