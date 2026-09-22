# Transcription capacity with remote publishers

How many participants can one agent host transcribe at the same time, when the
host runs nothing but the OpenVidu deployment and the agent?

A browser-based probe that ran the deployment, the agent and a browser
publishing every track on one machine was retired in favour of this one: on a
4-vCPU box it saturated the box (the browser starved first) at 12-13 tracks
whatever the agent did. The two scripts here split the roles:

| Script | Runs on | Does |
| --- | --- | --- |
| `agent-host.ts` | the machine under test | `start`: configure the agent provider in the local deployment, start it, wait for the agent worker. `hold`: keep it up while the publishers run, logging and recording `Host load [...]` (agent CPU/RAM, other containers, host total, VRAM + GPU utilization) every 15 s. `stop`: dump the agent log tail, stop the deployment. |
| `headless-capacity-probe.ts` | any other machine | LiveKit Node SDK publishers stream a WAV in a loop, the publishers of each room in their own worker process (`publisher-worker.ts`; one Node process saturates at 15-20 publishers and stops counting finals that keep arriving); a track counts when the agent's final transcription of it (`lk.transcription` text stream) arrives within 60 s; the ramp adds one publisher at a time and re-checks the oldest track at the end; prints `CAPACITY RESULT:`. Each publisher costs the probe process about 13 % of a core (Opus encoding), so the default `c6i.2xlarge` (8 vCPU) carries the 40-track hard cap with margin; the result line reports the publisher host's own load. |

## CI: `speech-processing-capacity-remote-publishers.yml`

Two EC2 runners in the same subnet. The agent host builds (or pulls) the sherpa
image, starts the local deployment with GPU passthrough when `accel=cuda12`,
prints `AGENT HOST READY url=ws://<private-ip>:7880` and holds. The publishers
instance is launched only when that step has succeeded (a GitHub-hosted job
polls the Actions REST API for it), so it is not billed during the image build;
it then checks connectivity, runs the probe against the agent host's private IP
and prints the result, and its completion releases the hold. Nothing is exposed
publicly; the deployment is reached exactly as a LAN client would.

Keep `deployment-edition` at `community`: the
local Pro server runs in evaluation mode, which allows 8 participants across
all rooms, agent participants included, and closes rooms after 5 minutes, so a
ramp on it stops at 6 tracks with `HTTP 500` joins (`OpenVidu Pro evaluation
mode only allows a maximum of 8 participants across all rooms` in the server
log). The agent image is the same on both editions and receives the Pro license
through the operator.

With its default `job_executor: thread`, the agent runs every Room of a local
provider as a thread of one process, and that process's single event loop
saturates around 20 tracks whatever the model: the SDK's events lag, the
Rooms' signaling times out and no final reaches any publisher any more while
decoding continues. The `job-executor` input (`thread` or `process`) sets that
property in the agent host's `agent-speech-processing.yaml`; `process` gives
every Room its own process, loop and model copy, so the probe measures the
model instead of that ceiling.

The `provider` (`sherpa` or `vosk`) and `model` inputs select what is measured:
any model directory of that provider's image, with the same ramp for all of
them. Both images are built from the sources (or pulled when
`docker-tag-agent-speech-processing` is set); `vosk` ignores `accel`. The
result label carries the provider and the model.

For the CPU vs GPU comparison, dispatch it twice with the same vCPU count and RAM:

| Input | GPU host | CPU host |
| --- | --- | --- |
| `agent-instance-type` | `g4dn.xlarge` (T4, 4 vCPU, 16 GiB) | `m6i.xlarge` (4 vCPU, 16 GiB) |
| `accel` | `cuda12` (float32 export) | `cpu` (int8 export) |

Read `CAPACITY RESULT` in the Publishers job and the `Host load` lines in the
Agent host job: the agent's CPU per track, VRAM, GPU utilization, and whether
`host` busy stays close to the containers' sum (it should, nothing else runs
there). Note the CPUs differ (Cascade Lake on g4dn, Ice Lake on m6i), so quote
the instance types with the numbers.

**AWS prerequisites** (outside these repositories): one security group per
role, both in the VPC of the runner subnet, stored in the repository secrets
`AWS_SECURITY_GROUP_ID_AGENT_HOST` and `AWS_SECURITY_GROUP_ID_PUBLISHERS` (the
workflow falls back to `AWS_SECURITY_GROUP_ID` when one is missing). The
publishers group needs no inbound rule: the publishers open every connection
and security groups are stateful. The agent-host group must allow inbound
7880/tcp (LiveKit HTTP/WS through Caddy), 7881/tcp (ICE over TCP) and
7900-7999/udp (media) **with the publishers group as source**; outbound stays
at the default "all traffic" (GitHub, image pulls, model downloads). The
standard runner AMI is used for the publishers, the GPU AMI for a `cuda12`
agent host.

## Running the probe by hand

Against a deployment already running on this or another machine:

```bash
cd test && npm ci
LIVEKIT_URL=ws://<deployment-ip>:7880 CAPACITY_HARD_CAP=12 npm run capacity:headless
```

Starting the deployment with the agent on the current machine (needs
`openvidu-local-deployment` checked out next to this repository, Docker, and
`OPENVIDU_PRO_LICENSE` for the sherpa image):

```bash
STT_ACCEL=cuda12 npm run capacity:agent-host -- start   # or without STT_ACCEL for the CPU image
npm run capacity:agent-host -- hold                     # Ctrl-C when done, or HOLD_MAX_MINUTES=10
npm run capacity:agent-host -- stop
```

Provider and model default to the sherpa provider with the Nemotron 3.5 model
(`e2e/utils/models.ts`, forced English); override with `CAPACITY_PROVIDER_JSON`,
e.g. `{"vosk":{"model":"vosk-model-en-us-0.22-lgraph","use_silero_vad":false}}`.

## Rebuilding the capacity table of the docs

The table under "Capacity estimate of local provider models" in
`openvidu.io/docs/docs/ai/live-captions.md` is one dispatch per row, all on the
`community` edition with the default publishers instance:

| Docs row | `provider` | `model` | `agent-instance-type` / `accel` |
| --- | --- | --- | --- |
| Vosk small models | `vosk` | `vosk-model-small-en-in-0.4` (the only small English model in the image; its transcripts of the US English fixture are poor, which does not matter for a capacity count) | `m6i.2xlarge` / `cpu` |
| Vosk `vosk-model-en-us-0.22-lgraph` | `vosk` | empty | `m6i.2xlarge` / `cpu` |
| Sherpa Kroko | `sherpa` | `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | `m6i.2xlarge` / `cpu` |
| Sherpa multilingual zipformer | `sherpa` | `sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10` | `m6i.2xlarge` / `cpu` |
| Sherpa Nemotron 3.5 int8 | `sherpa` | empty | `m6i.2xlarge` / `cpu` |
| Sherpa Nemotron 3.5 float32 | `sherpa` | empty | `g4dn.2xlarge` / `cuda12` (the T4 is the limit; `g4dn.xlarge` gives the 4-vCPU figure) |

A 4-vCPU host (`m6i.xlarge`) gives half the tracks of the 8-vCPU one for every
CPU model; measure whichever is cheaper and scale. Every run writes its figures
to the GitHub **job summaries**: the Publishers job reports the tracks and how the
ramp ended, the Agent host job adds the CPU behind them (agent CPU at full load,
CPU per track, tracks per vCPU and per 8 vCPUs, GPU utilization) and the row
skeleton for the docs table. The same figures for any past run:

```sh
npm run capacity:summarize -- <run-id>
```

prints the track count and how the ramp ended, the agent's CPU at full load
and the derived cost per track, the tracks-per-8-vCPUs figure, a warning when
the ramp was accepted but degraded or when the publishers' own host was near
saturation (then the count is the probe's limit, not the agent's), and a Markdown
row skeleton for the docs table (the quality column is filled by hand from the
e2e accuracy runs). Adding a model to the table is adding it to the image
(`speech-processing/download-models.sh`), dispatching one run and pasting the
row.
