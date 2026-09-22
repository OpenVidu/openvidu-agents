# Transcription capacity with remote publishers

How many participants can one agent host transcribe at the same time, when the
host runs nothing but the OpenVidu deployment and the agent?

A browser-based probe that ran the deployment, the agent and a browser
publishing every track on one machine was retired in favour of this one: on a
4-vCPU box it saturated the box (the browser starved first) at 12-13 tracks
whatever the agent did. The two scripts here split the roles:

| Script | Runs on | Does |
| --- | --- | --- |
| `agent-host.ts` | the machine under test | `start`: configure the agent provider in the local deployment, start it, wait for the agent worker. `hold`: keep it up while the publishers run, logging `Host load [...]` (agent CPU/RAM, other containers, host total, VRAM + GPU utilization) every 15 s. `stop`: dump the agent log tail, stop the deployment. |
| `headless-capacity-probe.ts` | any other machine | LiveKit Node SDK publishers stream a WAV in a loop; a track counts when the agent's final transcription of it (`lk.transcription` text stream) arrives within 60 s; the ramp adds one publisher at a time and re-checks the oldest track at the end; prints `CAPACITY RESULT:`. Each publisher costs the probe process about 13 % of a core (Opus encoding), so the default `c6i.2xlarge` (8 vCPU) carries the 40-track hard cap with margin; the result line reports the publisher host's own load. |

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

The `sherpa-model` input measures any other model of the image (for example the
smaller English zipformer `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06`)
with the same ramp; the label of the result line carries the model.

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
