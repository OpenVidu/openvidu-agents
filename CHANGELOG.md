# Changelog

Notable changes of the OpenVidu agents built from this repository (the `agent-speech-processing`
base, cloud and vosk images), which are versioned in lockstep with OpenVidu. The sherpa images are
built together with `openvidu-agents-pro`, and their changes are listed in its CHANGELOG.

## [Unreleased]

### Added

- **OpenVidu agent mark**: the speech-processing agent joins every Room with the participant
  attribute `openvidu.agent: speech-processing`, which tells it apart from any other LiveKit agent.
  Its key and value come from `openviduagentutils` (`openvidu_agent_attributes`).

### Changed

- **livekit-agents 1.8.12** (from 1.8.2), with every cloud STT plugin at the same version.
- **Smaller images with fewer vulnerabilities**: the base image installs only runtime packages,
  with every pending OS update, and no longer contains pip, gcc, g++, python3-dev or git (563 MB to
  123 MB). The cloud and vosk images install their Python dependencies in a separate build stage
  (cloud: 1168 MB to 722 MB). Images that add files to a base image with `COPY`, as the
  documentation shows for custom models, are not affected. Installing Python packages on top of a
  base image now needs `python -m ensurepip` or a build stage of its own.
- **Vosk Python bindings** updated to the latest vosk-api commit, which fixes the model lookup by
  language and the cleanup of models and recognizers that failed to load.
- **Less memory per agent**: the agent no longer preloads livekit-agents' own VAD and end-of-turn
  models, which it never uses (with the vosk English model, about 124 MiB at idle instead of
  338 MiB in 3.9.0). Set `LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE=1` to preload them again.
- **Silero VAD from livekit-agents**: the VAD used with `use_silero_vad`, with providers that do not
  stream and with Speechmatics is now livekit-agents' own Silero VAD instead of the
  `livekit-plugins-silero` plugin. It segments speech the same way with less than half the CPU, and
  the images no longer contain the plugin's ONNX Runtime (about 80 MB). Providers that do not stream
  now run one VAD per audio track instead of two.
