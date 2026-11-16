# Changelog

Notable changes of the OpenVidu agents built from this repository (the `agent-speech-processing`
base, cloud and vosk images), which are versioned in lockstep with OpenVidu. The sherpa images are
built together with `openvidu-agents-pro`, and their changes are listed in its CHANGELOG.

## [Unreleased]

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
