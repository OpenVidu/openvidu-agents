# Nemotron models

This directory holds the NVIDIA Nemotron 3.5 streaming ASR checkpoint used by the
`nemotron` local STT provider. The checkpoint is **not committed to git** (it is
~2.4 GB, F32) — only this README is tracked, so the directory exists.

## Download

```bash
./download-nemotron-model.sh
```

This downloads `nvidia/nemotron-3.5-asr-streaming-0.6b` into
`nemotron-models/nemotron-3.5-asr-streaming-0.6b/`. Run it before building the
nemotron Docker image (`build-nemotron.sh` expects this directory to be populated;
the checkpoint is baked into the image so it can run offline with
`HF_HUB_OFFLINE=1`).

## License

The model weights are distributed by NVIDIA under the **OpenMDW-1.1** license.
Review its terms before distribution. The plugin code itself is Apache-2.0.
