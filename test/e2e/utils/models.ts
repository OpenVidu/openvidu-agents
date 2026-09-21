/**
 * NVIDIA Nemotron 3.5 streaming ASR served by the `sherpa` provider.
 *
 * The CPU image (agent-speech-processing-sherpa) bakes the int8 export and the
 * cuda12 image the float32 one, because int8 graphs carry quantized ops that
 * ONNX Runtime's CUDA provider cannot run (they would silently execute on the
 * CPU). The model directory name therefore follows the acceleration in use:
 * STT_ACCEL is set on GPU runs (see LocalDeployment.configureProvider).
 */
export const SHERPA_NEMOTRON_MODEL = process.env.STT_ACCEL
  ? "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-2026-06-11"
  : "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-int8-2026-06-11";
