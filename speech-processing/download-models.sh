#!/bin/bash
# Download the speech recognition models baked into the local-provider images
# (vosk and sherpa). Run it after cloning and before building those images:
#
#   ./download-models.sh [--accel cpu|cuda12]
#
# - The vosk models and the sherpa zipformer models come as bundles from the
#   "stt-local-models" GitHub release of this repository.
# - The NVIDIA Nemotron 3.5 streaming model served by the sherpa provider comes
#   in two variants; --accel (default "cpu", or $STT_ACCEL when set) selects the
#   one the target image bakes in:
#     cpu    -> int8 export from the k2-fsa/sherpa-onnx "asr-models" release
#               (image agent-speech-processing-sherpa)
#     cuda12 -> float32 export from Hugging Face (image agent-speech-processing-sherpa-cuda12).
#               int8 graphs carry quantized ops that ONNX Runtime's CUDA provider
#               cannot run, so on a GPU they would silently execute on the CPU.
#   Both are pinned, because upstream overwrites its release assets in place: the
#   tarball is checked against a sha256 and the Hugging Face files are fetched
#   from a fixed revision and checked one by one.
#
# Idempotent: bundles and model directories already present (and matching their
# pins) are skipped, so CI can call it unconditionally.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_RELEASE="stt-local-models-1.0"
BASE_URL="https://github.com/OpenVidu/openvidu-agents/releases/download/${MODELS_RELEASE}"
SHERPA_MODELS_DIR="$SCRIPT_DIR/sherpa-onnx-streaming-models"

# Nemotron 3.5 streaming ASR, 320 ms chunks (the latency the nemotron provider
# defaulted to). Other chunk sizes (80/160/560/1120 ms) exist upstream; users who
# want one add it to a custom image built on the -base image.
NEMOTRON_INT8_DIR="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-int8-2026-06-11"
NEMOTRON_INT8_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NEMOTRON_INT8_DIR}.tar.bz2"
NEMOTRON_INT8_SHA256="5f311142337a5c161e92d49f7a3009d8607d3836f39d610bff5307c74d1d2c53"

NEMOTRON_FP32_DIR="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-2026-06-11"
NEMOTRON_FP32_HF_REPO="csukuangfj2/${NEMOTRON_FP32_DIR}"
NEMOTRON_FP32_HF_REVISION="9fafa57a2ba5f3bab0823274b7543fb353e53ffd"
# "<file> <sha256>" pairs. encoder.onnx is the graph, encoder.data its weights
# (ONNX external data, loaded by name from the same directory).
NEMOTRON_FP32_FILES=(
    "tokens.txt 729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a"
    "encoder.onnx 93efa4bdad4a3ca47d171f22e31c5828fc86cad30b08b4af999f25122fd65925"
    "decoder.onnx f9c59ee6fa130bc2ba349dbcbba7c74a4a960a98d4196295ba4e8e5d6bde6b68"
    "joiner.onnx a6bd74c0a31cbde0da0368c1e29d171752108c7ad1231630e079a7d97ceee6f0"
    "encoder.data 7584f85df76bc9ae6fbdfa53aa8d97b07a842525d1c501d536d77fd9e4f57ac7"
)

usage() {
    sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

ACCEL="${STT_ACCEL:-cpu}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --accel)
            [[ $# -ge 2 ]] || { echo "[ERROR] --accel needs a value (cpu|cuda12)"; exit 1; }
            ACCEL="$2"
            shift 2
            ;;
        --accel=*)
            ACCEL="${1#--accel=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done
case "$ACCEL" in
    cpu|cuda12) ;;
    "") ACCEL="cpu" ;;
    *)
        echo "[ERROR] Unsupported --accel '$ACCEL' (expected cpu or cuda12)"
        exit 1
        ;;
esac

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

# download_file URL DEST [SHA256]: atomic download (DEST.part -> DEST) with
# retries; with SHA256 the file is verified and removed on mismatch.
download_file() {
    local url="$1" dest="$2" expected="${3:-}"
    rm -f "$dest.part"
    curl -fL --retry 3 --retry-delay 5 --progress-bar -o "$dest.part" "$url"
    if [[ -n "$expected" ]]; then
        local actual
        actual="$(sha256_of "$dest.part")"
        if [[ "$actual" != "$expected" ]]; then
            rm -f "$dest.part"
            echo "[ERROR] sha256 mismatch for $(basename "$dest")"
            echo "        expected $expected"
            echo "        got      $actual"
            echo "        The upstream asset changed; review it before updating the pin in this script."
            exit 1
        fi
    fi
    mv "$dest.part" "$dest"
}

# has_model_dirs DIR PATTERN: true when DIR holds a model sub-directory matching
# PATTERN (the bundle's own models, so a directory that only holds a Nemotron
# export does not count as "bundle present").
has_model_dirs() {
    [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -maxdepth 1 -type d -name "$2" 2>/dev/null | head -n 1)" ]]
}

download_bundle() {
    local name="$1" pattern="$2"
    if has_model_dirs "$SCRIPT_DIR/$name" "$pattern"; then
        echo "→ $name bundle already present, skipping"
        return
    fi
    echo "→ Downloading $name bundle..."
    download_file "${BASE_URL}/${name}.tar.gz" "$SCRIPT_DIR/${name}.tar.gz"
    echo "→ Extracting $name bundle..."
    tar -xzf "$SCRIPT_DIR/${name}.tar.gz" -C "$SCRIPT_DIR"
    rm -f "$SCRIPT_DIR/${name}.tar.gz"
    echo "✓ $name ready"
}

download_nemotron_int8() {
    local dir="$SHERPA_MODELS_DIR/$NEMOTRON_INT8_DIR"
    local marker="$dir/.source.sha256"
    if [[ -f "$marker" ]] && [[ "$(cat "$marker")" == "$NEMOTRON_INT8_SHA256" ]]; then
        echo "→ $NEMOTRON_INT8_DIR already present (pinned tarball), skipping"
        return
    fi
    if [[ -f "$dir/encoder.int8.onnx" && -f "$dir/decoder.int8.onnx" && -f "$dir/joiner.int8.onnx" && -f "$dir/tokens.txt" ]]; then
        echo "→ $NEMOTRON_INT8_DIR already present (not downloaded by this script, left as is)"
        return
    fi
    echo "→ Downloading $NEMOTRON_INT8_DIR (int8, ~450 MB)..."
    local tarball="$SHERPA_MODELS_DIR/${NEMOTRON_INT8_DIR}.tar.bz2"
    mkdir -p "$SHERPA_MODELS_DIR"
    download_file "$NEMOTRON_INT8_URL" "$tarball" "$NEMOTRON_INT8_SHA256"
    echo "→ Extracting $NEMOTRON_INT8_DIR..."
    rm -rf "$dir"
    tar -xjf "$tarball" -C "$SHERPA_MODELS_DIR"
    rm -f "$tarball"
    # Sample clips are not needed in the image.
    rm -rf "$dir/test_wavs"
    echo "$NEMOTRON_INT8_SHA256" > "$marker"
    echo "✓ $NEMOTRON_INT8_DIR ready"
}

download_nemotron_fp32() {
    local dir="$SHERPA_MODELS_DIR/$NEMOTRON_FP32_DIR"
    local base_url="https://huggingface.co/${NEMOTRON_FP32_HF_REPO}/resolve/${NEMOTRON_FP32_HF_REVISION}"
    mkdir -p "$dir"
    local entry file expected missing=0
    for entry in "${NEMOTRON_FP32_FILES[@]}"; do
        file="${entry%% *}"
        expected="${entry##* }"
        if [[ -f "$dir/$file" ]] && [[ "$(sha256_of "$dir/$file")" == "$expected" ]]; then
            continue
        fi
        if [[ $missing -eq 0 ]]; then
            echo "→ Downloading $NEMOTRON_FP32_DIR (float32, ~2.4 GB) from Hugging Face @ ${NEMOTRON_FP32_HF_REVISION:0:8}..."
        fi
        missing=1
        echo "  · $file"
        download_file "$base_url/$file" "$dir/$file" "$expected"
    done
    if [[ $missing -eq 0 ]]; then
        echo "→ $NEMOTRON_FP32_DIR already present (all files match their pins), skipping"
    fi
    if [[ ! -f "$dir/README.md" ]]; then
        cat > "$dir/README.md" <<EOF
# Introduction

This model is the float32 sherpa-onnx export of https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
with 320 ms chunks, downloaded from https://huggingface.co/${NEMOTRON_FP32_HF_REPO}
at revision ${NEMOTRON_FP32_HF_REVISION} by download-models.sh.

It is meant for GPU (CUDA) inference: the int8 export cannot run on ONNX Runtime's
CUDA provider. Use per-stream language strings such as "en", "es-ES" or "auto".
EOF
    fi
    echo "✓ $NEMOTRON_FP32_DIR ready"
}

echo "========================================"
echo "Downloading STT Models (accel: $ACCEL)"
echo "========================================"
echo ""

download_bundle vosk-models 'vosk-model-*'
echo ""
download_bundle sherpa-onnx-streaming-models 'sherpa-onnx-streaming-zipformer-*'
echo ""

if [[ "$ACCEL" == "cuda12" ]]; then
    download_nemotron_fp32
else
    download_nemotron_int8
fi

echo ""
echo "========================================"
echo "✓ All models downloaded successfully"
echo "========================================"
echo ""
echo "Models available in:"
echo "  - ${SCRIPT_DIR}/vosk-models/"
echo "  - ${SCRIPT_DIR}/sherpa-onnx-streaming-models/"
echo ""
