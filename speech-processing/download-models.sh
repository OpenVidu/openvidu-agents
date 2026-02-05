#!/bin/bash
# Download speech recognition models from GitHub Releases
# Run this script after cloning the repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_RELEASE="stt-local-models-1.0"
BASE_URL="https://github.com/OpenVidu/openvidu-agents/releases/download/${MODELS_RELEASE}"

echo "========================================"
echo "Downloading STT Models"
echo "========================================"
echo ""

cd "$SCRIPT_DIR"

# Download and extract vosk models
echo "→ Downloading vosk models..."
curl -L -o vosk-models.tar.gz "${BASE_URL}/vosk-models.tar.gz"
echo "→ Extracting vosk models..."
tar -xzf vosk-models.tar.gz
rm vosk-models.tar.gz
echo "✓ vosk models ready"
echo ""

# Download and extract sherpa-onnx models
echo "→ Downloading sherpa-onnx models..."
curl -L -o sherpa-onnx-streaming-models.tar.gz "${BASE_URL}/sherpa-onnx-streaming-models.tar.gz"
echo "→ Extracting sherpa-onnx models..."
tar -xzf sherpa-onnx-streaming-models.tar.gz
rm sherpa-onnx-streaming-models.tar.gz
echo "✓ sherpa-onnx models ready"
echo ""

echo "========================================"
echo "✓ All models downloaded successfully"
echo "========================================"
echo ""
echo "Models available in:"
echo "  - ${SCRIPT_DIR}/vosk-models/"
echo "  - ${SCRIPT_DIR}/sherpa-onnx-streaming-models/"
echo ""
