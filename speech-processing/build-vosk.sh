#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="openvidu/agent-speech-processing-vosk"
TAG="${TAG:-main}"
NO_CACHE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            echo "Usage: $0 [--no-cache] [--tag TAG]"
            exit 1
            ;;
    esac
done

# Check for Vosk models
if [[ ! -d "$PUBLIC_REPO/speech-processing/vosk-models" ]]; then
    echo "[ERROR] Vosk models not found in public repo"
    echo "[ERROR] Run: $PUBLIC_REPO/speech-processing/download-models.sh"
    exit 1
fi

# Copy livekit-plugins-vosk to the build context
rm -rf "$SCRIPT_DIR/livekit-plugins-vosk"
cp -r "$SCRIPT_DIR/../livekit-plugins-vosk" "$SCRIPT_DIR/livekit-plugins-vosk"

# Setup cleanup trap
cleanup() {
    echo "Cleaning up build context..."
    rm -rf "$SCRIPT_DIR/livekit-plugins-vosk"
    echo "Cleanup complete"
}
trap cleanup EXIT

# Build the Docker image
docker build $NO_CACHE -t "$IMAGE_NAME:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk" "$SCRIPT_DIR"