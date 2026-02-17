#!/bin/bash
# Build script for openvidu/agent-speech-processing-vosk
#
# Usage:
#   ./build-vosk.sh [--no-cache] [--tag TAG] [--local-only]
# Flags:
#   --no-cache: Do not use Docker build cache
#   --tag TAG: Specify a custom Docker tag (default: main)
#   --local-only: Build only for local platform (no multi-arch, no buildx)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="openvidu/agent-speech-processing-vosk"
PLATFORMS="linux/amd64,linux/arm64"

# Flags
NO_CACHE=""
TAG="${TAG:-main}"
LOCAL_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --tag)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "[ERROR] --tag requires a value"
                exit 1
            fi
            TAG="$2"
            shift 2
            ;;
        --local-only)
            LOCAL_ONLY=true
            shift
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            echo "Usage: $0 [--no-cache] [--tag TAG] [--local-only]"
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

echo ""
echo "Build configuration:"
echo "  Image: $IMAGE_NAME:$TAG"
echo "  Local only: $LOCAL_ONLY"
if [[ "$LOCAL_ONLY" == "true" ]]; then
    echo "  Platform: $(uname -m)"
else
    echo "  Platforms: $PLATFORMS"
fi

# Build the Docker image
if [[ "$LOCAL_ONLY" == "true" ]]; then
    # Build for local platform only and load into Docker
    echo "Building for local platform with --load..."
    docker buildx build --load $NO_CACHE -t "$IMAGE_NAME:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk" "$SCRIPT_DIR"
else

    # Setup buildx builder for multi-platform builds
    BUILDER_NAME="multiarch-builder"
    if [[ "$LOCAL_ONLY" == "false" ]]; then
        echo ""
        echo "Setting up buildx builder for multi-platform builds..."
        
        # Check if builder exists and is usable
        if ! docker buildx inspect "$BUILDER_NAME" > /dev/null 2>&1; then
            echo "Creating buildx builder: $BUILDER_NAME"
            docker buildx create --name "$BUILDER_NAME" --driver docker-container --bootstrap --use
        else
            echo "Using existing buildx builder: $BUILDER_NAME"
            docker buildx use "$BUILDER_NAME"
        fi
    fi

    # Build for multiple platforms (stored in buildx cache, not loaded into Docker)
    echo "Building for multiple platforms: $PLATFORMS"
    echo "Note: Image will be stored in buildx cache (not loaded into local Docker)"
    docker buildx build --platform "$PLATFORMS" $NO_CACHE -t "$IMAGE_NAME:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk" "$SCRIPT_DIR"
    
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "=== Build Complete ==="
        echo "Image: $IMAGE_NAME:$TAG"
        echo "Platforms: $PLATFORMS"
        echo "Image is stored in buildx cache and ready for push"
        echo ""
    fi
fi