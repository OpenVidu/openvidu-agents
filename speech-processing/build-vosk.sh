#!/bin/bash
# Build script for openvidu/agent-speech-processing-vosk
#
# Usage:
#   ./build-vosk.sh [--no-cache] [--tag TAG] [--local-only] [--push]
# Flags:
#   --no-cache: Do not use Docker build cache
#   --tag TAG: Specify a custom Docker tag (default: main)
#   --local-only: Build only for local platform (no multi-arch, no buildx)
#   --push: Push the multi-arch image to registry (only valid without --local-only)

set -e

# Help function
show_help() {
    cat << EOF
Build script for openvidu/agent-speech-processing-vosk

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --help, -h
        Display this help message and exit

    --no-cache
        Do not use Docker build cache. Forces a complete rebuild of all layers.

    --tag TAG
        Specify a custom Docker tag (default: main)
        Example: --tag v2.0.0

    --local-only
        Build only for the local platform architecture (no multi-arch build).
        Uses 'docker buildx build --load' to load the image into local Docker.
        Faster for local development and testing.

    --push
        Push the multi-arch image to the Docker registry after building.
        Cannot be used with --local-only.
        Requires appropriate Docker registry authentication.

EXAMPLES:
    # Build for local testing
    $0 --local-only

    # Build and push multi-arch image with custom tag
    $0 --tag v2.0.0 --push

    # Force rebuild without cache
    $0 --no-cache --local-only

    # Build multi-arch image without pushing (stored in buildx cache)
    $0

EOF
    exit 0
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="openvidu/agent-speech-processing-vosk"
PLATFORMS="linux/amd64,linux/arm64"

# Flags
NO_CACHE=""
TAG="${TAG:-main}"
LOCAL_ONLY=false
PUSH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            ;;
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
        --push)
            PUSH=true
            shift
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate flags
if [[ "$LOCAL_ONLY" == "true" && "$PUSH" == "true" ]]; then
    echo "[ERROR] Cannot use --push with --local-only"
    exit 1
fi

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
    if [[ "$PUSH" == "true" ]]; then
        echo "Will push to registry after build"
        docker buildx build --platform "$PLATFORMS" $NO_CACHE -t "$IMAGE_NAME:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk" --push "$SCRIPT_DIR"
    else
        echo "Note: Image will be stored in buildx cache (not loaded into local Docker)"
        docker buildx build --platform "$PLATFORMS" $NO_CACHE -t "$IMAGE_NAME:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk" "$SCRIPT_DIR"
    fi
    
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "=== Build Complete ==="
        echo "Image: $IMAGE_NAME:$TAG"
        echo "Platforms: $PLATFORMS"
        if [[ "$PUSH" == "true" ]]; then
            echo "Image has been pushed to registry"
        else
            echo "Image is stored in buildx cache (run with --push flag to upload to registry)"
        fi
        echo ""
    fi
fi