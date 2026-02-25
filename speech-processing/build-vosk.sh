#!/bin/bash
# Build script for openvidu/agent-speech-processing-vosk
# Builds two images:
#   1. openvidu/agent-speech-processing-vosk-base (everything except language models)
#   2. openvidu/agent-speech-processing-vosk (base + default language models)
#
# Usage:
#   ./build-vosk.sh [--no-cache] [--local-only] [--push] [--tag TAG] [--parent-base-image IMAGE] 
# Flags:
#   --no-cache: Do not use Docker build cache
#   --local-only: Build only for local platform (no multi-arch, no buildx)
#   --push: Push the multi-arch image to registry (only valid without --local-only)
#   --tag TAG: Specify a custom Docker tag (default: main)
#   --parent-base-image IMAGE: Override the parent base image (default: openvidu/agent-speech-processing-base:TAG)

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

    --local-only
        Build only for the local platform architecture (no multi-arch build).
        Uses 'docker buildx build --load' to load the image into local Docker.
        Faster for local development and testing.

    --push
        Push the multi-arch image to the Docker registry after building.
        Cannot be used with --local-only.
        Requires appropriate Docker registry authentication.

    --tag TAG
        Specify a custom Docker tag (default: main)
        Example: --tag 3.3.0

    --parent-base-image IMAGE
        Override the parent base image used by Dockerfile.vosk-base.
        Defaults to openvidu/agent-speech-processing-base:<TAG>.
        Example: --parent-base-image docker.io/openvidu/agent-speech-processing-base:3.3.0

EXAMPLES:
    # Build for local testing
    $0 --local-only

    # Build and push multi-arch image with custom tag
    $0 --tag 3.3.0 --push

    # Force rebuild without cache
    $0 --no-cache --local-only

    # Build multi-arch image without pushing (stored in buildx cache)
    $0

EOF
    exit 0
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_BASE_IMAGE_NAME="openvidu/agent-speech-processing-base"
IMAGE_NAME_BASE="openvidu/agent-speech-processing-vosk-base"
IMAGE_NAME="openvidu/agent-speech-processing-vosk"
PLATFORMS="linux/amd64,linux/arm64"

# Flags
NO_CACHE=""
TAG="${TAG:-main}"
LOCAL_ONLY=false
PUSH=false
PARENT_BASE_IMAGE_OVERRIDE=""

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
        --local-only)
            LOCAL_ONLY=true
            shift
            ;;
        --push)
            PUSH=true
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
        --parent-base-image)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "[ERROR] --parent-base-image requires a value"
                exit 1
            fi
            PARENT_BASE_IMAGE_OVERRIDE="$2"
            shift 2
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

PARENT_BASE_IMAGE="${PARENT_BASE_IMAGE_OVERRIDE:-$PARENT_BASE_IMAGE_NAME:$TAG}"

echo ""
echo "Build configuration:"
echo "  Parent base image: $PARENT_BASE_IMAGE"
echo "  Base image: $IMAGE_NAME_BASE:$TAG"
echo "  Final image: $IMAGE_NAME:$TAG"
echo "  Local only: $LOCAL_ONLY"
if [[ "$LOCAL_ONLY" == "true" ]]; then
    echo "  Platform: $(uname -m)"
else
    echo "  Platforms: $PLATFORMS"
fi

# Build the Docker images
if [[ "$LOCAL_ONLY" == "true" ]]; then
    # Build for local platform only and load into Docker
    # Use --builder default to ensure local Docker daemon driver (can access locally-loaded images)
    echo ""
    echo "=== Building base image (without models) ==="
    docker buildx build --builder default --load $NO_CACHE --build-arg BASE_IMAGE="$PARENT_BASE_IMAGE" -t "$IMAGE_NAME_BASE:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk-base" "$SCRIPT_DIR"

    echo ""
    echo "=== Building final image (with default models) ==="
    docker buildx build --builder default --load $NO_CACHE --build-arg BASE_IMAGE="$IMAGE_NAME_BASE:$TAG" -t "$IMAGE_NAME:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk" "$SCRIPT_DIR"
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

        echo ""
        echo "=== Building and pushing base image (without models) ==="
        docker buildx build --platform "$PLATFORMS" $NO_CACHE --build-arg BASE_IMAGE="$PARENT_BASE_IMAGE" -t "$IMAGE_NAME_BASE:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk-base" --push "$SCRIPT_DIR"

        echo ""
        echo "=== Building and pushing final image (with default models) ==="
        docker buildx build --platform "$PLATFORMS" $NO_CACHE --build-arg BASE_IMAGE="$IMAGE_NAME_BASE:$TAG" -t "$IMAGE_NAME:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk" --push "$SCRIPT_DIR"
    else
        echo "Note: Images will be stored in buildx cache (not loaded into local Docker)"

        echo ""
        echo "=== Building base image (without models) ==="
        docker buildx build --platform "$PLATFORMS" $NO_CACHE --build-arg BASE_IMAGE="$PARENT_BASE_IMAGE" -t "$IMAGE_NAME_BASE:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk-base" "$SCRIPT_DIR"

        echo ""
        echo "=== Building final image (with default models) ==="
        docker buildx build --platform "$PLATFORMS" $NO_CACHE --build-arg BASE_IMAGE="$IMAGE_NAME_BASE:$TAG" -t "$IMAGE_NAME:$TAG" -f "$SCRIPT_DIR/Dockerfile.vosk" "$SCRIPT_DIR"
    fi
    
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "=== Build Complete ==="
        echo "Base image: $IMAGE_NAME_BASE:$TAG"
        echo "Final image: $IMAGE_NAME:$TAG"
        echo "Platforms: $PLATFORMS"
        if [[ "$PUSH" == "true" ]]; then
            echo "Images have been pushed to registry"
        else
            echo "Images are stored in buildx cache (run with --push flag to upload to registry)"
        fi
        echo ""
    fi
fi