#!/bin/bash
# Build script for the Nuitka-binarized Sherpa agent Docker image
# This script simplifies building the Docker image with proper GitHub authentication

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Nuitka Binary Build for Sherpa Agent${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check if models are downloaded
if [ ! -d "sherpa-onnx-streaming-models" ] || [ $(find sherpa-onnx-streaming-models -mindepth 1 ! -name 'README.md' | wc -l) -eq 0 ]; then
    echo -e "${RED}ERROR: Sherpa models not found!${NC}"
    echo ""
    echo "Please download the models first by running:"
    echo "  ./download-models.sh"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Sherpa models found${NC}"
echo ""

# Check for GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    echo -e "${YELLOW}WARNING: GITHUB_TOKEN environment variable not set${NC}"
    echo ""
    echo "The build requires a GitHub Personal Access Token with 'repo' scope"
    echo "to access private repositories."
    echo ""
    echo "You can:"
    echo "  1. Set GITHUB_TOKEN environment variable:"
    echo "     export GITHUB_TOKEN=ghp_your_token_here"
    echo ""
    echo "  2. Or create a token file at ~/.github_token"
    echo ""
    read -p "Continue without token? (build may fail) [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    TOKEN_ARG=""
else
    echo -e "${GREEN}✓ GITHUB_TOKEN found${NC}"
    TOKEN_ARG="--secret id=github_token,env=GITHUB_TOKEN"
fi

# Image name and tag
IMAGE_NAME="${IMAGE_NAME:-openvidu/agent-speech-processing-sherpa-binary}"
IMAGE_TAG="${IMAGE_TAG:-main}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo -e "${GREEN}Building Docker image: ${FULL_IMAGE_NAME}${NC}"
echo ""
echo "This will:"
echo "  1. Install all Python dependencies"
echo "  2. Compile the agent code with Nuitka (this may take 5-15 minutes)"
echo "  3. Create a minimal runtime image with only the binary"
echo ""
read -p "Continue? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    exit 0
fi

echo ""
echo -e "${GREEN}Starting build...${NC}"
echo ""

# Build the Docker image
if [ -n "$TOKEN_ARG" ]; then
    docker build \
        $TOKEN_ARG \
        -f Dockerfile.sherpa-binary \
        -t "$FULL_IMAGE_NAME" \
        .
else
    docker build \
        -f Dockerfile.sherpa-binary \
        -t "$FULL_IMAGE_NAME" \
        .
fi

BUILD_STATUS=$?

echo ""
if [ $BUILD_STATUS -eq 0 ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo "Image: $FULL_IMAGE_NAME"
    echo ""
    echo "To verify the binary was created and source code is not present:"
    echo "  docker run --rm -it --entrypoint /bin/bash $FULL_IMAGE_NAME"
    echo "  # Then inside the container:"
    echo "  ls -la /app/agent_dist/main"
    echo "  ls /app/*.py  # Should not exist!"
    echo ""
    echo "To run the agent:"
    echo "  docker run --rm $FULL_IMAGE_NAME"
    echo ""
else
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}Build failed!${NC}"
    echo -e "${RED}======================================${NC}"
    echo ""
    echo "Please check the error messages above."
    echo ""
    exit 1
fi
