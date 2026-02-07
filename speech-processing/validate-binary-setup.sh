#!/bin/bash
# Validation script for the Nuitka-based binarization setup
# This script performs pre-build validation checks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=================================="
echo "Binarization Setup Validation"
echo "=================================="
echo ""

# Track failures
FAILURES=0

# Check 1: Dockerfile exists
echo -n "Checking if Dockerfile.sherpa-binary exists... "
if [ -f "Dockerfile.sherpa-binary" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: Dockerfile.sherpa-binary not found"
    FAILURES=$((FAILURES + 1))
fi

# Check 2: Build script exists and is executable
echo -n "Checking if build-sherpa-binary.sh exists and is executable... "
if [ -f "build-sherpa-binary.sh" ] && [ -x "build-sherpa-binary.sh" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: build-sherpa-binary.sh not found or not executable"
    FAILURES=$((FAILURES + 1))
fi

# Check 3: Documentation exists
echo -n "Checking if BINARIZATION.md exists... "
if [ -f "BINARIZATION.md" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: BINARIZATION.md not found"
    FAILURES=$((FAILURES + 1))
fi

# Check 4: Required Python files exist
echo -n "Checking if required Python files exist... "
MISSING_FILES=()
for file in main.py stt_impl.py vad_stt_wrapper.py; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: Missing files: ${MISSING_FILES[*]}"
    FAILURES=$((FAILURES + 1))
fi

# Check 5: Requirements files exist
echo -n "Checking if requirements files exist... "
MISSING_REQS=()
for file in requirements-base.txt requirements-sherpa.txt; do
    if [ ! -f "$file" ]; then
        MISSING_REQS+=("$file")
    fi
done

if [ ${#MISSING_REQS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: Missing requirements files: ${MISSING_REQS[*]}"
    FAILURES=$((FAILURES + 1))
fi

# Check 6: Validate Dockerfile syntax (basic check)
echo -n "Checking Dockerfile syntax (basic)... "
if grep -q "^FROM.*AS builder" Dockerfile.sherpa-binary && \
   grep -q "^FROM.*AS compiler" Dockerfile.sherpa-binary && \
   grep -q "nuitka" Dockerfile.sherpa-binary; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: Dockerfile appears to be missing key sections"
    FAILURES=$((FAILURES + 1))
fi

# Check 7: Docker is available (optional)
echo -n "Checking if Docker is available... "
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    echo "  Docker version: $DOCKER_VERSION"
else
    echo -e "${YELLOW}⚠${NC}"
    echo "  WARNING: Docker not found. You'll need Docker to build the image."
fi

# Check 8: Sherpa models (optional but recommended)
echo -n "Checking if Sherpa models are downloaded... "
if [ -d "sherpa-onnx-streaming-models" ] && [ $(find sherpa-onnx-streaming-models -mindepth 1 ! -name 'README.md' | wc -l) -gt 0 ]; then
    echo -e "${GREEN}✓${NC}"
    MODEL_COUNT=$(find sherpa-onnx-streaming-models -type f ! -name 'README.md' | wc -l)
    echo "  Found $MODEL_COUNT model files"
else
    echo -e "${YELLOW}⚠${NC}"
    echo "  WARNING: Sherpa models not found or empty"
    echo "  Run './download-models.sh' to download them before building"
fi

# Check 9: GitHub token (optional)
echo -n "Checking if GITHUB_TOKEN is set... "
if [ -n "$GITHUB_TOKEN" ]; then
    echo -e "${GREEN}✓${NC}"
    echo "  Token length: ${#GITHUB_TOKEN} characters"
elif [ -f "$HOME/.github_token" ]; then
    echo -e "${YELLOW}⚠${NC}"
    echo "  WARNING: GITHUB_TOKEN not set, but found ~/.github_token"
    echo "  You can use: export GITHUB_TOKEN=\$(cat ~/.github_token)"
else
    echo -e "${YELLOW}⚠${NC}"
    echo "  WARNING: GITHUB_TOKEN not set"
    echo "  You'll need a GitHub token with 'repo' scope to build"
    echo "  Generate one at: https://github.com/settings/tokens"
fi

echo ""
echo "=================================="
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}All critical checks passed!${NC}"
    echo ""
    echo "You can now build the binary image with:"
    echo "  ./build-sherpa-binary.sh"
    echo ""
    exit 0
else
    echo -e "${RED}Validation failed with $FAILURES error(s)${NC}"
    echo ""
    echo "Please fix the errors above before building."
    echo ""
    exit 1
fi
