#!/bin/bash
# Build script for ARM64 architecture
# FR_VERS_JP v2.1

set -e

echo "================================"
echo "FR_VERS_JP v2.1 - ARM64 Builder"
echo "================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${CYAN}Docker found: $(docker --version)${NC}"
echo ""

# Check if buildx is available
if ! docker buildx version &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker Buildx not found. Installing...${NC}"
    docker buildx install
fi

# Create buildx builder if it doesn't exist
if ! docker buildx ls | grep -q "arm64-builder"; then
    echo -e "${CYAN}Creating ARM64 builder...${NC}"
    docker buildx create --name arm64-builder --platform linux/arm64,linux/amd64
fi

echo -e "${CYAN}Using builder: arm64-builder${NC}"
docker buildx use arm64-builder

# Bootstrap builder
echo -e "${CYAN}Bootstrapping builder...${NC}"
docker buildx inspect --bootstrap

# Build options
BUILD_PLATFORM=${1:-"linux/arm64"}
IMAGE_TAG="fr-vers-jp:2.1-arm64"

echo ""
echo -e "${GREEN}Building for platform: ${BUILD_PLATFORM}${NC}"
echo -e "${GREEN}Image tag: ${IMAGE_TAG}${NC}"
echo ""

# Build the image
echo -e "${CYAN}Building Docker image...${NC}"
docker buildx build \
    --platform "${BUILD_PLATFORM}" \
    --tag "${IMAGE_TAG}" \
    --load \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    echo -e "${CYAN}To run the container:${NC}"
    echo -e "  docker run -it --rm --privileged -v /dev/video0:/dev/video0 ${IMAGE_TAG}"
    echo ""
    echo -e "${CYAN}Or use docker-compose:${NC}"
    echo -e "  docker-compose up"
    echo ""
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
