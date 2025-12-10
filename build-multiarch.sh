#!/bin/bash
# Multi-architecture build script (ARM64 + AMD64)
# FR_VERS_JP v2.1

set -e

echo "========================================"
echo "FR_VERS_JP v2.1 - Multi-Arch Builder"
echo "ARM64 + AMD64"
echo "========================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not installed${NC}"
    exit 1
fi

echo -e "${CYAN}Docker: $(docker --version)${NC}"

# Setup buildx
if ! docker buildx version &> /dev/null; then
    echo -e "${YELLOW}Installing Docker Buildx...${NC}"
    docker buildx install
fi

# Create or use existing builder
BUILDER_NAME="multiarch-builder"
if ! docker buildx ls | grep -q "$BUILDER_NAME"; then
    echo -e "${CYAN}Creating multi-arch builder...${NC}"
    docker buildx create --name "$BUILDER_NAME" --platform linux/arm64,linux/amd64
else
    echo -e "${CYAN}Using existing builder: $BUILDER_NAME${NC}"
fi

docker buildx use "$BUILDER_NAME"
docker buildx inspect --bootstrap

# Build configurations
IMAGE_NAME="fr-vers-jp"
VERSION="2.1"
PLATFORMS="linux/arm64,linux/amd64"

echo ""
echo -e "${GREEN}Building for platforms: ${PLATFORMS}${NC}"
echo -e "${GREEN}Image: ${IMAGE_NAME}:${VERSION}${NC}"
echo ""

# Build for multiple architectures
echo -e "${CYAN}Building multi-architecture image...${NC}"
docker buildx build \
    --platform "${PLATFORMS}" \
    --tag "${IMAGE_NAME}:${VERSION}" \
    --tag "${IMAGE_NAME}:latest" \
    --load \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Multi-arch build successful!${NC}"
    echo ""
    echo -e "${CYAN}Available images:${NC}"
    docker images | grep "${IMAGE_NAME}" | head -5
    echo ""
    echo -e "${CYAN}To run:${NC}"
    echo -e "  docker run -it --rm --privileged -v /dev/video0:/dev/video0 ${IMAGE_NAME}:${VERSION}"
    echo ""
    echo -e "${CYAN}Or with docker-compose:${NC}"
    echo -e "  docker-compose up"
    echo ""
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
