#!/bin/bash
# Validation script for ARM64 build
# FR_VERS_JP v2.1

set -e

echo "=================================="
echo "FR_VERS_JP v2.1 - Build Validator"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

# Function to check file
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
    else
        echo -e "${RED}✗${NC} $1 missing"
        ((ERRORS++))
    fi
}

# Function to check executable
check_executable() {
    if [ -x "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 is executable"
    else
        echo -e "${YELLOW}⚠${NC} $1 not executable"
        ((WARNINGS++))
    fi
}

echo -e "${CYAN}Checking build files...${NC}"
echo ""

# Check Docker files
check_file "Dockerfile"
check_file "docker-compose.yml"
check_file ".dockerignore"

# Check build scripts
check_file "build-arm64.sh"
check_executable "build-arm64.sh"
check_file "build-multiarch.sh"
check_executable "build-multiarch.sh"

# Check documentation
check_file "BUILD_ARM64.md"
check_file "README.md"

# Check GitHub workflow
check_file ".github/workflows/build-arm64.yml"

# Check core files
echo ""
echo -e "${CYAN}Checking core files...${NC}"
echo ""
check_file "requirements.txt"
check_file "launcher.py"
check_file "fr_core/config.py"

# Validate Dockerfile syntax
echo ""
echo -e "${CYAN}Validating Dockerfile syntax...${NC}"
if docker version &> /dev/null; then
    if grep -qE "FROM python:3\.[0-9]+-slim" Dockerfile; then
        echo -e "${GREEN}✓${NC} Dockerfile base image OK"
    else
        echo -e "${RED}✗${NC} Dockerfile base image not found"
        ((ERRORS++))
    fi
    
    if grep -q "ENV PYTHONPATH=/app" Dockerfile; then
        echo -e "${GREEN}✓${NC} PYTHONPATH configured"
    else
        echo -e "${YELLOW}⚠${NC} PYTHONPATH not configured"
        ((WARNINGS++))
    fi
else
    echo -e "${YELLOW}⚠${NC} Docker not available for validation"
    ((WARNINGS++))
fi

# Validate docker-compose.yml
echo ""
echo -e "${CYAN}Validating docker-compose.yml...${NC}"
if grep -q "linux/arm64" docker-compose.yml; then
    echo -e "${GREEN}✓${NC} ARM64 platform configured"
else
    echo -e "${RED}✗${NC} ARM64 platform not configured"
    ((ERRORS++))
fi

if grep -q "/dev/video0" docker-compose.yml; then
    echo -e "${GREEN}✓${NC} Camera device configured"
else
    echo -e "${YELLOW}⚠${NC} Camera device not configured"
    ((WARNINGS++))
fi

# Check requirements.txt
echo ""
echo -e "${CYAN}Validating requirements.txt...${NC}"
required_packages=("numpy" "opencv-python" "mediapipe" "scipy" "scikit-learn")
for package in "${required_packages[@]}"; do
    if grep -qi "$package" requirements.txt; then
        echo -e "${GREEN}✓${NC} $package found"
    else
        echo -e "${RED}✗${NC} $package missing"
        ((ERRORS++))
    fi
done

# Summary
echo ""
echo "=================================="
echo -e "${CYAN}Validation Summary${NC}"
echo "=================================="
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warning(s)${NC}"
    fi
    exit 0
else
    echo -e "${RED}✗ $ERRORS error(s) found${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warning(s)${NC}"
    fi
    exit 1
fi
