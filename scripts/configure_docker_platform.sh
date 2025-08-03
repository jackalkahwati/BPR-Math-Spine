#!/bin/bash
# BPR-Math-Spine Docker Platform Configuration Script
#
# Automatically detects system architecture and configures docker-compose.yml
# for optimal compatibility with both Apple Silicon and Intel systems.
#
# Usage: ./scripts/configure_docker_platform.sh

set -e

echo "🔍 BPR Docker Platform Configuration"
echo "====================================="

# Detect architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Determine Docker platform
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    DOCKER_PLATFORM="linux/arm64"
    SYSTEM_TYPE="Apple Silicon (M1/M2/M3)"
elif [ "$ARCH" = "x86_64" ]; then
    DOCKER_PLATFORM="linux/amd64"
    SYSTEM_TYPE="Intel Mac/PC"
else
    echo "⚠️  Unknown architecture: $ARCH"
    echo "   Defaulting to linux/amd64"
    DOCKER_PLATFORM="linux/amd64"
    SYSTEM_TYPE="Unknown (defaulting to Intel)"
fi

echo "System type: $SYSTEM_TYPE"
echo "Docker platform: $DOCKER_PLATFORM"
echo

# Update docker-compose.yml
echo "📝 Updating docker-compose.yml..."

# Create backup
cp docker-compose.yml docker-compose.yml.backup
echo "   Created backup: docker-compose.yml.backup"

# Replace platform specifications
if [ "$DOCKER_PLATFORM" = "linux/arm64" ]; then
    # Set to ARM64
    sed -i.tmp 's/platform: linux\/amd64/platform: linux\/arm64/g' docker-compose.yml
    sed -i.tmp 's/platform: linux\/arm64  # Use linux\/amd64 for Intel Macs/platform: linux\/arm64  # Configured for Apple Silicon/g' docker-compose.yml
else
    # Set to AMD64
    sed -i.tmp 's/platform: linux\/arm64/platform: linux\/amd64/g' docker-compose.yml
    sed -i.tmp 's/platform: linux\/arm64  # Use linux\/amd64 for Intel Macs/platform: linux\/amd64  # Configured for Intel/g' docker-compose.yml
fi

# Clean up temporary files
rm -f docker-compose.yml.tmp

echo "✅ docker-compose.yml configured for $DOCKER_PLATFORM"
echo

# Show next steps
echo "🚀 Next Steps:"
echo "   1. Build and run: docker-compose up -d"
echo "   2. Open Jupyter:  http://localhost:8888"
echo "   3. Token:         bpr-token-2025"
echo "   4. Stop:          docker-compose down"
echo

echo "📋 Available Services:"
echo "   • bpr-math-spine  - Main Jupyter environment"
echo "   • bpr-test        - Run test suite (docker-compose --profile testing up)"
echo "   • bpr-benchmark   - Run benchmarks (docker-compose --profile benchmark up)"
echo

echo "🔧 Manual Configuration:"
echo "   To manually switch platforms, edit docker-compose.yml and change"
echo "   all 'platform: $DOCKER_PLATFORM' entries to the desired platform."