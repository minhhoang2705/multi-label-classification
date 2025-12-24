#!/usr/bin/env bash
#
# docker-build.sh - Build Cat Breeds Classification Docker image
#
# Usage:
#   ./scripts/docker-build.sh [OPTIONS]
#
# Options:
#   --no-cache    Build without cache
#   --tag TAG     Custom image tag (default: cat-breeds-api:latest)
#   --dry-run     Print commands without executing
#   --help        Show this help message

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="cat-breeds-api"
IMAGE_TAG="latest"
NO_CACHE=""
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            grep '^#' "$0" | cut -c 3-
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

# Validate prerequisites
log_info "Validating prerequisites..."

if ! command -v docker &> /dev/null; then
    log_error "Docker not found. Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+' || echo "0.0")
if (( $(echo "$DOCKER_VERSION < 20.10" | bc -l 2>/dev/null || echo 0) )); then
    log_warn "Docker version $DOCKER_VERSION detected. Recommend 20.10+"
fi

# Check Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    log_error "Dockerfile not found. Run from project root."
    exit 1
fi

# Check requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    log_error "requirements.txt not found."
    exit 1
fi

# Build image
log_info "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
log_info "Options: ${NO_CACHE:-with cache}"

BUILD_START=$(date +%s)

run_cmd docker build $NO_CACHE \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile \
    .

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

if [ "$DRY_RUN" = false ]; then
    log_info "Build completed in ${BUILD_TIME}s"

    # Show image info
    IMAGE_SIZE=$(docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "{{.Size}}")
    log_info "Image size: ${IMAGE_SIZE}"

    # Verify CUDA availability (build-time check in Dockerfile)
    log_info "Image built successfully!"
    log_info "Run container: ./scripts/docker-run.sh"
    log_info "Or use docker-compose: docker compose up"
fi
