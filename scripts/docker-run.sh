#!/usr/bin/env bash
#
# docker-run.sh - Run Cat Breeds Classification Docker container
#
# Usage:
#   ./scripts/docker-run.sh [OPTIONS]
#
# Options:
#   --detach      Run in detached mode (background)
#   --port PORT   Host port (default: 8000)
#   --gpu ID      GPU device ID (default: 0)
#   --no-gpu      Disable GPU (CPU only)
#   --dry-run     Print commands without executing
#   --help        Show this help message

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
IMAGE_NAME="cat-breeds-api"
IMAGE_TAG="latest"
CONTAINER_NAME="cat-breeds-api"
HOST_PORT="8000"
GPU_ID="0"
DETACH=""
USE_GPU=true
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --detach)
            DETACH="-d"
            shift
            ;;
        --port)
            HOST_PORT="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --no-gpu)
            USE_GPU=false
            shift
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

# Check Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker not found."
    exit 1
fi

# Check image exists
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
    log_error "Image ${IMAGE_NAME}:${IMAGE_TAG} not found."
    log_info "Build image first: ./scripts/docker-build.sh"
    exit 1
fi

# Check GPU support
if [ "$USE_GPU" = true ]; then
    log_info "Checking GPU support..."

    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Install NVIDIA drivers."
        exit 1
    fi

    # Check NVIDIA Container Toolkit
    if ! docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
        log_error "NVIDIA Container Toolkit not configured."
        log_info "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi

    GPU_ARGS="--gpus device=${GPU_ID}"
    log_info "GPU ${GPU_ID} enabled"
else
    GPU_ARGS=""
    log_warn "Running in CPU-only mode (slow inference)"
fi

# Check model volume
MODEL_DIR="$(pwd)/outputs/checkpoints"
if [ ! -d "$MODEL_DIR" ]; then
    log_warn "Model directory not found: $MODEL_DIR"
    log_info "Ensure model checkpoint exists before starting container"
fi

# Stop existing container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_info "Stopping existing container..."
    run_cmd docker stop "$CONTAINER_NAME" || true
    run_cmd docker rm "$CONTAINER_NAME" || true
fi

# Run container
log_info "Starting container: ${CONTAINER_NAME}"
log_info "Port: ${HOST_PORT}"
log_info "Mode: $([ -n "$DETACH" ] && echo 'detached' || echo 'attached')"

run_cmd docker run $DETACH \
    --name "$CONTAINER_NAME" \
    $GPU_ARGS \
    -p "127.0.0.1:${HOST_PORT}:8000" \
    -v "${MODEL_DIR}:/app/outputs/checkpoints:ro" \
    -v "cat_breeds_logs:/app/logs" \
    -v "cat_breeds_cache:/home/appuser/.cache" \
    --env-file .env \
    --restart unless-stopped \
    "${IMAGE_NAME}:${IMAGE_TAG}"

if [ "$DRY_RUN" = false ] && [ -n "$DETACH" ]; then
    log_info "Container started in background"
    log_info "Logs: docker logs -f ${CONTAINER_NAME}"
    log_info "API: http://localhost:${HOST_PORT}/docs"
    log_info "Health: curl http://localhost:${HOST_PORT}/health/live"
fi
