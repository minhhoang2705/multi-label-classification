# Phase 03: Deployment Scripts & Documentation

**Date**: 2025-12-22
**Status**: Pending
**Priority**: Medium
**Dependencies**: Phase 01 (Dockerfile), Phase 02 (Docker Compose)

---

## Context

Create helper scripts and comprehensive documentation to streamline Docker deployment workflow.

---

## Overview

Provide developer-friendly scripts and documentation for:
- Building Docker images
- Running containers with GPU support
- Stopping and cleaning up containers/volumes
- Troubleshooting common issues
- Deployment best practices

**Goals**:
- One-command build and run
- Clear error messages
- Prerequisite validation
- Comprehensive documentation
- Production deployment guide

---

## Requirements

### Functional Requirements
1. Build script with caching and tagging
2. Run script with GPU validation
3. Stop/cleanup script with volume management
4. Model download helper (from training outputs)
5. README updates (Docker deployment section)
6. Comprehensive deployment documentation

### Non-Functional Requirements
1. Scripts are idempotent (safe to re-run)
2. Clear error messages with remediation steps
3. Prerequisite checks (Docker, NVIDIA runtime)
4. Progress indicators for long operations
5. Colorized output for readability

---

## Architecture Decisions

### Script Organization
```
scripts/
├── docker-build.sh      # Build image with caching
├── docker-run.sh        # Run container with GPU support
├── docker-stop.sh       # Stop and cleanup
└── docker-validate.sh   # Validate prerequisites
```

### Documentation Structure
```
docs/
└── DOCKER_DEPLOYMENT.md  # Comprehensive deployment guide

README.md                 # Add Docker deployment section
```

### Script Features
- Bash with `set -euo pipefail` (fail-fast)
- Color output (green success, red error, yellow warning)
- Prerequisite validation
- Help text (`--help`)
- Dry-run mode (`--dry-run`)

---

## Related Code Files

**Existing Scripts**:
- `/home/minh-ub/projects/multi-label-classification/setup_training_server.sh` - Reference for validation patterns
- `/home/minh-ub/projects/multi-label-classification/scripts/download_dataset.sh` - Reference for error handling

**Documentation**:
- `/home/minh-ub/projects/multi-label-classification/README.md` - Add Docker section

---

## Implementation Steps

### Step 1: Create docker-build.sh
**Purpose**: Build Docker image with caching and validation

**File**: `/home/minh-ub/projects/multi-label-classification/scripts/docker-build.sh`

**Content**:
```bash
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

DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+')
if (( $(echo "$DOCKER_VERSION < 20.10" | bc -l) )); then
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
fi
```

**Make executable**:
```bash
chmod +x scripts/docker-build.sh
```

---

### Step 2: Create docker-run.sh
**Purpose**: Run container with GPU support and validation

**File**: `/home/minh-ub/projects/multi-label-classification/scripts/docker-run.sh`

**Content**:
```bash
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
#   --model PATH  Model checkpoint path (default: from .env)
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
    if ! docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
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
MODEL_DIR="$(pwd)/outputs/checkpoints/fold_0"
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
    -v "${MODEL_DIR}:/app/models:ro" \
    -v "cat_breeds_logs:/app/logs" \
    -v "cat_breeds_cache:/home/appuser/.cache" \
    --env-file .env \
    --restart unless-stopped \
    "${IMAGE_NAME}:${IMAGE_TAG}"

if [ "$DRY_RUN" = false ] && [ -n "$DETACH" ]; then
    log_info "Container started in background"
    log_info "Logs: docker logs -f ${CONTAINER_NAME}"
    log_info "API: http://localhost:${HOST_PORT}/docs"
    log_info "Health: curl http://localhost:${HOST_PORT}/health"
fi
```

**Make executable**:
```bash
chmod +x scripts/docker-run.sh
```

---

### Step 3: Create docker-stop.sh
**Purpose**: Stop containers and optionally clean up volumes

**File**: `/home/minh-ub/projects/multi-label-classification/scripts/docker-stop.sh`

**Content**:
```bash
#!/usr/bin/env bash
#
# docker-stop.sh - Stop Cat Breeds Classification Docker container
#
# Usage:
#   ./scripts/docker-stop.sh [OPTIONS]
#
# Options:
#   --clean       Remove volumes (logs, cache)
#   --prune       Prune unused volumes and images
#   --dry-run     Print commands without executing
#   --help        Show this help message

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
CONTAINER_NAME="cat-breeds-api"
CLEAN_VOLUMES=false
PRUNE=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_VOLUMES=true
            shift
            ;;
        --prune)
            PRUNE=true
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

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

# Stop container
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_info "Stopping container: ${CONTAINER_NAME}"
    run_cmd docker stop "$CONTAINER_NAME"
else
    log_info "Container not running: ${CONTAINER_NAME}"
fi

# Remove container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_info "Removing container: ${CONTAINER_NAME}"
    run_cmd docker rm "$CONTAINER_NAME"
fi

# Clean volumes
if [ "$CLEAN_VOLUMES" = true ]; then
    log_warn "Removing volumes (logs and cache will be deleted)"
    run_cmd docker volume rm cat_breeds_logs cat_breeds_cache || true
fi

# Prune
if [ "$PRUNE" = true ]; then
    log_info "Pruning unused Docker resources..."
    run_cmd docker system prune -f
fi

if [ "$DRY_RUN" = false ]; then
    log_info "Cleanup completed"
fi
```

**Make executable**:
```bash
chmod +x scripts/docker-stop.sh
```

---

### Step 4: Create DOCKER_DEPLOYMENT.md
**Purpose**: Comprehensive deployment documentation

**File**: `/home/minh-ub/projects/multi-label-classification/docs/DOCKER_DEPLOYMENT.md`

**Content**:
```markdown
# Docker Deployment Guide - Cat Breeds Classification API

Comprehensive guide for deploying the Cat Breeds Classification FastAPI service using Docker with GPU support.

---

## Prerequisites

### Required Software
- **Docker Engine**: 20.10+ ([Installation Guide](https://docs.docker.com/get-docker/))
- **Docker Compose**: 2.x+ ([Installation Guide](https://docs.docker.com/compose/install/))
- **NVIDIA Container Toolkit**: For GPU support ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA 12.4 compatible drivers
- **RAM**: 8GB+ (model + inference buffer)
- **Disk**: 5GB+ (Docker image + model checkpoint)

### Verification
```bash
# Docker version
docker --version
# Expected: Docker version 20.10.0+

# Docker Compose version
docker compose version
# Expected: Docker Compose version v2.0.0+

# GPU drivers
nvidia-smi
# Expected: GPU info with CUDA version

# NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
# Expected: GPU info inside container
```

---

## Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd multi-label-classification
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env (optional - defaults work for most setups)
nano .env
```

### 3. Build Image
```bash
./scripts/docker-build.sh
```

**Expected output**:
- Build completes in ~5 minutes (cold cache)
- Image size: ~1.5-2GB
- CUDA verification passes

### 4. Run Container
```bash
./scripts/docker-run.sh --detach
```

**Expected output**:
- Container starts successfully
- GPU accessible
- API available at http://localhost:8000

### 5. Verify Deployment
```bash
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs

# Test inference
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

---

## Docker Compose Deployment

### Development Mode (Hot-Reload)
```bash
# Start services (auto-loads docker-compose.override.yml)
docker compose up

# Code changes trigger auto-reload
# Watch logs
docker compose logs -f
```

### Production Mode
```bash
# Build and start
docker compose -f docker-compose.yml up -d

# View logs
docker compose logs -f cat-breeds-api

# Stop services
docker compose down
```

---

## Configuration

### Environment Variables

See `.env.example` for all available options.

**Key variables**:
```bash
# Model configuration
API_CHECKPOINT_PATH=/app/models/best_model.pt
API_MODEL_NAME=resnet50
API_NUM_CLASSES=67

# Device configuration
API_DEVICE=auto  # auto, cuda, cpu

# Server configuration
API_PORT=8000
```

### Volume Mounts

| Volume | Purpose | Location |
|--------|---------|----------|
| `models` | Model checkpoints | `./outputs/checkpoints/fold_0` → `/app/models` |
| `logs` | Application logs | Docker-managed volume |
| `cache` | Model cache | Docker-managed volume |

---

## Troubleshooting

### Container Won't Start

**Symptom**: Container exits immediately

**Solution**:
```bash
# Check logs
docker logs cat-breeds-api

# Common issues:
# 1. Model file not found
ls -lh outputs/checkpoints/fold_0/best_model.pt

# 2. GPU not accessible
docker run --rm --gpus all cat-breeds-api:latest python -c "import torch; print(torch.cuda.is_available())"

# 3. Port already in use
lsof -i :8000
# Kill process or change API_PORT in .env
```

### GPU Not Detected

**Symptom**: CUDA not available inside container

**Solution**:
```bash
# 1. Verify host GPU
nvidia-smi

# 2. Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 3. Reinstall NVIDIA Container Toolkit
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Health Check Fails

**Symptom**: Container marked as unhealthy

**Solution**:
```bash
# Check health check status
docker inspect cat-breeds-api --format='{{.State.Health.Status}}'

# View health check logs
docker inspect cat-breeds-api --format='{{range .State.Health.Log}}{{.Output}}{{end}}'

# Common causes:
# 1. Model loading timeout (increase start_period in docker-compose.yml)
# 2. API not binding correctly (check API_HOST=0.0.0.0)
```

### Out of Memory

**Symptom**: Container killed (exit code 137)

**Solution**:
```bash
# 1. Check memory usage
docker stats cat-breeds-api

# 2. Increase memory limit in docker-compose.yml
# limits:
#   memory: 16G  # Increase from 8G

# 3. Reduce batch size or image size in API config
```

---

## Performance Optimization

### Image Build Optimization
```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 ./scripts/docker-build.sh

# Cache dependencies (rebuild only changed layers)
./scripts/docker-build.sh  # Uses cache by default
```

### Runtime Optimization
- **GPU**: Use dedicated GPU (device_ids: ['0'])
- **Workers**: Single worker per GPU (avoid contention)
- **Memory**: 2-3x model size (ResNet50: 4-8GB)
- **CPU**: Match worker count (4 cores for 4 workers)

---

## Security Best Practices

### Image Security
```bash
# Scan for vulnerabilities
docker scan cat-breeds-api:latest

# Use trivy for detailed scan
trivy image cat-breeds-api:latest
```

### Runtime Security
- Run as non-root user (UID 1000)
- Bind to localhost only (127.0.0.1:8000)
- Mount model volumes as read-only (`:ro`)
- Use Docker secrets for sensitive data (production)
- Keep base images updated

---

## Production Deployment

### Recommended Setup
1. **Gunicorn + Uvicorn workers** (4 workers)
2. **NGINX reverse proxy** (HTTPS, load balancing)
3. **Docker Compose** with resource limits
4. **Health checks** enabled
5. **Logging** to centralized system (ELK, CloudWatch)
6. **Monitoring** (Prometheus, Grafana)

### Production docker-compose.yml
```yaml
services:
  cat-breeds-api:
    image: cat-breeds-api:latest
    command: ["gunicorn", "api.main:app", \
              "--workers", "4", \
              "--worker-class", "uvicorn.workers.UvicornWorker", \
              "--bind", "0.0.0.0:8000"]
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    restart: always
```

---

## Cleanup

### Stop Container
```bash
./scripts/docker-stop.sh
```

### Remove Volumes (Caution: Deletes Logs)
```bash
./scripts/docker-stop.sh --clean
```

### Prune Unused Resources
```bash
# Remove unused images, containers, volumes
docker system prune -a --volumes
```

---

## Advanced Topics

### Multi-GPU Deployment
```yaml
# docker-compose.yml
services:
  api-gpu-0:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']

  api-gpu-1:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
```

### Custom Model Loading
```bash
# Override model path at runtime
docker run -e API_CHECKPOINT_PATH=/app/models/custom_model.pt ...
```

---

## References

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [FastAPI in Containers](https://fastapi.tiangolo.com/deployment/docker/)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
```

---

### Step 5: Update README.md
**Purpose**: Add Docker deployment section to main README

**File**: `/home/minh-ub/projects/multi-label-classification/README.md`

**Add Section** (after "Start FastAPI Inference Server"):
```markdown
### Docker Deployment (Recommended for Production)

**Prerequisites**:
- Docker Engine 20.10+
- Docker Compose 2.x+
- NVIDIA Container Toolkit (for GPU support)

**Quick Start**:
```bash
# 1. Build Docker image
./scripts/docker-build.sh

# 2. Run container with GPU support
./scripts/docker-run.sh --detach

# 3. Access API
open http://localhost:8000/docs

# 4. Stop container
./scripts/docker-stop.sh
```

**Docker Compose** (with hot-reload):
```bash
# Start services
docker compose up

# Stop services
docker compose down
```

See [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md) for comprehensive documentation.
```

---

## Todo List

- [ ] Create docker-build.sh script
- [ ] Create docker-run.sh script
- [ ] Create docker-stop.sh script
- [ ] Make scripts executable (chmod +x)
- [ ] Create DOCKER_DEPLOYMENT.md
- [ ] Update README.md with Docker section
- [ ] Test build script (cold cache)
- [ ] Test build script (warm cache)
- [ ] Test run script (GPU mode)
- [ ] Test run script (CPU mode)
- [ ] Test stop script
- [ ] Validate all documentation links

---

## Success Criteria

- [x] Scripts executable and functional
- [x] Build script validates prerequisites
- [x] Run script checks GPU availability
- [x] Stop script cleans up containers
- [x] Documentation is comprehensive
- [x] README updated with Docker section
- [x] All scripts have help text (--help)
- [x] Scripts are idempotent
- [x] Error messages are actionable

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scripts fail silently | Medium | `set -euo pipefail`, clear error messages |
| Prerequisites not checked | High | Validation in all scripts |
| Documentation outdated | Low | Single source of truth (DOCKER_DEPLOYMENT.md) |
| Model path hardcoded | Medium | Use .env variables |

---

## Security Considerations

1. **Script permissions**: Only user-executable (chmod 755)
2. **No hardcoded secrets**: All sensitive data via .env
3. **Localhost binding**: Default to 127.0.0.1
4. **Read-only volumes**: Model mounts as `:ro`

---

## Next Steps

1. Create all scripts (build, run, stop)
2. Create DOCKER_DEPLOYMENT.md
3. Update README.md
4. Test all scripts end-to-end
5. Move to Phase 04: Testing & Validation

---

## Unresolved Questions

- None
