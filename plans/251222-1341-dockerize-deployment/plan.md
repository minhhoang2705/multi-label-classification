# Dockerize Cat Breeds Classification API - Implementation Plan

**Date**: 2025-12-22
**Plan Directory**: `/home/minh-ub/projects/multi-label-classification/plans/251222-1341-dockerize-deployment/`
**Status**: Planning
**Priority**: High

---

## Overview

Comprehensive plan to dockerize the FastAPI-based cat breeds classification service with GPU support for local deployment.

**Objective**: Production-ready Docker containerization with NVIDIA GPU support, optimized multi-stage builds, and local deployment via Docker Compose.

---

## Project Context

**Service Components**:
- FastAPI application (`api/main.py`)
- PyTorch ResNet50/EfficientNet models (TIMM)
- 67 cat breeds multi-label classification
- Model checkpoint: `outputs/checkpoints/fold_0/best_model.pt`
- Python 3.12, CUDA 12.4, cuDNN 9

**Key Requirements**:
- Multi-stage Dockerfile (builder + runtime)
- GPU support via NVIDIA Container Toolkit
- Production-grade security (non-root user, health checks)
- Volume mounts for model artifacts
- Docker Compose orchestration with GPU reservation
- Resource limits and restart policies
- Development and production configurations

---

## Research Foundation

Implementation based on research reports:
- [docker-ml-best-practices.md](../reports/docker-ml-best-practices.md) - Multi-stage builds, GPU patterns, optimization
- [docker-compose-ml-patterns.md](../reports/docker-compose-ml-patterns.md) - Orchestration, volumes, health checks
- [pytorch-serving-patterns.md](../reports/pytorch-serving-patterns.md) - Model loading, inference optimization

---

## Implementation Phases

### Phase 01: Multi-Stage Dockerfile with GPU Support
**File**: `phase-01-dockerfile.md`
**Status**: Pending
**Priority**: Critical

Create optimized multi-stage Dockerfile with:
- Builder stage (pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel)
- Runtime stage (pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime)
- Layer caching optimization (requirements.txt)
- Non-root user (appuser:1000)
- Health check endpoint integration
- Volume mount points for models
- Environment variable configuration

**Key Deliverables**:
- `Dockerfile` (multi-stage with GPU support)
- `.dockerignore` (exclude unnecessary files)

---

### Phase 02: Docker Compose Configuration
**File**: `phase-02-docker-compose.md`
**Status**: Pending
**Priority**: Critical

Implement Docker Compose setup with:
- GPU device reservation (NVIDIA runtime)
- Service definition with health checks
- Named volumes (models, logs, cache)
- Environment variables via .env
- Resource limits (CPU, memory, GPU)
- Port mapping (localhost:8000)
- Restart policy (unless-stopped)
- Network configuration (bridge)

**Key Deliverables**:
- `docker-compose.yml` (base configuration)
- `docker-compose.override.yml` (dev overrides)
- `.env.example` (environment template)

---

### Phase 03: Deployment Scripts & Documentation
**File**: `phase-03-deployment-scripts.md`
**Status**: Pending
**Priority**: Medium

Create helper scripts and documentation:
- Build script (`scripts/docker-build.sh`)
- Run script (`scripts/docker-run.sh`)
- Stop/cleanup script (`scripts/docker-stop.sh`)
- Model download helper
- README updates (Docker deployment section)
- Troubleshooting guide

**Key Deliverables**:
- Deployment scripts (build, run, stop)
- README.md updates
- DOCKER_DEPLOYMENT.md (comprehensive guide)

---

### Phase 04: Testing & Validation
**File**: `phase-04-testing-validation.md`
**Status**: Pending
**Priority**: High

Validate deployment across scenarios:
- Image build verification (size, layers)
- GPU access validation inside container
- Health check endpoint testing
- Inference endpoint testing
- Volume persistence verification
- Resource limit testing
- Multi-container orchestration
- Security scan (trivy, docker scan)

**Key Deliverables**:
- Test scripts
- Validation checklist
- Performance benchmarks
- Security scan results

---

## Architecture Decisions

### Base Image Strategy
- **Builder**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` (includes compilers, headers)
- **Runtime**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` (production-optimized)
- **Rationale**: Official PyTorch images with CUDA 12.4 support, multi-stage for size reduction

### Model Artifact Strategy
- **Approach**: Volume mounts (not embedded in image)
- **Rationale**: Decouple model updates from image builds, enable dynamic model swapping
- **Structure**:
  - `/app/models` - Model checkpoints (volume mount)
  - `/app/logs` - Application logs (volume)
  - `/app/.cache` - Model cache (named volume)

### GPU Configuration
- **Runtime**: NVIDIA Container Toolkit
- **Device Reservation**: `device_ids: ['0']` for specific GPU
- **Memory Limit**: 2-3x model size (ResNet50 ~100MB → 4GB limit)

### Worker Configuration
- **Development**: Single Uvicorn worker (hot-reload)
- **Production**: Gunicorn + Uvicorn workers (preload for memory efficiency)
- **GPU Consideration**: Single worker per GPU to avoid contention

---

## Success Criteria

- [x] Multi-stage Dockerfile builds successfully
- [x] Image size <2GB (runtime stage)
- [x] GPU accessible inside container (torch.cuda.is_available() == True)
- [x] Health check passes (API responds at /health)
- [x] Inference endpoint returns predictions
- [x] Model volume mount works correctly
- [x] Docker Compose brings up service with GPU
- [x] Non-root user enforced
- [x] No critical vulnerabilities (trivy scan)
- [x] Startup time <60s (including model loading)
- [x] Memory usage within limits

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU drivers mismatch | High | Pin CUDA version (12.4), document compatibility |
| Model file not found | High | Volume mount validation, clear error messages |
| OOM during startup | Medium | Memory limits, health check start_period=60s |
| Large image size | Low | Multi-stage build, layer optimization |
| Security vulnerabilities | Medium | Non-root user, security scanning, distroless consideration |
| Port conflicts | Low | Configurable ports via .env |

---

## Security Considerations

1. **Non-root User**: UID 1000 (`appuser`)
2. **Read-only Volumes**: Model volumes mounted as `:ro`
3. **No Secrets in Image**: Environment variables only
4. **Minimal Attack Surface**: Runtime image, no build tools
5. **Version Pinning**: All dependencies locked
6. **Security Scanning**: Trivy/docker scan in CI
7. **Network Isolation**: Bridge network, localhost binding

---

## Dependencies

**Host Requirements**:
- Docker Engine 20.10+
- Docker Compose 2.x
- NVIDIA Container Toolkit
- NVIDIA GPU with drivers (compatible with CUDA 12.4)

**Verification**:
```bash
docker --version          # >= 20.10
docker compose version    # >= 2.0
nvidia-smi               # GPU visible
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## Implementation Order

1. **Phase 01** (Dockerfile) - Foundation, must be first
2. **Phase 02** (Docker Compose) - Depends on Phase 01
3. **Phase 03** (Scripts/Docs) - Parallel with Phase 04
4. **Phase 04** (Testing) - Final validation

**Estimated Timeline**: 4-6 hours (development + testing)

---

## Related Files

**Source Code**:
- `/home/minh-ub/projects/multi-label-classification/api/main.py`
- `/home/minh-ub/projects/multi-label-classification/api/config.py`
- `/home/minh-ub/projects/multi-label-classification/requirements.txt`

**Research Reports**:
- `/home/minh-ub/projects/multi-label-classification/plans/reports/docker-ml-best-practices.md`
- `/home/minh-ub/projects/multi-label-classification/plans/reports/docker-compose-ml-patterns.md`
- `/home/minh-ub/projects/multi-label-classification/plans/reports/pytorch-serving-patterns.md`

**Deliverables** (to be created):
- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`
- `docker-compose.override.yml`
- `.env.example`
- `scripts/docker-build.sh`
- `scripts/docker-run.sh`
- `scripts/docker-stop.sh`
- `docs/DOCKER_DEPLOYMENT.md`

---

## Todo List

- [ ] Phase 01: Create multi-stage Dockerfile with GPU support
- [ ] Phase 01: Create .dockerignore file
- [ ] Phase 02: Create docker-compose.yml (base config)
- [ ] Phase 02: Create docker-compose.override.yml (dev config)
- [ ] Phase 02: Create .env.example
- [ ] Phase 03: Create build script
- [ ] Phase 03: Create run script
- [ ] Phase 03: Create stop script
- [ ] Phase 03: Update README.md
- [ ] Phase 03: Create DOCKER_DEPLOYMENT.md
- [ ] Phase 04: Test image build
- [ ] Phase 04: Validate GPU access
- [ ] Phase 04: Test health checks
- [ ] Phase 04: Test inference endpoint
- [ ] Phase 04: Run security scan

---

## Notes

- **CUDA Version**: Locked to 12.4 (matches PyTorch 2.4.0 official images)
- **Model Path**: Default `outputs/checkpoints/fold_0/best_model.pt`, override via `API_CHECKPOINT_PATH`
- **Port**: Default 8000, configurable via `API_PORT`
- **Memory**: Recommend 8GB+ for model + inference buffer
- **GPU**: Single GPU required, multi-GPU support future enhancement

---

## Next Steps

1. Review this plan
2. Proceed to Phase 01: Create Dockerfile
3. Test build locally
4. Move to Phase 02: Docker Compose configuration
