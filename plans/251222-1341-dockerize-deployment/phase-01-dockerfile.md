# Phase 01: Multi-Stage Dockerfile with GPU Support

**Date**: 2025-12-22
**Status**: Pending
**Priority**: Critical
**Dependencies**: None (foundation phase)

---

## Context

Research reports informing this phase:
- [docker-ml-best-practices.md](../reports/docker-ml-best-practices.md) - Multi-stage patterns, GPU configuration, layer optimization
- [pytorch-serving-patterns.md](../reports/pytorch-serving-patterns.md) - Model loading strategies, inference optimization

---

## Overview

Create production-ready multi-stage Dockerfile for Cat Breeds Classification FastAPI service with NVIDIA GPU support.

**Goals**:
- Optimized image size (<2GB runtime)
- GPU access via CUDA 12.4 + cuDNN 9
- Layer caching for fast rebuilds
- Security hardening (non-root user)
- Health check integration
- Volume mount points for models

---

## Key Insights from Research

### Multi-Stage Build Benefits
- **Size Reduction**: 50-95% smaller images (eliminates compilers, build tools)
- **Security**: Runtime images lack build toolchains
- **Cache Optimization**: Dependencies installed once in builder stage

### Base Image Selection
- **Builder**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
  - Includes CUDA toolkit, compilers, headers
  - Required for building Python packages with native extensions
- **Runtime**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`
  - Production-optimized, shared CUDA libs only
  - Smaller size, minimal attack surface

### Layer Caching Strategy
```dockerfile
# GOOD: Dependencies cached separately
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# BAD: Code changes invalidate dependency layer
COPY . .
RUN pip install -r requirements.txt
```

### GPU Verification
```dockerfile
RUN python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Requirements

### Functional Requirements
1. Multi-stage build (builder + runtime)
2. Python 3.12 compatibility
3. PyTorch 2.4.0 with CUDA 12.4 support
4. All dependencies from `requirements.txt`
5. FastAPI application code
6. Health check endpoint at `/health`
7. Model volume mount point at `/app/models`
8. GPU access enabled

### Non-Functional Requirements
1. Image size: <2GB (runtime stage)
2. Build time: <5 min (with cold cache)
3. Rebuild time: <1 min (with warm cache)
4. Security: Non-root user (UID 1000)
5. Startup time: <60s (including model load)
6. No critical vulnerabilities

---

## Architecture Decisions

### Stage 1: Builder
- **Base**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
- **Purpose**: Install Python dependencies with native extensions
- **Output**: `/root/.local` with installed packages

### Stage 2: Runtime
- **Base**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`
- **Purpose**: Production execution environment
- **Copies**: Python packages from builder, application code

### User Configuration
- **User**: `appuser` (UID 1000, GID 1000)
- **Home**: `/home/appuser`
- **Working Dir**: `/app`
- **Rationale**: Non-root reduces attack surface, standard UID for compatibility

### Volume Structure
```
/app                    # Application code (COPY from build context)
/app/models             # Model checkpoints (volume mount)
/app/logs               # Application logs (volume mount)
/home/appuser/.cache    # Model cache (named volume)
```

### Environment Variables
```bash
PYTHONUNBUFFERED=1          # Real-time logging
PATH=/home/appuser/.local/bin:$PATH  # User-installed packages
API_CHECKPOINT_PATH=/app/models/best_model.pt  # Default model path
```

---

## Related Code Files

**Source Files**:
- `/home/minh-ub/projects/multi-label-classification/api/main.py` - FastAPI application with lifespan model loading
- `/home/minh-ub/projects/multi-label-classification/api/config.py` - Pydantic settings (env var support)
- `/home/minh-ub/projects/multi-label-classification/requirements.txt` - Python dependencies

**Key Dependencies** (from requirements.txt):
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
albumentations>=1.3.0
pydantic-settings>=2.0.0
```

---

## Implementation Steps

### Step 1: Create .dockerignore
**Purpose**: Exclude unnecessary files from build context (faster builds, smaller images)

**File**: `/home/minh-ub/projects/multi-label-classification/.dockerignore`

**Content**:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
env/

# Data and outputs
data/
outputs/
checkpoints/
*.pt
*.pth
*.pkl
*.h5

# Notebooks and EDA
*.ipynb
.ipynb_checkpoints/

# MLflow
mlruns/
mlartifacts/

# Git
.git/
.gitignore
.gitattributes

# Docker
Dockerfile*
docker-compose*.yml
.dockerignore

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Documentation
*.md
docs/

# Tests
tests/
.pytest_cache/
.coverage
htmlcov/

# Scripts (development only)
scripts/download_dataset.sh
scripts/validate_env.py
setup_training_server.sh
```

**Rationale**:
- Exclude `data/` and `outputs/` (large training artifacts)
- Exclude `.venv/` (dependencies installed in container)
- Exclude notebooks (EDA only, not needed in container)
- Keep `api/` and `requirements.txt` (needed for build)

---

### Step 2: Create Multi-Stage Dockerfile
**File**: `/home/minh-ub/projects/multi-label-classification/Dockerfile`

**Content**:
```dockerfile
# =============================================================================
# Stage 1: Builder
# Purpose: Install Python dependencies with native extensions
# =============================================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS builder

WORKDIR /build

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# Purpose: Production-optimized execution environment
# =============================================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy installed packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser api/ /app/api/

# Create directories for volumes
RUN mkdir -p /app/models /app/logs /home/appuser/.cache && \
    chown -R appuser:appuser /app/models /app/logs /home/appuser/.cache

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    API_CHECKPOINT_PATH=/app/models/best_model.pt \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Verify GPU access (build-time check)
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
# Development: Single Uvicorn worker
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Production alternative (uncomment for Gunicorn):
# CMD ["gunicorn", "api.main:app", \
#      "--workers", "4", \
#      "--worker-class", "uvicorn.workers.UvicornWorker", \
#      "--bind", "0.0.0.0:8000", \
#      "--timeout", "120", \
#      "--graceful-timeout", "30", \
#      "--access-logfile", "-", \
#      "--error-logfile", "-"]
```

**Key Features**:
1. **Multi-stage build**: Builder (devel) + Runtime (runtime)
2. **Layer caching**: requirements.txt copied before code
3. **Non-root user**: appuser (UID 1000)
4. **Volume mount points**: /app/models, /app/logs, /home/appuser/.cache
5. **Health check**: 30s interval, 60s startup grace period
6. **GPU verification**: Build-time CUDA check
7. **Configurable**: Environment variables for settings

---

### Step 3: Validate Dockerfile Syntax
```bash
cd /home/minh-ub/projects/multi-label-classification
docker build --target builder -t cat-breeds-api:builder .
docker build -t cat-breeds-api:latest .
```

**Expected Output**:
- Builder stage completes successfully
- Runtime stage completes successfully
- GPU verification prints CUDA availability
- Image tagged as `cat-breeds-api:latest`

---

### Step 4: Test Image Build
```bash
# Build with progress output
docker build -t cat-breeds-api:latest .

# Check image size
docker images cat-breeds-api:latest

# Expected: ~1.5-2GB (runtime stage)
```

---

### Step 5: Verify GPU Access
```bash
# Test GPU inside container
docker run --rm --gpus all cat-breeds-api:latest \
  python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Expected Output**:
```
CUDA: True
Device: NVIDIA GeForce RTX 3090 (or your GPU model)
```

---

## Todo List

- [ ] Create .dockerignore file
- [ ] Create multi-stage Dockerfile
- [ ] Build image (cold cache)
- [ ] Verify image size (<2GB)
- [ ] Test GPU access inside container
- [ ] Rebuild image (warm cache, <1 min)
- [ ] Verify CUDA version (12.4)
- [ ] Test health check endpoint
- [ ] Document build process

---

## Success Criteria

- [x] `.dockerignore` excludes training data, notebooks, venv
- [x] Dockerfile builds without errors
- [x] Builder stage installs all dependencies
- [x] Runtime stage <2GB
- [x] GPU accessible inside container
- [x] Non-root user enforced
- [x] Health check configured
- [x] Volume mount points created
- [x] Environment variables set correctly
- [x] Build time <5 min (cold), <1 min (warm)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dependency conflicts | High | Pin exact versions in requirements.txt |
| GPU driver mismatch | High | Use official PyTorch CUDA 12.4 images |
| Build fails on builder stage | Medium | Test requirements.txt locally first |
| Image too large | Low | Multi-stage build, .dockerignore optimization |
| Health check fails | Medium | start_period=60s for model loading time |

---

## Security Considerations

1. **Non-root user**: UID 1000 (appuser)
2. **Minimal base**: Runtime image only
3. **No secrets**: Environment variables only
4. **Version pinning**: All dependencies locked
5. **Vulnerability scanning**: `docker scan cat-breeds-api:latest`
6. **Read-only filesystem**: Consider `--read-only` flag for production

---

## Testing Validation

### Build Test
```bash
docker build -t cat-breeds-api:test .
```

### Size Test
```bash
docker images cat-breeds-api:test --format "{{.Size}}"
# Expected: <2GB
```

### GPU Test
```bash
docker run --rm --gpus all cat-breeds-api:test python -c "import torch; assert torch.cuda.is_available()"
```

### User Test
```bash
docker run --rm cat-breeds-api:test whoami
# Expected: appuser
```

### Health Check Test
```bash
docker run -d --name api-test -p 8000:8000 cat-breeds-api:test
sleep 5
docker inspect api-test --format='{{.State.Health.Status}}'
# Expected: healthy (after start_period)
docker rm -f api-test
```

---

## Next Steps

1. Create .dockerignore
2. Create Dockerfile
3. Test build locally
4. Verify all success criteria
5. Move to Phase 02: Docker Compose configuration

---

## Unresolved Questions

- Should we include Gunicorn by default or keep single Uvicorn worker? (Decision: Gunicorn commented out, enable via docker-compose CMD override)
- Model warming in Dockerfile vs runtime? (Decision: Runtime via lifespan event)
- Distroless base consideration for even smaller image? (Future optimization, stick with official PyTorch images for now)
