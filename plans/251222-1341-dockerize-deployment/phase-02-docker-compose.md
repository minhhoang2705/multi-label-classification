# Phase 02: Docker Compose Configuration with GPU Support

**Date**: 2025-12-22
**Status**: Pending
**Priority**: Critical
**Dependencies**: Phase 01 (Dockerfile must be completed)

---

## Context

Research reports informing this phase:
- [docker-compose-ml-patterns.md](../reports/docker-compose-ml-patterns.md) - Orchestration, volumes, GPU configuration, health checks
- [docker-ml-best-practices.md](../reports/docker-ml-best-practices.md) - GPU setup, resource limits

---

## Overview

Create Docker Compose configuration for local deployment with GPU support, volume management, and environment variable configuration.

**Goals**:
- GPU device reservation via NVIDIA runtime
- Named volumes for models, logs, cache
- Environment variable management (.env)
- Health checks with model loading grace period
- Resource limits (CPU, memory, GPU)
- Development and production configurations

---

## Key Insights from Research

### GPU Configuration in Compose
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0']  # Specific GPU
          capabilities: [gpu]
```

### Volume Strategies
- **Named volumes**: Portable, Docker-managed (logs, cache)
- **Bind mounts**: Development only (code hot-reload)
- **Host paths**: Large model files (via driver_opts)

### Health Checks
- **start_period**: Must be ≥ model loading time (60s for ML models)
- **interval**: 30s (production standard)
- **retries**: 3 failures → unhealthy

### Resource Limits for ML
- **Memory**: 2-3x model size (ResNet50 ~100MB → 4GB limit)
- **CPU**: Match worker count
- **GPU**: Single GPU per service to avoid contention

### Override Pattern
- `docker-compose.yml`: Production-ready base config
- `docker-compose.override.yml`: Auto-loaded dev overrides (hot-reload, debug ports)
- `docker-compose.prod.yml`: Production-specific settings (Gunicorn, replicas)

---

## Requirements

### Functional Requirements
1. GPU access via NVIDIA runtime
2. Service definition for cat-breeds-api
3. Named volumes: models, logs, cache
4. Environment variables via .env file
5. Health check with 60s start period
6. Port mapping (8000)
7. Restart policy (unless-stopped)
8. Bridge network for isolation

### Non-Functional Requirements
1. Startup time: <60s (including model load)
2. Graceful shutdown on SIGTERM
3. Auto-recovery from crashes (restart policy)
4. Memory limit: 8GB (model + inference buffer)
5. CPU limit: 4 cores
6. Development workflow support (code hot-reload)

---

## Architecture Decisions

### Service Configuration
- **Name**: `cat-breeds-api`
- **Image**: `cat-breeds-api:latest` (built from Phase 01)
- **Network**: Bridge (default, isolated)
- **Restart**: `unless-stopped` (manual stop persists)

### Volume Strategy
```
models:     Named volume with bind mount to host path (large model files)
logs:       Named volume (Docker-managed, persistent logs)
cache:      Named volume (model cache, versioned)
```

### Environment Variables
**Required**:
- `API_CHECKPOINT_PATH`: Model file path
- `API_PORT`: API port (default 8000)
- `API_HOST`: Bind address (default 0.0.0.0)
- `API_DEVICE`: Device (auto, cuda, cpu)

**Optional**:
- `API_MODEL_NAME`: Model architecture (resnet50, efficientnet_b3)
- `API_NUM_CLASSES`: Number of classes (67)
- `API_IMAGE_SIZE`: Input size (224)

### Resource Limits
```yaml
limits:
  cpus: '4.0'
  memory: 8G
reservations:
  cpus: '2.0'
  memory: 4G
  devices:
    - driver: nvidia
      device_ids: ['0']
      capabilities: [gpu]
```

---

## Related Code Files

**Configuration**:
- `/home/minh-ub/projects/multi-label-classification/api/config.py` - Pydantic settings (env var support via `API_` prefix)
- `/home/minh-ub/projects/multi-label-classification/api/main.py` - Lifespan event for model loading

**Existing Model**:
- `/home/minh-ub/projects/multi-label-classification/outputs/checkpoints/fold_0/best_model.pt` - Trained model checkpoint

---

## Implementation Steps

### Step 1: Create Base docker-compose.yml
**Purpose**: Production-ready base configuration

**File**: `/home/minh-ub/projects/multi-label-classification/docker-compose.yml`

**Content**:
```yaml
version: '3.8'

services:
  cat-breeds-api:
    image: cat-breeds-api:latest
    container_name: cat-breeds-api
    build:
      context: .
      dockerfile: Dockerfile

    # GPU support (requires NVIDIA Container Toolkit)
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
          devices:
            - driver: nvidia
              device_ids: ['0']  # Use first GPU
              capabilities: [gpu]

    # Port mapping (localhost only)
    ports:
      - "127.0.0.1:8000:8000"

    # Environment variables (from .env file)
    environment:
      - API_CHECKPOINT_PATH=${API_CHECKPOINT_PATH:-/app/models/best_model.pt}
      - API_MODEL_NAME=${API_MODEL_NAME:-resnet50}
      - API_NUM_CLASSES=${API_NUM_CLASSES:-67}
      - API_IMAGE_SIZE=${API_IMAGE_SIZE:-224}
      - API_DEVICE=${API_DEVICE:-auto}
      - API_HOST=0.0.0.0
      - API_PORT=8000

    # Volume mounts
    volumes:
      - models:/app/models:ro  # Read-only model weights
      - logs:/app/logs
      - cache:/home/appuser/.cache

    # Health check (60s grace period for model loading)
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

    # Restart policy
    restart: unless-stopped

    # Network
    networks:
      - ml-network

volumes:
  models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${PWD}/outputs/checkpoints/fold_0  # Bind to host model directory
  logs:
    name: cat_breeds_logs
  cache:
    name: cat_breeds_cache_v1

networks:
  ml-network:
    driver: bridge
```

**Key Features**:
1. GPU reservation with specific device ID
2. Resource limits (CPU, memory)
3. Health check with 60s start period
4. Named volumes with host bind mount for models
5. Environment variable support
6. Localhost-only port binding (security)
7. Restart policy for auto-recovery

---

### Step 2: Create Development Override
**Purpose**: Auto-loaded development configuration (hot-reload, debug)

**File**: `/home/minh-ub/projects/multi-label-classification/docker-compose.override.yml`

**Content**:
```yaml
version: '3.8'

services:
  cat-breeds-api:
    # Override build target (if multi-target Dockerfile)
    build:
      context: .
      dockerfile: Dockerfile
      # target: development  # Uncomment if Dockerfile has dev stage

    # Bind mount code for hot-reload
    volumes:
      - ./api:/app/api:ro  # Read-only code mount

    # Override CMD for hot-reload
    command: ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

    # Development environment variables
    environment:
      - API_DEVICE=cuda  # Force CUDA in dev
      - LOG_LEVEL=DEBUG

    # Expose additional ports (debugger, monitoring)
    ports:
      - "127.0.0.1:8000:8000"
      - "127.0.0.1:5678:5678"  # Python debugger (debugpy)
```

**Key Features**:
1. Code hot-reload via bind mount + --reload flag
2. Debug port for Python debugger
3. Force CUDA device in dev
4. Verbose logging (DEBUG)

**Usage**:
```bash
# Auto-loads override in same directory
docker compose up

# Explicitly specify
docker compose -f docker-compose.yml -f docker-compose.override.yml up
```

---

### Step 3: Create .env.example
**Purpose**: Template for environment variables (commit to repo)

**File**: `/home/minh-ub/projects/multi-label-classification/.env.example`

**Content**:
```bash
# =============================================================================
# Cat Breeds Classification API - Environment Variables
# =============================================================================
# Copy this file to .env and customize for your environment
# cp .env.example .env

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
API_CHECKPOINT_PATH=/app/models/best_model.pt
API_MODEL_NAME=resnet50  # resnet50, efficientnet_b3, convnext_base
API_NUM_CLASSES=67
API_IMAGE_SIZE=224

# -----------------------------------------------------------------------------
# Device Configuration
# -----------------------------------------------------------------------------
API_DEVICE=auto  # auto (detect GPU/CPU), cuda, mps, cpu

# -----------------------------------------------------------------------------
# Server Configuration
# -----------------------------------------------------------------------------
API_HOST=0.0.0.0
API_PORT=8000

# -----------------------------------------------------------------------------
# CORS Configuration (add your frontend origins)
# -----------------------------------------------------------------------------
# API_CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# -----------------------------------------------------------------------------
# GPU Configuration (Docker Compose)
# -----------------------------------------------------------------------------
GPU_DEVICE_ID=0  # GPU device ID (0 for first GPU)

# -----------------------------------------------------------------------------
# Resource Limits
# -----------------------------------------------------------------------------
CPU_LIMIT=4.0
MEMORY_LIMIT=8G
CPU_RESERVATION=2.0
MEMORY_RESERVATION=4G
```

**Instructions**:
1. Copy to `.env`: `cp .env.example .env`
2. Customize values (model path, GPU ID, etc.)
3. `.env` is gitignored (never commit)

---

### Step 4: Create .env (User-Specific, Gitignored)
**Purpose**: Actual environment file (not committed)

**File**: `/home/minh-ub/projects/multi-label-classification/.env`

**Content**:
```bash
# Local environment (auto-generated)
API_CHECKPOINT_PATH=/app/models/best_model.pt
API_MODEL_NAME=resnet50
API_NUM_CLASSES=67
API_IMAGE_SIZE=224
API_DEVICE=auto
API_HOST=0.0.0.0
API_PORT=8000
GPU_DEVICE_ID=0
CPU_LIMIT=4.0
MEMORY_LIMIT=8G
CPU_RESERVATION=2.0
MEMORY_RESERVATION=4G
```

**Note**: Add `.env` to `.gitignore`

---

### Step 5: Update .gitignore
**File**: `/home/minh-ub/projects/multi-label-classification/.gitignore`

**Add**:
```
# Docker environment variables
.env

# Docker volumes
docker-volumes/
```

---

## Todo List

- [ ] Create base docker-compose.yml
- [ ] Create docker-compose.override.yml (dev)
- [ ] Create .env.example (template)
- [ ] Create .env (local, gitignored)
- [ ] Update .gitignore (exclude .env)
- [ ] Test `docker compose up` (auto-loads override)
- [ ] Verify GPU access in container
- [ ] Test health check endpoint
- [ ] Test volume persistence (model, logs)
- [ ] Test code hot-reload (dev mode)

---

## Success Criteria

- [x] docker-compose.yml defines service with GPU support
- [x] Named volumes created correctly (models, logs, cache)
- [x] .env variables loaded into container
- [x] Health check passes after start_period
- [x] GPU accessible inside container
- [x] Model volume mount works (can read model file)
- [x] Port 8000 accessible from host
- [x] Restart policy enforced
- [x] Resource limits applied
- [x] Development override enables hot-reload

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| NVIDIA runtime not installed | High | Document prerequisite, provide install script |
| Model volume mount fails | High | Validate path in .env, clear error messages |
| GPU OOM with resource limits | Medium | Memory limit 2-3x model size (8GB safe for ResNet50) |
| Port 8000 already in use | Low | Configurable via API_PORT env var |
| Health check fails on startup | Medium | start_period=60s accommodates model loading |

---

## Security Considerations

1. **Localhost binding**: `127.0.0.1:8000` prevents external access
2. **Read-only volumes**: Model volume mounted as `:ro`
3. **Bridge network**: Service isolated from host network
4. **No secrets in compose**: All sensitive data via .env (gitignored)
5. **Non-root user**: Enforced by Dockerfile

---

## Testing Validation

### Compose Up Test
```bash
cd /home/minh-ub/projects/multi-label-classification
docker compose up -d
docker compose logs -f
```

**Expected**:
- Service starts successfully
- Model loads (check logs)
- Health check passes (after 60s)

### GPU Test
```bash
docker compose exec cat-breeds-api python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Expected**: `CUDA: True`

### Health Check Test
```bash
curl http://localhost:8000/health
```

**Expected**: `200 OK` with health status

### Inference Test
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

**Expected**: JSON response with predictions

### Volume Persistence Test
```bash
# Create test log
docker compose exec cat-breeds-api touch /app/logs/test.log

# Restart container
docker compose restart

# Check if log persists
docker compose exec cat-breeds-api ls /app/logs/test.log
```

**Expected**: File persists across restarts

### Hot-Reload Test (Dev Mode)
```bash
# Start in dev mode
docker compose up -d

# Edit api/main.py (add comment)
# Check logs for reload message
docker compose logs -f cat-breeds-api
```

**Expected**: Uvicorn detects change and reloads

---

## Development Workflow

### Daily Development
```bash
# Start services (auto-loads override)
docker compose up -d

# Watch logs
docker compose logs -f

# Make code changes (auto-reload in dev mode)

# Restart if needed
docker compose restart

# Stop services
docker compose down
```

### Production Deployment (Future)
```bash
# Use production config
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Next Steps

1. Create docker-compose.yml
2. Create docker-compose.override.yml
3. Create .env.example and .env
4. Update .gitignore
5. Test all validation scenarios
6. Move to Phase 03: Deployment scripts and documentation

---

## Unresolved Questions

- Should we add MLflow service for experiment tracking? (Decision: Separate deployment, out of scope)
- Multi-GPU support via replicas? (Future enhancement, single GPU sufficient for now)
- Production Gunicorn configuration in separate compose file? (Yes, docker-compose.prod.yml)
