# Docker Deployment Test Report - Cat Breeds Classification API

**Test Date:** 2025-12-23
**Project:** /home/minh-ub/projects/multi-label-classification
**Tester:** QA Automation (tester subagent)

---

## Executive Summary

**Overall Status:** PARTIAL PASS (Build Successful, Runtime Configuration Issue Detected)

- **Critical Issues:** 1 BLOCKER
- **Warnings:** 2
- **Tests Passed:** 15/18
- **Tests Failed:** 3/18

### Quick Status

| Phase | Status | Details |
|-------|--------|---------|
| Build Tests | ✓ PASS | Image built successfully in 282s |
| Image Inspection | ✓ PASS | Security and config compliant |
| Compose Validation | ✓ PASS | Syntax valid, GPU configured |
| Container Startup | ✗ FAIL | Path validation blocks model load |
| API Endpoints | ⊘ SKIP | App failed to start |
| GPU Tests | ~ PARTIAL | Hardware detected, runtime blocked |

---

## Phase 1: Build Tests - PASSED ✓

### Build Success
- ✓ Docker image build completed successfully
- ✓ Build script executed without errors
- ✓ Multi-stage build (builder + runtime)

### Build Metrics
| Metric | Value | Status |
|--------|-------|--------|
| Build Time | 282s (4m 42s) | ✓ Acceptable |
| Image Size | 13.5 GB | ⚠ Large but acceptable for ML |
| Base Image | pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime | ✓ |
| Python | 3.11 | ✓ |
| PyTorch | 2.4.0 | ✓ |
| CUDA | 12.4 | ✓ |

### Dependencies Installed
- **Core ML:** pandas, numpy, torch (2.4.0), torchvision, timm (1.0.22), albumentations (2.0.8)
- **API:** fastapi (0.127.0), uvicorn (0.40.0), pydantic-settings (2.12.0)
- **MLOps:** mlflow (3.8.0), scikit-learn (1.8.0)
- **Dev Tools:** jupyter, notebook, ipykernel, matplotlib, seaborn

### Build Warnings
⚠ **Image size 13.5GB exceeds recommended 3GB threshold**
- Cause: PyTorch CUDA base image (~10GB) + ML dependencies
- Impact: Slower deployment, higher storage cost
- Mitigation: Acceptable for ML applications with GPU requirements

---

## Phase 2: Image Inspection - PASSED ✓

### Security Configuration
| Aspect | Value | Status |
|--------|-------|--------|
| User | appuser (UID 1000, GID 1000) | ✓ Non-root |
| Working Dir | /app | ✓ |
| Exposed Ports | 8000/tcp | ✓ |
| Health Check | Configured (30s interval) | ✓ |

### Image Details
- **Image ID:** d6422fea59ba
- **Architecture:** amd64
- **OS:** linux (Ubuntu 22.04)
- **Created:** 2025-12-23 07:54:55 UTC
- **Layers:** 13
- **Compressed Size:** ~4.27 GB

### Environment Variables
```bash
PATH=/home/appuser/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:...
PYTHONUNBUFFERED=1
API_CHECKPOINT_PATH=/app/models/best_model.pt
API_HOST=0.0.0.0
API_PORT=8000
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Health Check Configuration
```bash
Test: curl -f http://localhost:8000/health/live || exit 1
Interval: 30s
Timeout: 10s
Start Period: 60s  # Model loading grace period
Retries: 3
```

### Volumes
- `/app/models` - Model weights (read-only)
- `/app/logs` - Application logs
- `/home/appuser/.cache` - Pip/HuggingFace cache

### CMD
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Phase 3: Docker Compose Validation - PASSED ✓

### Validation Results
- ✓ docker-compose.yml syntax valid
- ✓ docker-compose.override.yml syntax valid
- ✓ Merged configuration valid
- **Docker Compose Version:** v5.0.0

### Services
- **cat-breeds-api:** Main API service with GPU support

### Networks
- **ml-network:** Bridge network for service isolation

### Volumes
| Volume | Type | Source | Target | Mode |
|--------|------|--------|--------|------|
| models | bind | outputs/checkpoints/fold_0 | /app/models | ro |
| logs | named | cat_breeds_logs | /app/logs | rw |
| cache | named | cat_breeds_cache_v1 | /home/appuser/.cache | rw |

### Resource Limits
```yaml
CPU Limit: 4 cores
Memory Limit: 8GB
CPU Reservation: 2 cores
Memory Reservation: 4GB
```

### GPU Configuration
```yaml
Driver: nvidia
Device IDs: ['0']  # First GPU
Capabilities: [gpu]
```

### Port Mappings
- `127.0.0.1:8000:8000` - API endpoint (localhost only, secure)
- `127.0.0.1:5678:5678` - debugpy (dev only)

### Development Overrides
- Hot-reload enabled (`--reload` flag)
- Code bind mounts (./api, ./src) read-only
- Debug port exposed (5678)
- LOG_LEVEL=DEBUG
- API_DEVICE=cuda (forced)

### Warnings
⚠ **version attribute obsolete** in compose files (cosmetic, no functional impact)
⚠ **Port 5678 exposed** for debugpy (dev only, remove in production)

---

## Phase 4: Container Startup - FAILED ✗

### Container Lifecycle
| Step | Status | Details |
|------|--------|---------|
| Container created | ✓ | ID: 5b144125a7f9 |
| Container started | ✓ | Port binding successful |
| Uvicorn initialized | ✓ | Server process started |
| Application startup | ✗ | Model loading failed |

### Startup Logs
```
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: Started reloader process [1] using WatchFiles
INFO: Started server process [8]
INFO: Starting Cat Breeds Classification API
ERROR: Failed to load model
```

### CRITICAL ERROR
```python
ValueError: Invalid checkpoint path. Must be within outputs/checkpoints/ directory.
Got: /app/models/best_model.pt
```

### Root Cause Analysis

**Location:** `api/services/model_service.py` line 158

**Problem:** Hardcoded path validation
```python
allowed_base = Path("outputs/checkpoints").resolve()
# Resolves to /app/outputs/checkpoints/ inside container
```

**Path Mismatch:**
```
Expected:      /app/outputs/checkpoints/best_model.pt
Actual Mount:  /app/models/best_model.pt
Config Env:    API_CHECKPOINT_PATH=/app/models/best_model.pt
```

**Files Present:**
- ✓ Host model exists: `outputs/checkpoints/fold_0/best_model.pt` (94.9 MB)
- ✓ Volume mount configured correctly
- ✗ Path validation rejects mounted path

**Impact:**
- Container starts but application fails during startup
- Health check status: starting (never reaches healthy)
- API endpoints not accessible
- GPU test cannot proceed

---

## Phase 5: API Endpoint Tests - SKIPPED

Cannot test endpoints due to startup failure:

- `GET /health/live` - NOT TESTED
- `GET /` - NOT TESTED
- `GET /docs` - NOT TESTED
- `POST /predict` - NOT TESTED

---

## Phase 6: GPU Tests - PARTIAL ✓

### Host GPU Status
| Property | Value | Status |
|----------|-------|--------|
| GPU | NVIDIA GeForce RTX 4070 Ti | ✓ Detected |
| Driver | 590.44.01 | ✓ |
| CUDA | 13.1 | ✓ |
| VRAM | 12282 MiB | ✓ |
| Processes | 0 running | ✓ Available |

### Build-time GPU Check
```
CUDA available: False  # Expected during build
CUDA version: 12.4     # Correct
```

### Runtime GPU Check
- ✗ Cannot verify (application failed to start)
- ✗ `torch.cuda.is_available()` not tested in container

### Container GPU Configuration
- ✓ NVIDIA device configured in compose
- ✓ nvidia runtime requested
- ✓ GPU device 0 assigned

**Note:** GPU hardware available but runtime test blocked by startup failure.

---

## Critical Issues

### Issue #1: Path Validation Prevents Model Loading [BLOCKER]

**Severity:** CRITICAL - BLOCKER
**Component:** api/services/model_service.py
**Location:** Line 158 (_validate_checkpoint_path)

**Problem:**
Hardcoded path validation requires models in `outputs/checkpoints/` but Docker mounts them at `/app/models/`. Application startup fails.

**Code:**
```python
allowed_base = Path("outputs/checkpoints").resolve()
# Resolves to /app/outputs/checkpoints/
```

**Expected Behavior:**
Should accept checkpoint path from `API_CHECKPOINT_PATH` environment variable regardless of directory structure.

**Recommendations:**
1. Remove hardcoded path validation for Docker deployments
2. Add `ALLOW_EXTERNAL_CHECKPOINTS` config flag
3. Validate only file existence, not directory structure
4. OR: Change Docker mount from `/app/models` to `/app/outputs/checkpoints`

**Workaround (Option A - Change Docker config):**
```yaml
# docker-compose.yml
volumes:
  - models:/app/outputs/checkpoints/fold_0:ro  # Changed path

environment:
  API_CHECKPOINT_PATH=outputs/checkpoints/fold_0/best_model.pt  # Relative path
```

**Workaround (Option B - Fix code):**
```python
# api/services/model_service.py
def _validate_checkpoint_path(self, checkpoint_path: str) -> Path:
    checkpoint_file = Path(checkpoint_path).resolve()

    # Only validate file exists, not directory structure
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_file
```

---

## Warnings & Recommendations

### Warning #1: Image Size Exceeds Threshold
**Severity:** MEDIUM

**Current:** 13.5 GB
**Threshold:** 3 GB (acceptable), 2 GB (recommended)
**Reason:** PyTorch CUDA base image (10GB+) + ML dependencies

**Recommendations:**
- Consider CPU-only base for inference-only deployments
- Use `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` (already used)
- Implement multi-stage build cleanup (already implemented)
- Consider model quantization to reduce checkpoint size
- Split requirements: requirements-prod.txt vs requirements-dev.txt

### Warning #2: Development Features in Production Image
**Severity:** LOW

**Issues:**
- Jupyter, notebook, ipykernel installed (~180MB+)
- debugpy port exposed (5678)
- `--reload` flag in dev override

**Recommendations:**
- Create separate requirements-dev.txt and requirements-prod.txt
- Use multi-target Dockerfile (dev vs prod stages)
- Remove dev dependencies from production builds
- Disable `--reload` and debug ports in production

### Warning #3: Compose Version Attribute Obsolete
**Severity:** COSMETIC

**Files:** docker-compose.yml, docker-compose.override.yml
**Line:** `version: '3.8'`

**Recommendation:**
Remove version attribute (obsolete in Docker Compose v2+)

---

## Performance Metrics

### Build Performance
| Metric | Value |
|--------|-------|
| Build Time (no cache) | 282s (4m 42s) |
| Build Time (with cache) | ~30-60s (estimated) |
| Image Size | 13.5 GB |
| Layer Count | 13 layers |
| Compressed Size | ~4.27 GB |

### Expected Runtime Performance
| Metric | Value |
|--------|-------|
| Model Load Time | ~5-10s (90MB checkpoint) |
| Cold Start | ~15-20s (with model loading) |
| Health Check Grace | 60s (sufficient) |
| Memory Usage | ~2-4GB (model + framework) |

---

## Compliance & Best Practices

### Security - PASS ✓
- ✓ Non-root user (appuser)
- ✓ Read-only volume mounts for code
- ✓ Port binding to localhost only (127.0.0.1)
- ✓ No secrets in environment variables
- ✓ Minimal attack surface

### Docker Best Practices - MOSTLY PASS ✓
- ✓ Multi-stage build
- ✓ Layer caching optimization
- ✓ .dockerignore configured
- ✓ Health check configured
- ✓ Explicit base image tags (not :latest)
- ✓ Environment variable configuration
- ⚠ Large image size (acceptable for ML)

### Production Readiness - PARTIAL ✗
- ✗ Application fails to start (path issue)
- ✓ Health check configured
- ✓ Resource limits defined
- ✓ Logging configured
- ✓ Restart policy defined
- ⚠ Dev dependencies included

### ML-Specific Considerations
- ✓ GPU support configured
- ✓ CUDA/cuDNN versions matched
- ✓ Model volume separate from code
- ✓ Cache volume for model downloads
- ✓ Sufficient startup grace period (60s)

---

## Test Environment

### Host System
- **OS:** Linux 6.8.0-90-generic (Ubuntu)
- **Architecture:** x86_64
- **Docker:** 20.10+ (estimated)
- **Docker Compose:** v5.0.0

### GPU Hardware
- **GPU:** NVIDIA GeForce RTX 4070 Ti
- **VRAM:** 12282 MiB
- **Driver:** 590.44.01
- **CUDA:** 13.1

### Container Environment
- **Base OS:** Ubuntu 22.04
- **Python:** 3.11
- **PyTorch:** 2.4.0
- **CUDA:** 12.4
- **cuDNN:** 9

### Model Assets
- **Model File:** outputs/checkpoints/fold_0/best_model.pt
- **Model Size:** 94.9 MB
- **Model Type:** ResNet50 (67 classes)

---

## Recommendations Summary

### IMMEDIATE (BLOCKER)
1. **Fix path validation** in model_service.py to accept `/app/models/` OR
2. **Update docker-compose.yml** to mount at `/app/outputs/checkpoints/fold_0/`

### HIGH PRIORITY
3. **Reduce image size** by removing dev dependencies from production
4. **Create separate dev/prod Dockerfiles** or targets
5. **Remove debugpy port** from production deployment

### MEDIUM PRIORITY
6. Add integration tests for Docker deployment
7. Document Docker deployment in README
8. Add smoke tests in Dockerfile (currently only GPU check)
9. Implement model health check (verify model loads successfully)

### LOW PRIORITY
10. Remove obsolete version attribute from compose files
11. Add Docker build caching strategy documentation
12. Consider alternative base images for size optimization

---

## Conclusion

### BUILD STATUS: PASS ✓
Docker image builds successfully with all required dependencies. Multi-stage build properly implemented. Image size large but acceptable for ML applications.

### CONFIGURATION STATUS: PASS ✓
Docker Compose configuration valid with proper GPU support, resource limits, health checks, and security settings.

### RUNTIME STATUS: FAIL ✗
Application fails to start due to hardcoded path validation rejecting Docker's mounted model path. This is a **CRITICAL BLOCKER** preventing deployment.

### OVERALL ASSESSMENT: BLOCKED

Docker infrastructure well-designed but application code has hardcoded assumptions that conflict with Docker's file structure. Requires code fix or configuration change.

### Next Steps
1. Apply workaround: Update volume mount and checkpoint path
2. Test with corrected configuration
3. Verify all endpoints operational
4. Run GPU inference test
5. Update documentation with Docker deployment guide

---

## Unresolved Questions

1. Should path validation be removed entirely for containerized deployments?
2. Is there a use case for the hardcoded `outputs/checkpoints/` validation?
3. Should we create separate prod/dev Docker images with different dependencies?
4. What is acceptable image size limit for this ML application in production?
5. Should GPU support be optional or required for the Docker deployment?

---

**Report Generated:** 2025-12-23 15:02:00 UTC
**Test Duration:** ~15 minutes
**Report Path:** `/home/minh-ub/projects/multi-label-classification/plans/reports/tester-251223-docker-deployment-validation.md`
