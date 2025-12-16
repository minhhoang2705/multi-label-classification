# Cat Breeds Classification API - Phase 01: Core API & Model Loading

## Overview

Phase 01 implements the foundation of the Cat Breeds Classification inference API using FastAPI. This phase establishes core infrastructure for model loading, health checks, and security hardening.

**Status:** Complete
**Version:** 1.0.0
**Date:** Phase 01 Complete

---

## Architecture

### Core Components

1. **FastAPI Application** (`api/main.py`)
   - Application lifecycle management (startup/shutdown)
   - CORS middleware configuration
   - Route registration

2. **Configuration** (`api/config.py`)
   - Environment-based settings using Pydantic
   - Model, server, and device configuration
   - CORS policy management

3. **ModelManager Service** (`api/services/model_service.py`)
   - Singleton pattern for model lifecycle
   - Checkpoint loading with path validation
   - Device auto-detection (CUDA/MPS/CPU)
   - Inference interface for Phase 02

4. **Health Router** (`api/routers/health.py`)
   - Liveness probe (`/health/live`)
   - Readiness probe (`/health/ready`)

---

## Configuration

### Environment Variables

All settings support environment variable overrides via `API_` prefix:

```bash
# Model Configuration
API_CHECKPOINT_PATH=outputs/checkpoints/fold_0/best_model.pt
API_MODEL_NAME=resnet50
API_NUM_CLASSES=67
API_IMAGE_SIZE=224

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Device Configuration (auto, cuda, mps, cpu)
API_DEVICE=auto

# CORS Configuration
API_CORS_ORIGINS=["http://localhost:3000","http://localhost:8080"]
API_CORS_ALLOW_CREDENTIALS=true

# API Metadata
API_API_TITLE="Cat Breeds Classification API"
API_API_VERSION=1.0.0
```

### Default Settings

```python
# Model
checkpoint_path = "outputs/checkpoints/fold_0/best_model.pt"
model_name = "resnet50"
num_classes = 67
image_size = 224

# Server
host = "0.0.0.0"
port = 8000

# Device
device = "auto"  # Auto-detects CUDA > MPS > CPU

# CORS
cors_origins = ["http://localhost:3000", "http://localhost:8080"]
cors_allow_credentials = True
```

---

## API Endpoints - Phase 01

### 1. Root Endpoint

**GET** `/`

Returns basic API information.

**Response:**
```json
{
  "message": "Cat Breeds Classification API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health/live"
}
```

### 2. Liveness Probe

**GET** `/health/live`

Simple health check indicating application is running.

**Response:**
```json
{
  "status": "alive"
}
```

**Use Case:** Kubernetes/container orchestration liveness probes.

### 3. Readiness Probe

**GET** `/health/ready`

Indicates if API is ready to serve requests (model loaded).

**Response:**
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_name": "resnet50",
  "device": "cuda",
  "num_classes": 67
}
```

**Status Values:**
- `"ready"` - Model loaded, API ready for inference
- `"not_ready"` - Model not yet loaded, API starting up

**Use Case:** Kubernetes/container orchestration readiness probes.

---

## API Startup Process

### 1. Application Initialization

```python
# api/main.py lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    manager = await ModelManager.get_instance()
    await manager.load_model(...)

    yield  # Application running

    # Shutdown: Cleanup
```

### 2. Model Loading Sequence

1. API starts
2. Lifespan context manager triggers startup
3. Singleton ModelManager instantiated
4. Model loaded from checkpoint (typically 5-15 seconds)
5. Device auto-detection completes
6. Class names loaded (67 cat breeds)
7. Model set to evaluation mode
8. API ready for requests

### 3. Monitoring Startup

Check readiness status:

```bash
curl http://localhost:8000/health/ready
```

---

## ModelManager Service

### Singleton Pattern

```python
# Get instance (creates on first call, returns same instance thereafter)
manager = await ModelManager.get_instance()
```

**Benefits:**
- Single model instance shared across all requests
- Efficient memory usage
- Consistent model behavior
- Thread-safe (async-aware)

### Device Auto-Detection

```python
device = manager._get_device("auto")
```

**Priority:**
1. CUDA (if available)
2. MPS (Apple Metal, if available)
3. CPU (fallback)

**Force Specific Device:**
```bash
API_DEVICE=cpu    # Force CPU
API_DEVICE=cuda   # Force CUDA
API_DEVICE=mps    # Force MPS
```

### Path Security

Checkpoint paths validated against path traversal attacks:

```python
await manager.load_model(checkpoint_path="outputs/checkpoints/fold_0/best_model.pt")
```

**Security Checks:**
- Path must be within `outputs/checkpoints/` directory
- File must exist
- Must be regular file (not directory)
- Absolute path resolution to prevent symlink attacks

**Raises:**
- `FileNotFoundError` - Checkpoint doesn't exist
- `ValueError` - Invalid path or outside allowed directory

### Class Names

67 cat breed names in alphabetical order:

```python
class_names = manager.class_names
# ["Abyssinian", "American Bobtail", ..., "York Chocolate"]
```

### Model Properties

```python
manager.is_loaded        # bool - Model loaded successfully
manager.device           # torch.device - Current device
manager.model_name       # str - Model architecture (resnet50)
manager.checkpoint_path  # str - Path to loaded checkpoint
manager.class_names      # List[str] - 67 breed names
```

---

## Security Features

### 1. Path Validation

- Prevents path traversal attacks (`../` sequences)
- Restricts to `outputs/checkpoints/` directory
- Validates file existence and type

### 2. CORS Configuration

- Restricted origins (default: localhost:3000/8080)
- Credentials allowed
- Methods: GET, POST
- Configurable via environment variables

### 3. Logging

- INFO level logs for startup/shutdown
- ERROR logs with tracebacks for failures
- Structured format: `timestamp - module - level - message`

### 4. No Anonymous Model Loading

- Model path fixed at configuration time
- No runtime path specification
- Prevents arbitrary file access

---

## Error Handling

### Model Loading Errors

**Missing Checkpoint:**
```
FileNotFoundError: Checkpoint not found: outputs/checkpoints/fold_0/best_model.pt
```

**Invalid Path:**
```
ValueError: Invalid checkpoint path. Must be within outputs/checkpoints/ directory.
```

**Corrupted Checkpoint:**
```
RuntimeError: Failed to load checkpoint state dict - incompatible format
```

### API Errors

All errors logged with full traceback for debugging.

---

## Testing

### Test Coverage

**Unit Tests:** `tests/test_api_phase01.py`

1. **Device Detection** (4 tests)
   - Auto detection
   - CUDA override
   - CPU override
   - MPS override

2. **Class Names** (3 tests)
   - Loading 67 breeds
   - Expected breeds present
   - Alphabetical order

3. **Singleton Pattern** (2 tests)
   - Instance reuse
   - State persistence

4. **ModelManager Properties** (2 tests)
   - Initial state
   - State after loading

5. **Model Loading** (4 tests)
   - Real checkpoint loading
   - File not found handling
   - Checkpoint structure compatibility
   - Load time performance

6. **Health Endpoints** (3 tests)
   - Root endpoint
   - Liveness probe
   - Readiness probe

7. **API Startup** (4 tests)
   - FastAPI creation
   - Router registration
   - CORS middleware configuration
   - TestClient creation

8. **Configuration** (3 tests)
   - Settings loaded
   - Default values
   - Model configuration

9. **Integration Tests** (3 tests)
   - Model loading on startup
   - Health endpoints after startup
   - All endpoints accessible

10. **Performance Tests** (1 test)
    - Model load time < 30 seconds

### Running Tests

```bash
# Run all Phase 01 tests
pytest tests/test_api_phase01.py -v

# Run specific test class
pytest tests/test_api_phase01.py::TestHealthEndpoints -v

# Run with detailed output
pytest tests/test_api_phase01.py -vv --tb=short
```

### Test Requirements

- Checkpoint file must exist at configured path
- Tests use CPU device to avoid CUDA requirements
- Model loads within 30 seconds on CPU
- All 67 class names loaded correctly

---

## Running the API

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run API (defaults to port 8000)
python -m uvicorn api.main:app --reload

# Custom configuration
API_PORT=5000 API_DEVICE=cpu python -m uvicorn api.main:app --reload
```

### Production Deployment

```bash
# Using Gunicorn with Uvicorn workers
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

# With custom settings
API_DEVICE=cuda API_CHECKPOINT_PATH=outputs/checkpoints/fold_0/best_model.pt \
  gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV API_DEVICE=auto

CMD ["gunicorn", "api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

### Health Check (Docker)

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health/live || exit 1
```

---

## API Documentation

Interactive API documentation available at:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Phase 01 Completion Checklist

- [x] FastAPI application setup
- [x] Configuration system (Pydantic Settings)
- [x] ModelManager singleton service
- [x] Device auto-detection
- [x] Checkpoint loading with path validation
- [x] Health endpoints (live/ready)
- [x] CORS middleware configuration
- [x] Structured logging
- [x] Comprehensive test suite (30+ tests)
- [x] Error handling and validation
- [x] Security hardening (path validation)

---

## Next Phases

### Phase 02: Image Validation & Preprocessing
**Status:** COMPLETE - See [`api-phase02.md`](./api-phase02.md)

- [x] Multi-layer image validation (5 stages)
- [x] Security: decompression bomb + pixel flood protection
- [x] ImageNet normalization pipeline
- [x] Type-safe metadata with TypedDict
- [x] Custom exception types
- [x] Dependency injection factories
- [x] 61 comprehensive tests

### Phase 03: Batch Inference
- `/predict/batch` endpoint
- Bulk image processing
- Result aggregation

### Phase 04: Metrics & Monitoring
- Prometheus metrics
- Inference latency tracking
- Model performance metrics

---

## Dependencies

**Core:**
- fastapi >= 0.104.1
- pydantic-settings >= 2.0.0
- uvicorn >= 0.24.0
- torch >= 2.0.0
- timm >= 0.9.0

**Development:**
- pytest >= 7.4.0
- pytest-asyncio >= 0.21.0
- httpx >= 0.25.0
- gunicorn >= 21.2.0

See `requirements.txt` for complete list.

---

## Troubleshooting

### Model Fails to Load

**Problem:** "Checkpoint not found"
```bash
# Verify checkpoint exists
ls -la outputs/checkpoints/fold_0/best_model.pt

# Use correct path
API_CHECKPOINT_PATH=/full/path/to/checkpoint.pt python -m uvicorn api.main:app
```

**Problem:** "Path outside allowed directory"
```bash
# Checkpoints must be in outputs/checkpoints/
# Invalid: /home/user/model.pt
# Valid: outputs/checkpoints/fold_0/best_model.pt
```

### Out of Memory

**Problem:** CUDA out of memory during model loading
```bash
# Use CPU for testing
API_DEVICE=cpu python -m uvicorn api.main:app

# Or reduce batch size in Phase 02
```

### Startup Timeout

**Problem:** Model takes > 30 seconds to load
```bash
# Normal for large models on CPU
# Consider using CUDA or larger GPU
# Adjust Docker healthcheck timeout in production

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s
```

### CORS Errors

**Problem:** Requests blocked by CORS
```bash
# Update allowed origins
API_CORS_ORIGINS='["http://localhost:3000","http://localhost:5000"]'
```

---

## Performance Characteristics

### Model Loading
- CUDA: ~5-10 seconds
- CPU: ~15-30 seconds (test: < 30s)
- MPS (Apple): ~8-15 seconds

### Memory Usage
- ResNet50: ~100 MB on GPU, ~250 MB on CPU
- Model + API overhead: ~300-500 MB total

### Inference (Phase 02)
- Single image: ~5-20 ms on CUDA, ~50-200 ms on CPU
- Batch (32): ~10-30 ms per image on CUDA

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [TIMM Documentation](https://timm.fast.ai/)
- [PyTorch Documentation](https://pytorch.org/)

---

**Last Updated:** Phase 01 Complete
**Maintainer:** Development Team
