# Docker Best Practices for ML/AI Services (PyTorch + FastAPI) - 2025

**Research Date**: 2025-12-22
**Focus**: Production deployment patterns for ML services using PyTorch and FastAPI

---

## 1. Multi-Stage Docker Builds

**Why**: Separate build environment (compilers, dev tools) from runtime environment → smaller images, faster deployment.

**Pattern**:
```dockerfile
# Stage 1: Builder
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Benefits**:
- Eliminates build artifacts from final image
- Runtime images lack compilers, reducing attack surface
- Can reduce image size from 800MB to 15-30MB (distroless/static)

---

## 2. Base Image Selection

### Official PyTorch Images (Recommended)

**Available variants**:
- `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` - Production (shared CUDA libs only)
- `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` - Build stage (includes toolchain, headers, static libs)

**Current CUDA Support** (2025):
- CUDA 11.8, 12.1, 12.4 with cuDNN 9
- PyTorch 2.7.0 supports CUDA 12.8 (+cu128 suffix) for Blackwell architecture (RTX 50xx)

**Tag Format**: `pytorch/pytorch:<pytorch_ver>-cuda<cuda_ver>-cudnn<cudnn_ver>-<variant>`

### Alternative Base Images

| Base | Size | Use Case |
|------|------|----------|
| `ubuntu:22.04` | ~77MB | Custom builds, full control |
| `python:3.11-slim` | ~45MB | CPU-only workloads |
| `gcr.io/distroless/python3` | ~50MB | Minimal attack surface, no shell |
| `alpine:3.19` | ~5MB | Extreme size optimization (compatibility issues with PyTorch) |

**Best Practice**: Use official PyTorch images unless specific customization required. Pin exact versions (no `latest` tag).

---

## 3. GPU Support in Docker

### Requirements

1. **Host**: NVIDIA Container Toolkit installed
2. **Verify**: `nvidia-smi` shows GPU(s)
3. **Docker run**: `docker run --gpus all <image>`

### Dockerfile Pattern

```dockerfile
FROM nvidia/cuda:12.4.0-cudnn9-runtime-ubuntu22.04
# OR use official PyTorch image (already includes CUDA)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Verify GPU access
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Version Locking (Critical)

```dockerfile
# Lock CUDA, cuDNN, PyTorch versions - reproducible builds
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
RUN pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**Why**: Prevents breaking changes from automatic updates.

---

## 4. Dependency Management

### Pip Caching & Layer Optimization

```dockerfile
# GOOD: Leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# BAD: Changes to code invalidate dependency layer
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
```

### Pre-download Models (Avoid Cold Starts)

```dockerfile
# Cache Hugging Face models in image
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('bert-base-uncased')"
```

### Requirements.txt Best Practices

```txt
# Pin exact versions for reproducibility
torch==2.4.0
fastapi==0.115.0
uvicorn[standard]==0.32.0

# Use --no-cache-dir to reduce layer size
# Clean package manager cache after install
```

### Dockerfile Example

```dockerfile
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

---

## 5. Model Artifact Handling

### Strategy Comparison

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| **COPY into image** | Simple, self-contained | Huge images (GB+), slow builds, tight coupling | Small models (<100MB), prototypes |
| **Volume mounts** | Decoupled, fast updates | Requires external storage setup | Development, dynamic model swapping |
| **Cloud Storage FUSE** | Near-instant startup, elastic | Cloud dependency | Production (GCS, S3, Azure Blob) |
| **Init container download** | Flexible, version control | Slower pod startup | Kubernetes, versioned models |

### Recommended Pattern (Production)

```dockerfile
# DON'T: Embed large models in image
# COPY models/bert-large.bin /app/models/  # Creates 3GB+ layer

# DO: Mount models at runtime
# docker run -v /host/models:/app/models <image>
```

### Volume Structure

```bash
/app          # Application code (in image)
/data         # Datasets (volume mount)
/models       # Model weights (volume mount or cloud storage)
/checkpoints  # Training artifacts (persistent volume)
```

### docker-compose.yml Example

```yaml
services:
  ml-api:
    image: ml-service:latest
    volumes:
      - ./models:/app/models:ro  # Read-only model weights
      - model-cache:/app/.cache  # Persistent cache
    environment:
      - MODEL_PATH=/app/models/model.pth
volumes:
  model-cache:
```

---

## 6. Production Optimizations

### Image Size Reduction

**Multi-stage + Distroless**:
```dockerfile
FROM python:3.11 AS builder
RUN pip install --user fastapi uvicorn torch

FROM gcr.io/distroless/python3
COPY --from=builder /root/.local /root/.local
COPY app/ /app/
ENV PATH=/root/.local/bin:$PATH
WORKDIR /app
CMD ["uvicorn", "main:app"]
```

**Size Benchmarks (2025)**:
- Under 100MB: Excellent
- Under 50MB: Elite
- Under 20MB: Possible for Go/Rust services

### Security Best Practices

```dockerfile
# 1. Non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# 2. Minimal base (reduces CVEs)
FROM gcr.io/distroless/python3-debian12

# 3. Pin versions (no latest)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# 4. Scan for vulnerabilities
# docker scan <image>
# trivy image <image>

# 5. Read-only filesystem (where possible)
# docker run --read-only <image>
```

**Docker Hardened Images (DHI)**: May 2025 - official hardened images with near-zero CVEs, 95% smaller, now free/open source.

### Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

---

## 7. FastAPI Deployment

### Production Stack (2025 Standard)

**Gunicorn + Uvicorn Workers** (recommended):
```dockerfile
CMD ["gunicorn", "main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30"]
```

**Single Uvicorn** (Kubernetes):
```dockerfile
# K8s handles multi-pod scaling, avoid duplicate process managers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Worker Configuration

```python
# Recommended formula
workers = (2 * CPU_cores) + 1

# For ML workloads (GPU-bound)
workers = 1-2  # Avoid GPU contention
```

### Full Production Dockerfile

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Application code
COPY --chown=appuser:appuser . .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/model.pth \
    PATH=/home/appuser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run with Gunicorn + Uvicorn workers
CMD ["gunicorn", "main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

### Graceful Shutdown Pattern

```python
# main.py
import signal
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    model = load_model()
    yield
    # Shutdown: Cleanup
    model.cleanup()

app = FastAPI(lifespan=lifespan)

# Handle SIGTERM for graceful pod termination
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
```

---

## Key Takeaways

1. **Multi-stage builds** are mandatory for production ML images (50-95% size reduction)
2. **Official PyTorch images** preferred over custom builds (use `-runtime` variant for final stage)
3. **Volume mounts** for models >100MB; cloud storage FUSE for production scale
4. **Gunicorn + Uvicorn workers** for FastAPI (except Kubernetes where single process preferred)
5. **Pin all versions** (CUDA, PyTorch, Python packages) - never use `latest`
6. **Non-root user** + distroless base = minimal attack surface
7. **Pre-cache models** in image or init container to avoid cold start delays

---

## Sources

- [Docker Multi-Stage Builds for Python Developers](https://collabnix.com/docker-multi-stage-builds-for-python-developers-a-complete-guide/)
- [Optimizing Docker Setup for PyTorch Training with CUDA 12.8](https://www.runpod.io/articles/guides/docker-setup-pytorch-cuda-12-8-python-3-11)
- [PyTorch Official Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [FastAPI in Containers - Official Documentation](https://fastapi.tiangolo.com/deployment/docker/)
- [The Definitive Guide to FastAPI Production Deployment with Docker (2025)](https://blog.greeden.me/en/2025/09/02/the-definitive-guide-to-fastapi-production-deployment-with-dockeryour-one-stop-reference-for-uvicorn-gunicorn-nginx-https-health-checks-and-observability-2025-edition/)
- [Scalable AI Storage: Guide to Model Artifact Strategies](https://cloud.google.com/blog/topics/developers-practitioners/scalable-ai-starts-with-storage-guide-to-model-artifact-strategies)
- [Docker Security in 2025: Best Practices](https://cloudnativenow.com/topics/cloudnativedevelopment/docker/docker-security-in-2025-best-practices-to-protect-your-containers-from-cyberthreats/)
- [Reducing Docker Image Sizes to 5MB: 2025 Multi-Stage Build Tricks](https://markaicode.com/reducing-docker-image-sizes-multistage-builds-2025/)
