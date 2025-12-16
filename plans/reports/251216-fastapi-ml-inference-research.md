# FastAPI Best Practices for ML Inference Endpoints (2025)

## 1. Image Upload Handling

**Multipart Form-Data (Recommended)**
- Use `UploadFile` for direct file uploads from HTML forms or clients
- Stream files to disk with `shutil.copyfileobj` to avoid memory overhead
- Validate file types and sizes before processing
- Mark form fields explicitly with `Form()` when mixing text + files

```python
from fastapi import FastAPI, File, UploadFile, Form
import shutil

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model_name: str = Form(...)
):
    with open("temp.jpg", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    # process image
    return {"prediction": "..."}
```

**Base64 (JSON APIs)**
- Use for small files only; memory-intensive for large images
- Preferred for pure JSON APIs where HTTP form data is unavailable
- Avoid for production high-throughput inference

## 2. Model Loading Patterns

**Singleton Pattern with Dependency Injection** (Recommended)
- Load models once at startup, reuse across requests
- Use `functools.lru_cache` for simple singleton caching
- FastAPI's lifespan context managers for resource cleanup

```python
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
import torch

class ModelManager:
    _instance = None

    @staticmethod
    async def get_instance():
        if ModelManager._instance is None:
            ModelManager._instance = torch.load("model.pt")
        return ModelManager._instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await ModelManager.get_instance()
    yield
    # Cleanup

app = FastAPI(lifespan=lifespan)

async def get_model(model = Depends(ModelManager.get_instance)):
    return model
```

## 3. Inference Optimization

**Async Processing**
- Use `async def` endpoints for non-blocking I/O; outperforms sync by 30% on concurrent loads
- Offload CPU-heavy inference to background workers (Celery, RQ)

**Batching**
- Micro-batch requests queued over 10-50ms windows
- Scales to 1M+ predictions/hour with GPU-aware orchestration
- Use `asyncio.gather()` for concurrent batch processing

**Caching**
- In-memory `async_lru` for repeated predictions on same input
- Redis for distributed caching across instances
- Achieves 3x faster responses under peak load

## 4. Response Formats

**Standard JSON Structure**
```python
from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    predictions: List[float]
    confidence_scores: List[float]
    class_labels: List[str]
    metadata: dict = {
        "model_version": "1.0",
        "inference_time_ms": 45.2,
        "batch_size": 1
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile):
    # Return structured response
    return PredictionResponse(
        predictions=[0.92, 0.08],
        confidence_scores=[0.92, 0.08],
        class_labels=["cat", "dog"],
        metadata={...}
    )
```

- Use response_model for automatic validation & OpenAPI documentation
- Include metadata: model version, inference time, batch size
- Avoid exposing internal model details

## 5. Error Handling

**Custom Exception Handlers**
```python
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid input", "errors": exc.errors()}
    )

class ModelInferenceError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail)
```

**Input Validation**
- Pydantic auto-validates request schemas (type, size constraints)
- Validate image dimensions, formats, file sizes
- Return 400 for client errors, 500 for server/model errors

**Timeout Handling**
```python
import asyncio
try:
    result = await asyncio.wait_for(model.predict(data), timeout=30.0)
except asyncio.TimeoutError:
    raise HTTPException(status_code=504, detail="Inference timeout")
```

## 6. Performance Metrics

**OpenTelemetry + Prometheus**
- Instrument with `prometheus-fastapi-instrumentator` for auto-metrics
- Export traces to Tempo, metrics to Prometheus, logs to Loki
- Key metrics: request_duration_seconds, inference_duration_ms, throughput, error_rate

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

**Native Pydantic Metadata**
- Include inference_time_ms in response metadata
- Track batch_size per request
- Monitor model latency per class/label

## 7. Production Considerations

**CORS**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://domain.com"],  # Restrict in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    allow_credentials=True,
)
```

**Health Checks**
```python
@app.get("/health/live")
async def liveness():
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    model = await ModelManager.get_instance()
    return {"status": "ready", "model_loaded": model is not None}
```

**Versioning**
- Include model_version in response metadata
- Support multiple model versions via dependency injection
- Use URL versioning (/v1/predict, /v2/predict) for client compatibility

**Deployment**
- Uvicorn + Gunicorn: 2-4x CPU cores as workers
- Docker: multi-stage builds, slim Python base images
- Nginx reverse proxy for load balancing, SSL/TLS
- Graceful shutdown handlers for cleanup

**Database & Async Pools**
- Connection pooling: pool_size=20, max_overflow=10
- Async drivers for non-blocking database queries
- Select specific columns, avoid SELECT *

## Key Takeaways

1. Multipart form-data for image uploads; base64 only for small JSON payloads
2. Singleton pattern with lifespan context managers for model loading
3. Async endpoints + batching + caching for 3x throughput gains
4. Structured Pydantic response models with confidence scores & metadata
5. OpenTelemetry + Prometheus for production observability
6. Graceful error handling with custom exceptions and timeouts
7. Health checks, CORS, versioning mandatory for production deployments

---

## Sources

- [FastAPI ML Inference Best Practices](https://medium.com/@Nexumo_/12-fastapi-blueprints-for-sub-50-ms-ai-inference-apis-7efe0ee3772a)
- [FastAPI for Data Science](https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/)
- [Request Files - FastAPI](https://fastapi.tiangolo.com/tutorial/request-files/)
- [File Upload Best Practices](https://betterstack.com/community/guides/scaling-python/uploading-files-using-fastapi/)
- [Dependency Injection Patterns](https://vladiliescu.net/better-dependency-injection-in-fastapi/)
- [FastAPI Performance Tuning](https://blog.greeden.me/en/2025/12/09/complete-fastapi-performance-tuning-guide-build-scalable-apis-with-async-i-o-connection-pools-caching-and-rate-limiting/)
- [Scaling FastAPI ML Servers](https://medium.com/@connect.hashblock/how-i-scaled-a-fastapi-ml-inference-server-to-handle-1m-predictions-per-hour-6c2424aa4faf)
- [FastAPI Error Handling](https://betterstack.com/community/guides/scaling-python/error-handling-fastapi/)
- [OpenTelemetry FastAPI Integration](https://last9.io/blog/integrating-opentelemetry-with-fastapi/)
- [Production Deployment Guide](https://medium.com/@ramanbazhanau/preparing-fastapi-for-production-a-comprehensive-guide-d167e693aa2b)
- [CORS Configuration](https://fastapi.tiangolo.com/tutorial/cors/)
- [Health Checks & Monitoring](https://dev.to/lisan_al_gaib/building-a-health-check-microservice-with-fastapi-26jo)

