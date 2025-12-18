# Phase 04: Response Formatting & Metrics

## Context

- [Main Plan](./plan.md)
- [Phase 03: Inference Endpoint](./phase-03-inference-endpoint.md)
- Scout: [Model Inference Patterns](../reports/251216-scout-model-inference-patterns.md)
- Test Metrics: `outputs/test_results/fold_0/val/test_metrics.json`

## Overview

| Field | Value |
|-------|-------|
| Date | 2025-12-16 |
| Priority | Medium |
| Status | ✅ Complete (Code Review: 2025-12-18) |
| Est. Time | 1-2 hours |

## Key Insights

1. **Model metrics from test** available at checkpoint path
2. **Metrics categories**: accuracy, balanced_accuracy, macro/weighted F1, top-k accuracy
3. **Speed metrics**: avg_time_per_sample_ms, throughput_samples_per_sec
4. **Expose model performance** in /model/info endpoint
5. **Version info** via response headers or model_info field

## Requirements

- Model info endpoint with performance metrics
- Include model version in responses
- Optional: batch prediction endpoint
- OpenAPI documentation enhancement
- CORS configuration for frontend access

## Architecture

```python
# api/routers/model.py
@router.get("/model/info")
async def model_info() -> ModelInfoResponse:
    """Return model metadata and performance metrics."""
    ...

@router.get("/model/classes")
async def list_classes() -> ClassListResponse:
    """Return list of supported class names."""
    ...
```

## Related Code Files

| File | Purpose |
|------|---------|
| `outputs/test_results/fold_0/val/test_metrics.json` | Pre-computed test metrics |
| `src/metrics.py` | MetricsCalculator class |

## Implementation Steps

### 1. Create Model Info Response Schema

```python
# api/models.py (additions)

class PerformanceMetrics(BaseModel):
    """Model performance metrics from validation set."""
    accuracy: float = Field(..., description="Overall accuracy")
    balanced_accuracy: float = Field(..., description="Balanced accuracy")
    f1_macro: float = Field(..., description="Macro-averaged F1 score")
    f1_weighted: float = Field(..., description="Weighted F1 score")
    top_3_accuracy: float = Field(..., description="Top-3 accuracy")
    top_5_accuracy: float = Field(..., description="Top-5 accuracy")

class SpeedMetrics(BaseModel):
    """Model inference speed metrics."""
    avg_time_per_sample_ms: float
    throughput_samples_per_sec: float
    device: str

class ModelInfoResponse(BaseModel):
    """Model information and performance metrics."""
    model_name: str
    num_classes: int
    image_size: int
    checkpoint_path: str
    device: str
    is_loaded: bool
    performance_metrics: Optional[PerformanceMetrics] = None
    speed_metrics: Optional[SpeedMetrics] = None
    class_names: List[str]

class ClassListResponse(BaseModel):
    """List of supported classes."""
    num_classes: int
    classes: List[dict]  # [{id: 0, name: "Abyssinian"}, ...]
```

### 2. Create Model Router (api/routers/model.py)

```python
import json
from pathlib import Path
from fastapi import APIRouter, Depends

from ..models import ModelInfoResponse, ClassListResponse, PerformanceMetrics, SpeedMetrics
from ..services.model_service import ModelManager
from ..config import Settings
from ..dependencies import get_settings

router = APIRouter()

async def get_model_manager() -> ModelManager:
    return await ModelManager.get_instance()

def load_test_metrics(checkpoint_path: str) -> dict:
    """Load pre-computed test metrics if available."""
    # Derive metrics path from checkpoint path
    # outputs/checkpoints/fold_0/best_model.pt -> outputs/test_results/fold_0/val/test_metrics.json
    try:
        cp_path = Path(checkpoint_path)
        fold = cp_path.parent.name  # e.g., "fold_0"
        metrics_path = Path("outputs/test_results") / fold / "val" / "test_metrics.json"

        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(
    model_manager: ModelManager = Depends(get_model_manager),
    settings: Settings = Depends(get_settings)
) -> ModelInfoResponse:
    """
    Get model information and performance metrics.

    Returns model metadata, validation performance, and inference speed.
    """
    # Load test metrics
    test_data = load_test_metrics(settings.checkpoint_path)

    performance_metrics = None
    speed_metrics = None

    if test_data.get("metrics"):
        m = test_data["metrics"]
        performance_metrics = PerformanceMetrics(
            accuracy=m.get("accuracy", 0.0),
            balanced_accuracy=m.get("balanced_accuracy", 0.0),
            f1_macro=m.get("f1_macro", 0.0),
            f1_weighted=m.get("f1_weighted", 0.0),
            top_3_accuracy=m.get("top_3_accuracy", 0.0),
            top_5_accuracy=m.get("top_5_accuracy", 0.0)
        )

    if test_data.get("speed_metrics"):
        s = test_data["speed_metrics"]
        speed_metrics = SpeedMetrics(
            avg_time_per_sample_ms=s.get("avg_time_per_sample_ms", 0.0),
            throughput_samples_per_sec=s.get("throughput_samples_per_sec", 0.0),
            device=s.get("device", str(model_manager.device))
        )

    return ModelInfoResponse(
        model_name=model_manager.model_name,
        num_classes=len(model_manager.class_names),
        image_size=settings.image_size,
        checkpoint_path=settings.checkpoint_path,
        device=str(model_manager.device),
        is_loaded=model_manager.is_loaded,
        performance_metrics=performance_metrics,
        speed_metrics=speed_metrics,
        class_names=model_manager.class_names
    )

@router.get("/model/classes", response_model=ClassListResponse)
async def list_classes(
    model_manager: ModelManager = Depends(get_model_manager)
) -> ClassListResponse:
    """
    List all supported cat breed classes.

    Returns class IDs and names for reference.
    """
    classes = [
        {"id": i, "name": name}
        for i, name in enumerate(model_manager.class_names)
    ]

    return ClassListResponse(
        num_classes=len(classes),
        classes=classes
    )
```

### 3. Add CORS Middleware

```python
# api/main.py
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(...)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 4. Enhance OpenAPI Documentation

```python
# api/main.py
app = FastAPI(
    title="Cat Breeds Classification API",
    description="""
## Cat Breed Image Classification API

Predict cat breeds from uploaded images using deep learning.

### Features
- **67 cat breeds** supported
- **Top-5 predictions** with confidence scores
- **Fast inference** (~0.9ms/image on GPU)
- **Image validation** (JPEG, PNG, WebP)

### Model Information
- Architecture: ResNet50 / EfficientNet (TIMM)
- Input size: 224x224
- Normalization: ImageNet statistics

### Usage
1. Upload an image via POST /api/v1/predict
2. Receive predicted breed with confidence scores
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)
```

### 5. Add Version Header Middleware

```python
# api/middleware.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class VersionHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Model-Version"] = "resnet50-fold0"
        return response

# api/main.py
from .middleware import VersionHeaderMiddleware
app.add_middleware(VersionHeaderMiddleware)
```

### 6. Update Router Includes

```python
# api/main.py
from .routers import health, predict, model

app.include_router(health.router, prefix="", tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(model.router, prefix="/api/v1", tags=["Model"])
```

## Todo List

- [x] Add PerformanceMetrics, SpeedMetrics, ModelInfoResponse to models.py
- [x] Create api/routers/model.py
- [x] Add CORS middleware to main.py
- [x] Enhance OpenAPI documentation
- [x] Create api/middleware.py with version headers
- [x] Update router includes in main.py
- [ ] Test /model/info endpoint (Phase 05)
- [ ] Test /model/classes endpoint (Phase 05)
- [ ] Verify CORS headers in response (Phase 05)

## Success Criteria

1. GET /api/v1/model/info returns performance metrics
2. GET /api/v1/model/classes returns 67 breeds
3. CORS headers present in responses
4. OpenAPI docs accessible at /docs
5. X-API-Version header in all responses

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Test metrics file missing | Return null for metrics, not error |
| CORS too permissive | Document production restriction |
| Large class list response | List is small (67 items), OK |

## Security Considerations

- CORS allow_origins should be restricted in production
- No sensitive data in model info
- Version headers expose minimal info

## Code Review Results

**Report:** [code-reviewer-2025-12-18-phase04-response-metrics.md](../reports/code-reviewer-2025-12-18-phase04-response-metrics.md)

**Status:** ✅ APPROVED - Production ready

**Summary:**
- Critical Issues: 0
- Security Vulnerabilities: 0
- Code Quality: HIGH
- Files: 4 changed (370 lines)

**Findings:**
- 2 Medium: Duplicate get_model_manager(), NaN in JSON metrics
- 3 Low: Hardcoded versions, magic strings, broad exceptions
- All issues non-blocking

**Recommendations:** Address duplicate function, fix NaN in test metrics (training code)

## Next Steps

After Phase 04:
- [Phase 05: Testing & Validation](./phase-05-testing-validation.md)
