# Phase 03: Inference Endpoint Implementation

## Context

- [Main Plan](./plan.md)
- [Phase 01: Core API](./phase-01-core-api-model-loading.md)
- [Phase 02: Image Validation](./phase-02-image-validation-preprocessing.md)
- Scout: [Model Inference Patterns](../reports/251216-scout-model-inference-patterns.md)

## Overview

| Field | Value |
|-------|-------|
| Date | 2025-12-16 |
| Priority | High |
| Status | Pending |
| Est. Time | 2 hours |

## Key Insights

1. **torch.no_grad()** context required for inference (no gradient tracking)
2. **torch.softmax(outputs, dim=1)** converts logits to probabilities
3. **Top-K predictions** via `torch.topk(probs, k=5)`
4. **Inference timing** with `time.perf_counter()` + CUDA synchronization
5. **Warmup batch** recommended before benchmarking

## Requirements

- POST /predict endpoint accepting image file
- Return top predicted class with confidence
- Return top-5 predictions with confidences
- Include inference time in response
- Async endpoint for non-blocking I/O

## Architecture

```python
# api/routers/predict.py
@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    image_service: ImageService = Depends(get_image_service),
    model_manager: ModelManager = Depends(get_model_manager)
) -> PredictionResponse:
    # 1. Validate and preprocess
    tensor, metadata = await image_service.validate_and_preprocess(file)

    # 2. Run inference
    start_time = time.perf_counter()
    probs, _ = model_manager.predict(tensor)
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    # 3. Get top-5 predictions
    top5 = get_top_k_predictions(probs, model_manager.class_names, k=5)

    # 4. Return response
    return PredictionResponse(...)
```

## Related Code Files

| File | Purpose |
|------|---------|
| `scripts/test.py` | evaluate_model(), benchmark_inference_speed() |
| `src/metrics.py` | MetricsCalculator |
| `src/trainer.py` | validate_epoch() inference pattern |

## Implementation Steps

### 1. Create Pydantic Models (api/models.py)

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionItem(BaseModel):
    """Single prediction with class name and confidence."""
    rank: int = Field(..., ge=1, le=67, description="Prediction rank (1-67)")
    class_name: str = Field(..., description="Predicted cat breed name")
    class_id: int = Field(..., ge=0, le=66, description="Class index (0-66)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

class ImageMetadata(BaseModel):
    """Metadata about the uploaded image."""
    original_width: int
    original_height: int
    format: Optional[str] = None
    mode: str
    file_size_bytes: int

class PredictionResponse(BaseModel):
    """Response from the prediction endpoint."""
    # Top prediction
    predicted_class: str = Field(..., description="Top predicted breed")
    confidence: float = Field(..., description="Confidence of top prediction")

    # Top-5 predictions
    top_5_predictions: List[PredictionItem] = Field(
        ..., description="Top 5 predictions with confidence scores"
    )

    # Performance
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")

    # Metadata
    image_metadata: ImageMetadata
    model_info: dict = Field(
        default_factory=dict,
        description="Model information (name, device)"
    )

class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str
    errors: Optional[List[dict]] = None
```

### 2. Create Inference Service (api/services/inference_service.py)

```python
import time
import numpy as np
import torch
from typing import List, Tuple

from ..models import PredictionItem

class InferenceService:
    """Service for running model inference."""

    @staticmethod
    def get_top_k_predictions(
        probs: np.ndarray,
        class_names: List[str],
        k: int = 5
    ) -> List[PredictionItem]:
        """
        Get top-K predictions from probability array.

        Args:
            probs: Probability array of shape (1, num_classes)
            class_names: List of class names
            k: Number of top predictions to return

        Returns:
            List of PredictionItem
        """
        # probs shape: (1, 67) -> squeeze to (67,)
        probs_1d = probs.squeeze()

        # Get top-k indices
        top_k_indices = np.argsort(probs_1d)[::-1][:k]

        predictions = []
        for rank, idx in enumerate(top_k_indices, start=1):
            predictions.append(PredictionItem(
                rank=rank,
                class_name=class_names[idx],
                class_id=int(idx),
                confidence=float(probs_1d[idx])
            ))

        return predictions

    @staticmethod
    def synchronize_device(device: torch.device):
        """Synchronize CUDA device for accurate timing."""
        if device.type == "cuda":
            torch.cuda.synchronize()
```

### 3. Create Predict Router (api/routers/predict.py)

```python
import time
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException

from ..models import PredictionResponse, ImageMetadata, ErrorResponse
from ..services.model_service import ModelManager
from ..services.image_service import ImageService
from ..services.inference_service import InferenceService
from ..dependencies import get_image_service

router = APIRouter()

async def get_model_manager() -> ModelManager:
    """Dependency to get model manager instance."""
    manager = await ModelManager.get_instance()
    if not manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    return manager

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        413: {"model": ErrorResponse, "description": "Image too large"},
        503: {"model": ErrorResponse, "description": "Model not ready"}
    }
)
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    image_service: ImageService = Depends(get_image_service),
    model_manager: ModelManager = Depends(get_model_manager)
) -> PredictionResponse:
    """
    Predict cat breed from uploaded image.

    Returns top prediction with confidence score and top-5 predictions.
    """
    # 1. Validate and preprocess image
    tensor, metadata = await image_service.validate_and_preprocess(file)

    # 2. Run inference with timing
    start_time = time.perf_counter()

    probs, _ = model_manager.predict(tensor)

    # Synchronize for accurate timing
    InferenceService.synchronize_device(model_manager.device)

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    # 3. Get top-5 predictions
    top_5 = InferenceService.get_top_k_predictions(
        probs=probs,
        class_names=model_manager.class_names,
        k=5
    )

    # 4. Build response
    return PredictionResponse(
        predicted_class=top_5[0].class_name,
        confidence=top_5[0].confidence,
        top_5_predictions=top_5,
        inference_time_ms=round(inference_time_ms, 3),
        image_metadata=ImageMetadata(**metadata),
        model_info={
            "model_name": model_manager.model_name,
            "device": str(model_manager.device),
            "num_classes": len(model_manager.class_names)
        }
    )
```

### 4. Update Main App Router Includes

```python
# api/main.py
from .routers import health, predict

# ... lifespan and app creation ...

app.include_router(health.router, prefix="", tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
```

### 5. Add Model Name to ModelManager

Update `api/services/model_service.py`:

```python
class ModelManager:
    def __init__(self):
        # ... existing ...
        self._model_name: str = ""

    async def load_model(self, ...):
        # ... existing ...
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name
```

## Todo List

- [ ] Create api/models.py with Pydantic schemas
- [ ] Create api/services/inference_service.py
- [ ] Create api/routers/predict.py
- [ ] Update api/main.py with predict router
- [ ] Add model_name property to ModelManager
- [ ] Test with sample cat images
- [ ] Verify response matches expected schema
- [ ] Test error cases (invalid image, model not loaded)

## Success Criteria

1. POST /api/v1/predict accepts image file
2. Response includes predicted_class, confidence, top_5_predictions
3. Inference time <50ms on GPU, <500ms on CPU
4. Top-5 class names match expected breed names
5. Confidence scores sum to ~1.0 (due to softmax)
6. Proper 400/413/503 error responses

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Model not loaded at request time | 503 error with clear message |
| Inference timeout | Default timeout is 30s, should be sufficient |
| Wrong device type in timing | CUDA sync handles this |
| Class name mismatch | Use same breed list as training |

## Security Considerations

- No user input in model inference (tensor only)
- Response doesn't expose internal model weights
- Timing side-channel is acceptable for this use case

## Next Steps

After Phase 03:
- [Phase 04: Response Formatting & Metrics](./phase-04-response-metrics.md)
