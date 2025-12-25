# Phase 2: VLM Integration into Inference Pipeline

**Parent:** [plan.md](./plan.md) | **Depends:** [Phase 1](./phase-01-dataset-generation.md) (optional)
**Status:** Pending | **Priority:** High

## Overview

Integrate CLIP as post-classifier into existing FastAPI inference service with confidence-based invocation.

## Key Insights

- CLIP ViT-B/32: 120ms latency, ~500MB VRAM, zero-shot capable
- Current CNN latency: ~2ms (ConvNeXt-Base)
- Selective invocation reduces overhead: only 30-40% of predictions need VLM
- Priority breeds should always use VLM regardless of CNN confidence

## Requirements

1. Add CLIP model loading alongside CNN
2. Implement confidence threshold logic (0.75 default)
3. Create VLM service for post-classification
4. Extend `/predict` endpoint with optional VLM mode
5. Add priority breed detection

## Architecture

```
PredictRequest
    ↓
[ImageService] Preprocess
    ↓
[ModelManager] ConvNeXt Inference (~2ms)
    ↓
[ThresholdRouter] Confidence Check
    ├─ conf > 0.85 AND not priority breed → Return CNN result
    ├─ priority breed in top-3 → VLM path
    └─ 0.65 < conf < 0.85 → VLM path
    ↓
[VLMService] CLIP Inference (~120ms)
    ↓
[EnsembleService] Combine predictions
    ↓
PredictResponse
```

## Related Files

- `api/services/inference_service.py`
- `api/services/model_service.py`
- `api/routers/predict.py`
- `api/models.py`

## Implementation Steps

### Step 1: VLM Service Class

```python
# api/services/vlm_service.py

import torch
import numpy as np
from typing import List, Optional, Tuple
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)

PRIORITY_BREEDS = {
    "American Wirehair", "Burmilla", "Canadian Hairless",
    "Chinchilla", "Cymric", "Oriental Long Hair", "York Chocolate"
}

class VLMService:
    """CLIP-based post-classifier for uncertain predictions."""

    _instance: Optional["VLMService"] = None

    def __init__(self):
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None
        self._device: Optional[torch.device] = None
        self._class_embeddings: Optional[torch.Tensor] = None
        self._is_loaded: bool = False

    @classmethod
    async def get_instance(cls) -> "VLMService":
        if cls._instance is None:
            cls._instance = VLMService()
        return cls._instance

    async def load_model(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto"
    ) -> None:
        """Load CLIP model and precompute class embeddings."""
        # Device selection
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        logger.info(f"Loading CLIP model: {model_name}")
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

        self._is_loaded = True
        logger.info(f"CLIP loaded on {self._device}")

    def precompute_class_embeddings(self, class_names: List[str]) -> None:
        """Precompute text embeddings for all classes (cache for speed)."""
        prompts = [f"a photo of a {breed} cat" for breed in class_names]

        with torch.no_grad():
            inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            text_features = self._model.get_text_features(**inputs)
            self._class_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)

        logger.info(f"Precomputed embeddings for {len(class_names)} classes")

    def predict(
        self,
        image: np.ndarray,  # PIL Image or numpy array
        class_names: List[str],
        top_k: int = 5
    ) -> Tuple[List[int], np.ndarray]:
        """
        Run CLIP inference on image.

        Returns:
            Tuple of (top_k_indices, probabilities)
        """
        if not self._is_loaded:
            raise RuntimeError("VLM not loaded")

        # Ensure embeddings are computed
        if self._class_embeddings is None:
            self.precompute_class_embeddings(class_names)

        # Process image
        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            image_features = self._model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ self._class_embeddings.T).softmax(dim=-1)
            probs = similarity.cpu().numpy().squeeze()

        top_k_indices = np.argsort(probs)[::-1][:top_k]
        return top_k_indices.tolist(), probs

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @staticmethod
    def is_priority_breed(breed: str) -> bool:
        return breed in PRIORITY_BREEDS

    @staticmethod
    def should_invoke_vlm(
        cnn_confidence: float,
        top_predictions: List[str],
        high_threshold: float = 0.85,
        low_threshold: float = 0.65
    ) -> bool:
        """Determine if VLM should be invoked."""
        # Always invoke for priority breeds
        for breed in top_predictions[:3]:
            if VLMService.is_priority_breed(breed):
                return True

        # Invoke for uncertain predictions
        return cnn_confidence < high_threshold
```

### Step 2: Ensemble Service

```python
# api/services/ensemble_service.py

import numpy as np
from typing import List, Tuple

class EnsembleService:
    """Combine CNN and VLM predictions."""

    @staticmethod
    def weighted_ensemble(
        cnn_probs: np.ndarray,
        vlm_probs: np.ndarray,
        cnn_weight: float = 0.6,
        vlm_weight: float = 0.4
    ) -> np.ndarray:
        """Weighted average of CNN and VLM probabilities."""
        return cnn_weight * cnn_probs + vlm_weight * vlm_probs

    @staticmethod
    def confidence_weighted_ensemble(
        cnn_probs: np.ndarray,
        vlm_probs: np.ndarray,
        cnn_confidence: float
    ) -> np.ndarray:
        """
        Dynamic weighting based on CNN confidence.
        Lower CNN confidence -> higher VLM weight.
        """
        # Map confidence to weight: high conf -> more CNN, low conf -> more VLM
        cnn_weight = min(0.8, max(0.3, cnn_confidence))
        vlm_weight = 1.0 - cnn_weight
        return cnn_weight * cnn_probs + vlm_weight * vlm_probs

    @staticmethod
    def rerank_with_vlm(
        cnn_top_k: List[int],
        vlm_probs: np.ndarray,
        class_names: List[str]
    ) -> List[Tuple[int, str, float]]:
        """Re-rank CNN top-K using VLM scores."""
        # Only consider CNN's top-K candidates
        reranked = []
        for idx in cnn_top_k:
            reranked.append((idx, class_names[idx], vlm_probs[idx]))

        # Sort by VLM confidence
        reranked.sort(key=lambda x: x[2], reverse=True)
        return reranked
```

### Step 3: Update Prediction Endpoint

```python
# api/routers/predict.py (modified predict function)

from ..services.vlm_service import VLMService
from ..services.ensemble_service import EnsembleService

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_vlm: bool = Query(False, description="Force VLM post-classification"),
    vlm_threshold: float = Query(0.85, description="Confidence threshold for VLM"),
    image_service: ImageService = Depends(get_image_service),
    model_manager: ModelManager = Depends(get_model_manager)
) -> PredictionResponse:
    """Predict cat breed with optional VLM refinement."""

    # 1. Preprocess
    tensor, metadata = await image_service.validate_and_preprocess(file)

    # 2. CNN inference
    start_time = time.perf_counter()
    cnn_probs, _ = model_manager.predict(tensor)
    cnn_time = (time.perf_counter() - start_time) * 1000

    # 3. Get top-5 from CNN
    top_5 = InferenceService.get_top_k_predictions(
        probs=cnn_probs,
        class_names=model_manager.class_names,
        k=5
    )
    top_5_names = [p.class_name for p in top_5]
    cnn_confidence = top_5[0].confidence

    # 4. VLM decision
    vlm_time = 0.0
    vlm_used = False

    should_use_vlm = use_vlm or VLMService.should_invoke_vlm(
        cnn_confidence=cnn_confidence,
        top_predictions=top_5_names,
        high_threshold=vlm_threshold
    )

    if should_use_vlm:
        vlm_service = await VLMService.get_instance()
        if vlm_service.is_loaded:
            # Get original image for VLM
            pil_image = await image_service.get_pil_image(file)

            vlm_start = time.perf_counter()
            vlm_indices, vlm_probs = vlm_service.predict(
                image=pil_image,
                class_names=model_manager.class_names,
                top_k=5
            )
            vlm_time = (time.perf_counter() - vlm_start) * 1000
            vlm_used = True

            # Ensemble predictions
            final_probs = EnsembleService.confidence_weighted_ensemble(
                cnn_probs=cnn_probs.squeeze(),
                vlm_probs=vlm_probs,
                cnn_confidence=cnn_confidence
            )

            # Re-generate top-5 with ensembled probs
            top_5 = InferenceService.get_top_k_predictions(
                probs=final_probs[np.newaxis, :],
                class_names=model_manager.class_names,
                k=5
            )

    # 5. Build response
    return PredictionResponse(
        predicted_class=top_5[0].class_name,
        confidence=top_5[0].confidence,
        top_5_predictions=top_5,
        inference_time_ms=round(cnn_time + vlm_time, 3),
        image_metadata=ImageMetadata(**metadata),
        model_info={
            "model_name": model_manager.model_name,
            "device": str(model_manager.device),
            "num_classes": len(model_manager.class_names),
            "vlm_used": vlm_used,
            "vlm_time_ms": round(vlm_time, 3) if vlm_used else None
        }
    )
```

### Step 4: Update Application Startup

```python
# api/main.py (add VLM loading)

from api.services.vlm_service import VLMService

@app.on_event("startup")
async def startup_event():
    # Load CNN model (existing)
    model_manager = await ModelManager.get_instance()
    await model_manager.load_model(...)

    # Load VLM model (new)
    vlm_service = await VLMService.get_instance()
    await vlm_service.load_model(
        model_name="openai/clip-vit-base-patch32",
        device="auto"
    )
    vlm_service.precompute_class_embeddings(model_manager.class_names)
```

### Step 5: Add Dependencies

```
# requirements.txt additions
transformers>=4.36.0
```

## Todo

- [ ] Create `api/services/vlm_service.py`
- [ ] Create `api/services/ensemble_service.py`
- [ ] Update `api/routers/predict.py`
- [ ] Update `api/main.py` startup
- [ ] Add CLIP to requirements.txt
- [ ] Test endpoint with/without VLM
- [ ] Benchmark latency

## Success Criteria

- [ ] VLM loads successfully on startup
- [ ] `/predict` works with `use_vlm=true`
- [ ] Selective invocation works (threshold logic)
- [ ] Priority breeds always trigger VLM
- [ ] p95 latency < 200ms with VLM
- [ ] No regression on CNN-only path

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| CLIP download fails | High | Pre-download model, cache locally |
| Memory exhaustion | Medium | Quantize CLIP, monitor VRAM |
| Latency spikes | Medium | Precompute embeddings, batch if needed |
| Accuracy regression | High | A/B test, compare metrics |

## Testing Checklist

```bash
# Start server with VLM
uvicorn api.main:app --reload

# Test CNN-only (high confidence image)
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict

# Test with forced VLM
curl -X POST -F "file=@test_image.jpg" "http://localhost:8000/predict?use_vlm=true"

# Test priority breed
curl -X POST -F "file=@burmilla.jpg" http://localhost:8000/predict
# Should show vlm_used: true
```

---
**Estimated Effort:** 1-2 days | **Dependencies:** None (Phase 1 optional)
