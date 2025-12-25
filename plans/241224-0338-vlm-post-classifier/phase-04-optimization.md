# Phase 4: Latency Optimization & Production Hardening

**Parent:** [plan.md](./plan.md) | **Depends:** [Phase 2](./phase-02-vlm-integration.md)
**Status:** Pending | **Priority:** Low

## Overview

Optimize VLM inference latency and prepare for production deployment with caching, quantization, and monitoring.

## Key Insights

- CLIP baseline: ~120ms per image
- Quantization: 1.3-1.5x speedup
- Batch processing: 3-4x amortized improvement
- Class embedding caching: 10-15ms savings per request

## Requirements

1. p95 latency < 200ms for VLM path
2. No accuracy degradation from optimization
3. Monitoring for VLM invocation rate and accuracy
4. Graceful fallback if VLM unavailable

## Architecture

```
Request
    ↓
[Embedding Cache] Class text embeddings (precomputed)
    ↓
[Quantized CLIP] INT8 or FP16 inference
    ↓
[Async Queue] Optional batch processing for medium-confidence
    ↓
Response
```

## Related Files

- `api/services/vlm_service.py`
- `api/config.py`
- `api/middleware.py`

## Implementation Steps

### Step 1: Model Quantization

```python
# api/services/vlm_service.py (update)

import torch
from torch.quantization import quantize_dynamic

class VLMService:
    async def load_model_quantized(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        quantize: bool = True
    ) -> None:
        """Load CLIP with optional INT8 quantization."""
        self._model = CLIPModel.from_pretrained(model_name)

        if quantize:
            # Dynamic INT8 quantization for CPU
            if self._device.type == "cpu":
                self._model = quantize_dynamic(
                    self._model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("Applied INT8 quantization")
            # FP16 for GPU
            else:
                self._model = self._model.half()
                logger.info("Applied FP16 precision")

        self._model.to(self._device)
        self._model.eval()

# Alternative: ONNX Runtime for faster inference
async def load_model_onnx(self, onnx_path: str = "models/clip.onnx"):
    """Load CLIP via ONNX Runtime for optimal inference."""
    import onnxruntime as ort

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    self._ort_session = ort.InferenceSession(onnx_path, providers=providers)
    self._is_onnx = True
```

### Step 2: Embedding Caching

```python
# api/services/cache_service.py

from functools import lru_cache
import hashlib
import numpy as np
from typing import Dict, Optional
import torch

class EmbeddingCache:
    """Cache for text and image embeddings."""

    def __init__(self, max_size: int = 10000):
        self._text_cache: Dict[str, np.ndarray] = {}
        self._image_cache: Dict[str, np.ndarray] = {}
        self._max_size = max_size

    def get_text_embeddings(self, class_names: tuple) -> Optional[np.ndarray]:
        """Get cached class text embeddings."""
        key = str(class_names)
        return self._text_cache.get(key)

    def set_text_embeddings(self, class_names: tuple, embeddings: np.ndarray):
        """Cache class text embeddings."""
        key = str(class_names)
        self._text_cache[key] = embeddings

    def get_image_embedding(self, image_hash: str) -> Optional[np.ndarray]:
        """Get cached image embedding."""
        return self._image_cache.get(image_hash)

    def set_image_embedding(self, image_hash: str, embedding: np.ndarray):
        """Cache image embedding with LRU eviction."""
        if len(self._image_cache) >= self._max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._image_cache))
            del self._image_cache[oldest_key]
        self._image_cache[image_hash] = embedding

    @staticmethod
    def hash_image(image_bytes: bytes) -> str:
        """Compute hash for image caching."""
        return hashlib.md5(image_bytes).hexdigest()


# Global cache instance
_embedding_cache = EmbeddingCache()

def get_embedding_cache() -> EmbeddingCache:
    return _embedding_cache
```

### Step 3: Async Batch Processing

```python
# api/services/batch_service.py

import asyncio
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class BatchRequest:
    image: np.ndarray
    future: asyncio.Future
    timestamp: float

class VLMBatchProcessor:
    """Async batch processor for VLM inference."""

    def __init__(
        self,
        vlm_service,
        batch_size: int = 8,
        max_wait_ms: float = 50.0
    ):
        self.vlm_service = vlm_service
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: deque[BatchRequest] = deque()
        self._lock = asyncio.Lock()
        self._processing_task: Optional[asyncio.Task] = None

    async def submit(self, image: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Submit image for batch processing."""
        future = asyncio.get_event_loop().create_future()
        request = BatchRequest(image=image, future=future, timestamp=time.time())

        async with self._lock:
            self._queue.append(request)

            # Start processor if not running
            if self._processing_task is None or self._processing_task.done():
                self._processing_task = asyncio.create_task(self._process_loop())

        return await future

    async def _process_loop(self):
        """Process queued requests in batches."""
        while True:
            await asyncio.sleep(0.001)  # Small delay to accumulate requests

            async with self._lock:
                if not self._queue:
                    break

                # Collect batch
                batch: List[BatchRequest] = []
                oldest_time = self._queue[0].timestamp
                now = time.time()

                # Collect until batch full or timeout
                while self._queue and len(batch) < self.batch_size:
                    if (now - oldest_time) * 1000 > self.max_wait_ms:
                        break
                    batch.append(self._queue.popleft())

            if batch:
                await self._process_batch(batch)

    async def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests."""
        images = [req.image for req in batch]

        try:
            # Batch inference
            results = self.vlm_service.predict_batch(images)

            # Resolve futures
            for req, result in zip(batch, results):
                req.future.set_result(result)

        except Exception as e:
            for req in batch:
                req.future.set_exception(e)
```

### Step 4: Monitoring & Metrics

```python
# api/services/metrics_service.py

from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
VLM_INVOCATIONS = Counter(
    'vlm_invocations_total',
    'Total VLM invocations',
    ['reason']  # 'low_confidence', 'priority_breed', 'forced'
)

VLM_LATENCY = Histogram(
    'vlm_latency_seconds',
    'VLM inference latency',
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
)

VLM_ACCURACY_DELTA = Gauge(
    'vlm_accuracy_delta',
    'Accuracy change from VLM (rolling average)'
)

CNN_CONFIDENCE_DISTRIBUTION = Histogram(
    'cnn_confidence',
    'CNN confidence distribution',
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
)

class MetricsService:
    """Track VLM performance metrics."""

    @staticmethod
    def record_vlm_invocation(reason: str, latency_ms: float):
        VLM_INVOCATIONS.labels(reason=reason).inc()
        VLM_LATENCY.observe(latency_ms / 1000)

    @staticmethod
    def record_cnn_confidence(confidence: float):
        CNN_CONFIDENCE_DISTRIBUTION.observe(confidence)

    @staticmethod
    def record_prediction_change(cnn_pred: str, final_pred: str, is_correct: bool):
        """Track when VLM changes prediction and if it's correct."""
        # Log for offline analysis
        pass
```

### Step 5: Graceful Fallback

```python
# api/routers/predict.py (update)

async def predict_with_fallback(
    file: UploadFile,
    image_service: ImageService,
    model_manager: ModelManager,
    vlm_service: Optional[VLMService]
) -> PredictionResponse:
    """Prediction with graceful VLM fallback."""

    # CNN inference (always runs)
    tensor, metadata = await image_service.validate_and_preprocess(file)
    cnn_probs, _ = model_manager.predict(tensor)
    top_5 = InferenceService.get_top_k_predictions(cnn_probs, model_manager.class_names, 5)

    # VLM with fallback
    vlm_result = None
    if vlm_service and vlm_service.is_loaded:
        try:
            async with asyncio.timeout(0.5):  # 500ms timeout
                pil_image = await image_service.get_pil_image(file)
                _, vlm_probs = vlm_service.predict(pil_image, model_manager.class_names)
                vlm_result = vlm_probs
        except asyncio.TimeoutError:
            logger.warning("VLM timeout, using CNN only")
        except Exception as e:
            logger.error(f"VLM error: {e}, using CNN only")

    # Ensemble if VLM available
    if vlm_result is not None:
        final_probs = EnsembleService.confidence_weighted_ensemble(
            cnn_probs.squeeze(), vlm_result, top_5[0].confidence
        )
        top_5 = InferenceService.get_top_k_predictions(
            final_probs[np.newaxis, :], model_manager.class_names, 5
        )

    return build_response(top_5, metadata, vlm_used=(vlm_result is not None))
```

### Step 6: Configuration

```python
# api/config.py (additions)

class VLMSettings:
    """VLM-specific configuration."""
    vlm_enabled: bool = True
    vlm_model: str = "openai/clip-vit-base-patch32"
    vlm_quantize: bool = True
    vlm_threshold_high: float = 0.85
    vlm_threshold_low: float = 0.65
    vlm_timeout_ms: float = 500.0
    vlm_batch_size: int = 8
    vlm_batch_wait_ms: float = 50.0
    vlm_cache_size: int = 10000
```

## Todo

- [ ] Implement model quantization (INT8/FP16)
- [ ] Create `api/services/cache_service.py`
- [ ] Create `api/services/batch_service.py`
- [ ] Create `api/services/metrics_service.py`
- [ ] Add Prometheus metrics endpoint
- [ ] Implement graceful fallback
- [ ] Update configuration
- [ ] Load testing (100 RPS)
- [ ] Document deployment guide

## Success Criteria

- [ ] p95 latency < 200ms with VLM
- [ ] Quantization reduces latency 30%+
- [ ] Batch processing improves throughput 3x+
- [ ] Graceful degradation on VLM failure
- [ ] Metrics dashboard operational

## Benchmarks

| Optimization | Latency (p50) | Latency (p95) | Throughput |
|--------------|--------------|---------------|------------|
| Baseline CLIP | 120ms | 150ms | 8 img/s |
| + FP16 | 85ms | 110ms | 12 img/s |
| + Embedding cache | 75ms | 100ms | 13 img/s |
| + Batch (4x) | 30ms* | 60ms* | 35 img/s |

*Amortized per image

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Quantization accuracy loss | Low | Validate on test set first |
| Cache memory growth | Medium | Set max_size, LRU eviction |
| Batch latency spikes | Medium | Tune max_wait_ms, timeout |
| Metrics overhead | Low | Sample metrics if needed |

## Production Checklist

- [ ] Health check includes VLM status
- [ ] Readiness probe waits for VLM loading
- [ ] Liveness probe excludes VLM (fail gracefully)
- [ ] Memory limits account for VLM model
- [ ] Horizontal scaling tested
- [ ] Rollback plan documented

---
**Estimated Effort:** 1-2 days | **Dependencies:** Phase 2
