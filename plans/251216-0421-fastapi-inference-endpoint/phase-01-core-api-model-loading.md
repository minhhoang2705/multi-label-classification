# Phase 01: Core API Structure & Model Loading

## Context

- [Main Plan](./plan.md)
- Research: [FastAPI ML Inference](../reports/251216-fastapi-ml-inference-research.md)
- Scout: [Model Inference Patterns](../reports/251216-scout-model-inference-patterns.md)

## Overview

| Field | Value |
|-------|-------|
| Date | 2025-12-16 |
| Priority | High |
| Status | Completed ✓ |
| Est. Time | 2-3 hours |
| Actual Time | 2 hours |
| Review Date | 2025-12-16 |
| Completion Date | 2025-12-16 |

## Key Insights

1. **Singleton pattern** with lifespan context manager for model loading
2. **Dependency injection** via FastAPI Depends for clean architecture
3. **Device detection** - auto-select CUDA/MPS/CPU
4. **Checkpoint structure** contains `model_state_dict` key

## Requirements

- FastAPI app with lifespan context manager
- ModelManager singleton class
- Model loads at startup, cleanup at shutdown
- Configurable checkpoint path and model name
- Device auto-detection (cuda > mps > cpu)

## Architecture

```python
# api/services/model_service.py
class ModelManager:
    _instance: Optional["ModelManager"] = None
    _model: Optional[nn.Module] = None
    _device: Optional[torch.device] = None
    _class_names: List[str] = []

    @classmethod
    async def get_instance(cls) -> "ModelManager": ...

    async def load_model(self, checkpoint_path: str, model_name: str): ...

    def predict(self, tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]: ...
```

```python
# api/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    manager = await ModelManager.get_instance()
    await manager.load_model(
        checkpoint_path=settings.checkpoint_path,
        model_name=settings.model_name
    )
    yield
    # Shutdown: cleanup

app = FastAPI(lifespan=lifespan)
```

## Related Code Files

| File | Purpose |
|------|---------|
| `src/models.py` | TransferLearningModel, create_model(), load_checkpoint() |
| `src/config.py` | ModelConfig (num_classes=67, dropout=0.2) |
| `scripts/test.py` | Inference patterns, model.eval(), torch.no_grad() |

## Implementation Steps

### 1. Create API Directory Structure

```bash
mkdir -p api/services api/routers
touch api/__init__.py api/services/__init__.py api/routers/__init__.py
```

### 2. Create Configuration (api/config.py)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model
    checkpoint_path: str = "outputs/checkpoints/fold_0/best_model.pt"
    model_name: str = "resnet50"
    num_classes: int = 67
    image_size: int = 224

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Device
    device: str = "auto"  # auto, cuda, mps, cpu

    class Config:
        env_prefix = "API_"
```

### 3. Create Model Service (api/services/model_service.py)

Key patterns from existing code:
- Use `timm.create_model()` with `num_classes=0` then custom classifier
- Load via `torch.load(path, map_location=device)`
- Check for `model_state_dict` key in checkpoint
- Call `model.eval()` for inference mode

```python
import torch
import torch.nn as nn
import timm
from typing import Optional, List, Tuple
import numpy as np

class ModelManager:
    _instance = None

    def __init__(self):
        self._model = None
        self._device = None
        self._class_names = []
        self._is_loaded = False

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def _get_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    async def load_model(
        self,
        checkpoint_path: str,
        model_name: str,
        num_classes: int = 67,
        device: str = "auto"
    ):
        self._device = self._get_device(device)

        # Create model (same pattern as src/models.py)
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        num_features = backbone.num_features

        self._model = nn.Sequential(
            backbone,
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        if 'model_state_dict' in checkpoint:
            self._model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self._model.load_state_dict(checkpoint)

        self._model.to(self._device)
        self._model.eval()
        self._is_loaded = True

        # Load class names (67 breeds)
        self._load_class_names()

    def _load_class_names(self):
        # Hardcoded sorted breed names (from scout report)
        self._class_names = [
            "Abyssinian", "American Bobtail", "American Curl", ...
            # All 67 breeds
        ]

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def predict(self, tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference. Returns (probabilities, class_indices)."""
        with torch.no_grad():
            tensor = tensor.to(self._device)
            outputs = self._model(tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy(), outputs.cpu().numpy()
```

### 4. Create Main App (api/main.py)

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

from .config import Settings
from .services.model_service import ModelManager
from .routers import health, predict

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    manager = await ModelManager.get_instance()
    await manager.load_model(
        checkpoint_path=settings.checkpoint_path,
        model_name=settings.model_name,
        num_classes=settings.num_classes,
        device=settings.device
    )
    yield
    # Shutdown (cleanup if needed)

app = FastAPI(
    title="Cat Breeds Classification API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
```

### 5. Create Health Router (api/routers/health.py)

```python
from fastapi import APIRouter, Depends
from ..services.model_service import ModelManager

router = APIRouter()

@router.get("/health/live")
async def liveness():
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness():
    manager = await ModelManager.get_instance()
    return {
        "status": "ready" if manager.is_loaded else "not_ready",
        "model_loaded": manager.is_loaded,
        "device": str(manager.device) if manager.device else None
    }
```

## Todo List

- [x] Create api/ directory structure
- [x] Implement api/config.py with Settings
- [x] Implement api/services/model_service.py with ModelManager
- [x] Implement api/main.py with lifespan
- [x] Implement api/routers/health.py
- [x] Test model loading at startup
- [x] Verify health endpoints work

## Success Criteria

1. `uvicorn api.main:app` starts without errors
2. Model loads within 10 seconds
3. `/health/live` returns `{"status": "alive"}`
4. `/health/ready` returns model_loaded=true
5. Device correctly detected (cuda/mps/cpu)

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Checkpoint path not found | Validate path at startup, clear error message |
| Model architecture mismatch | Use same TransferLearningModel structure as training |
| CUDA OOM on load | Support CPU fallback |
| Slow startup | Log loading progress, async loading |

## Security Considerations

- No secrets in config defaults
- Environment variable override for paths
- Model file should be read-only

## Code Review Findings

**Review Date:** 2025-12-16
**Reviewer:** code-reviewer agent
**Status:** ✅ Passed - 3 critical security fixes required before production

### Implementation Success
- All 7 tasks completed ✅
- All 5 success criteria met ✅
- 25 comprehensive tests passing ✅
- Clean architecture following YAGNI/KISS/DRY ✅

### Critical Security Issues (Must Fix Before Production)
1. **CORS Wildcard** - `allow_origins=["*"]` enables CSRF attacks (OWASP A05, A07)
2. **Path Traversal** - No validation on checkpoint_path allows arbitrary file access (OWASP A01, A03)
3. **No Rate Limiting** - Enables DoS attacks (OWASP A04)

### High Priority Improvements
4. Async file loading (torch.load blocks event loop)
5. Replace print() with proper logging
6. Add retry logic for model loading failures

### Review Report
See detailed analysis: [Code Review Report](../reports/code-reviewer-251216-phase01-review.md)

**Recommendation:** Address 3 critical security issues before production deployment, then proceed to Phase 02.

---

## Next Steps

After Phase 01:
- **Immediate:** Fix critical security issues (CORS, path validation, rate limiting)
- **Then:** [Phase 02: Image Validation & Preprocessing](./phase-02-image-validation-preprocessing.md)
