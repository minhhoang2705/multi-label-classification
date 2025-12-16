# API Quick Reference - Phases 01 & 02

## Endpoints

| Method | Path | Purpose | Status | Phase |
|--------|------|---------|--------|-------|
| GET | `/` | API info | 200 | 01 |
| GET | `/health/live` | Liveness probe | 200 | 01 |
| GET | `/health/ready` | Readiness probe | 200/503 | 01 |
| POST | `/predict` | Single image inference | 200/400/413/500/503 | 02 |

## Configuration

**Environment Variables (API_ prefix):**
```bash
API_CHECKPOINT_PATH=outputs/checkpoints/fold_0/best_model.pt
API_MODEL_NAME=resnet50
API_DEVICE=auto  # auto|cuda|mps|cpu
API_PORT=8000
API_HOST=0.0.0.0
```

## Start API

```bash
python -m uvicorn api.main:app --reload
```

## Test API

```bash
# Phase 01 tests
pytest tests/test_api_phase01.py -v

# Phase 02 tests (61 tests)
pytest tests/test_api_phase02.py -v

# Health endpoints
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready

# Single image inference
curl -X POST "http://localhost:8000/predict" \
  -F "file=@cat.jpg"
```

## Key Components

| Component | File | Role | Phase |
|-----------|------|------|-------|
| **App** | `api/main.py` | FastAPI app, lifecycle, exception handlers | 01/02 |
| **Config** | `api/config.py` | Pydantic settings | 01 |
| **Manager** | `api/services/model_service.py` | Model singleton | 01 |
| **Health** | `api/routers/health.py` | Health probes | 01 |
| **ImageService** | `api/services/image_service.py` | Validation & preprocessing | 02 |
| **Dependencies** | `api/dependencies.py` | DI factories | 02 |
| **Exceptions** | `api/exceptions.py` | Custom HTTP exceptions | 02 |

## ModelManager

```python
# Get singleton instance
manager = await ModelManager.get_instance()

# Properties
manager.is_loaded       # bool
manager.device          # torch.device
manager.model_name      # str
manager.class_names     # List[67 breeds]
manager.checkpoint_path # str
```

## ImageService (Phase 02)

```python
# Validate and preprocess image
tensor, metadata = await image_service.validate_and_preprocess(file)

# metadata: {original_width, original_height, format, mode, file_size_bytes, filename}
# tensor: shape (1, 3, 224, 224) ready for inference
```

**Validation layers:**
1. MIME type (jpeg/png/webp only)
2. File size (max 10MB)
3. Image structure (PIL verify)
4. Dimensions (16-10000px)
5. Pixel loading (decompression bomb protection)

## Security Features

**Phase 01:**
- ✓ Path validation (prevents traversal)
- ✓ CORS restricted origins
- ✓ Structured logging
- ✓ Device auto-detection

**Phase 02:**
- ✓ Decompression bomb protection (PIL MAX_IMAGE_PIXELS)
- ✓ Pixel flood attack prevention (dims validated before pixel load)
- ✓ MIME type whitelist
- ✓ Memory exhaustion defense
- ✓ Type-safe metadata (TypedDict)

## Documentation

- **Phase 01:** `docs/api-phase01.md`
- **Phase 02:** `docs/api-phase02.md` (Image validation & preprocessing)
- **Testing Guide:** `docs/testing-guide.md`
- **Interactive Docs:** http://localhost:8000/docs

---

**Phase 02 Status:** Complete (61 tests, security hardened)
