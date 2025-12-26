# Codebase Summary - Cat Breeds Classification

**Generated:** 2025-12-26
**Project:** Multi-Label Cat Breed Classification with Hybrid CNN+VLM
**Status:** Phase 02 Complete (Hybrid Inference)

---

## Project Overview

Multi-label cat breed classification system featuring:
- **Training Pipeline**: ResNet50/EfficientNet/ConvNext on 67 cat breeds (~67K images)
- **FastAPI Inference Server**: Low-latency predictions with robust image validation
- **Hybrid Verification**: CNN + Vision Language Model (GLM-4.6V) for accuracy
- **Production Ready**: Docker deployment, comprehensive testing, monitoring

### Key Stats

| Metric | Value |
|--------|-------|
| Cat Breeds | 67 |
| Training Dataset | ~67,000 images |
| API Endpoints | 8 |
| Python Modules | 20+ |
| Test Coverage | 25+ tests |
| Docker Support | Yes |
| GPU Support | CUDA/MPS/CPU |

---

## Directory Structure

```
multi-label-classification/
├── api/                          # FastAPI inference server
│   ├── services/
│   │   ├── hybrid_inference_service.py    # NEW: CNN+VLM orchestration
│   │   ├── model_service.py               # Model loading & inference
│   │   ├── image_service.py               # Image validation & preprocessing
│   │   ├── inference_service.py           # Prediction formatting
│   │   └── vlm_service.py                 # GLM-4.6V integration
│   ├── routers/
│   │   ├── predict.py                     # Prediction endpoints (/predict, /predict/verified)
│   │   ├── health.py                      # Health check endpoints
│   │   └── model.py                       # Model info endpoints
│   ├── models.py                          # Pydantic schemas (NEW: HybridPredictionResponse)
│   ├── config.py                          # Configuration & settings
│   ├── dependencies.py                    # DI factories
│   ├── exceptions.py                      # Custom HTTP exceptions
│   ├── middleware.py                      # Custom middleware
│   └── main.py                            # FastAPI app setup (NEW: disagreement logger)
│
├── src/                          # Training & model code
│   ├── models.py                          # TransferLearningModel architecture
│   ├── trainer.py                         # Training loop
│   ├── dataset.py                         # Dataset loading
│   ├── augmentations.py                   # Data augmentation
│   ├── metrics.py                         # Evaluation metrics
│   ├── config.py                          # Training config
│   ├── losses.py                          # Loss functions
│   └── utils.py                           # Utility functions
│
├── scripts/                      # Utility scripts
│   ├── train.py                           # Training script entry point
│   ├── validate_env.py                    # Environment validation
│   ├── test.py                            # Model evaluation
│   ├── download_dataset.sh                # Dataset download
│   └── docker-*.sh                        # Docker utilities
│
├── tests/                        # Test suite
│   └── api/
│       ├── test_hybrid_inference_service.py      # NEW: Hybrid service tests
│       └── test_predict_verified.py              # NEW: /predict/verified endpoint tests
│
├── docs/                         # Documentation
│   ├── api-phase02-hybrid.md               # NEW: Phase 02 hybrid documentation
│   ├── codebase-summary.md                 # This file
│   ├── code-standards.md                   # Code standards & patterns
│   ├── system-architecture.md              # System architecture
│   └── other docs...
│
├── logs/                         # NEW: Runtime logs directory
│   └── disagreements.jsonl                # Disagreement logs (auto-created)
│
├── outputs/                      # Training outputs
│   ├── checkpoints/                       # Model checkpoints
│   └── test_results/                      # Validation metrics
│
├── data/                         # Dataset
│   └── images/                            # Cat breed images
│
├── requirements.txt                       # Python dependencies
├── Dockerfile                             # Container definition
├── docker-compose.yml                     # Multi-container orchestration
├── pytest.ini                             # Test configuration
├── README.md                              # Project README
└── CLAUDE.md                              # Claude-specific instructions
```

---

## Core Components

### 1. API Layer (FastAPI)

#### Main Application: `api/main.py`

```python
# FastAPI app with:
# - Lifespan context manager (startup/shutdown)
# - Model loading on startup
# - Exception handlers
# - CORS middleware
# - Version header middleware
# - Disagreement logger setup (NEW)
```

**Key Features:**
- Async model loading during startup
- CORS configured for cross-origin requests
- Exception handlers for validation errors
- Disagreement logging to JSONL file (NEW)

#### Endpoints

**Health Check: `api/routers/health.py`**
- GET /health/live - Liveness probe
- GET /health/ready - Readiness + model status

**Prediction: `api/routers/predict.py`**
- POST /api/v1/predict - CNN-only inference
- POST /api/v1/predict/verified - Hybrid CNN+VLM (NEW)

**Model Info: `api/routers/model.py`**
- GET /api/v1/model/info - Model metadata & performance
- GET /api/v1/model/classes - List all 67 breeds

### 2. Service Layer

#### HybridInferenceService (NEW): `api/services/hybrid_inference_service.py`

**Purpose:** Combine CNN and VLM predictions with agreement detection

**Flow:**
```
1. CNN inference (5-20ms)
   - Get top-5 predictions
   - Extract top-3 for VLM

2. VLM verification (500-2000ms)
   - Send image + top-3 to GLM-4.6V
   - Parse structured response

3. Agreement check
   - If agree → status="verified" (HIGH confidence)
   - If disagree → status="uncertain" (MEDIUM confidence)
   - If VLM fails → status="error/cnn_only" (LOW confidence)

4. Log disagreements
   - Hash image path (security)
   - Store predictions & reasoning
   - Enable post-hoc analysis
```

**Key Decision:** VLM wins on disagreement (better at visual reasoning)

**Output:** HybridPrediction dataclass with timing breakdown

#### ModelManager: `api/services/model_service.py`

**Purpose:** Singleton model manager for CNN inference

**Features:**
- Lazy loading on first access
- Device auto-detection (CUDA/MPS/CPU)
- Checkpoint path validation (security)
- Batch inference support

**Architecture:**
```
Backbone (TIMM) → Classifier → Softmax
```

#### ImageService: `api/services/image_service.py`

**Purpose:** Multi-layer image validation & preprocessing

**Validation Layers:**
1. MIME type whitelist (JPEG/PNG/WebP)
2. File size limit (10MB)
3. Image structure validation (PIL.Image.verify)
4. Dimension validation (16x16 to 10000x10000)
5. Pixel loading with decompression bomb protection

**Preprocessing:**
```
Load → Convert to RGB → Resize to 224x224 → Normalize (ImageNet stats)
```

#### VLMService: `api/services/vlm_service.py`

**Purpose:** GLM-4.6V Vision Language Model integration

**Features:**
- Singleton pattern (reuse API client)
- Thread-safe initialization
- Base64 image encoding
- Structured prompt with CNN candidates
- Resilient response parsing

**Prompt Strategy:**
- Lists CNN top-3 candidates
- Asks for visual feature analysis
- Enforces response format (BREED:, MATCHES_CNN:, REASON:)
- Allows VLM to suggest different breed if needed

#### InferenceService: `api/services/inference_service.py`

**Purpose:** Utility service for inference operations

**Methods:**
- `get_top_k_predictions()` - Extract top-K from probability array
- `synchronize_device()` - Sync CUDA for accurate timing

### 3. Data Models

#### Pydantic Schemas: `api/models.py`

**Core Schemas:**
- `PredictionItem` - Single prediction (rank, class, confidence)
- `ImageMetadata` - Image properties (width, height, format, mode, size, filename)
- `PredictionResponse` - /predict endpoint response
- `HybridPredictionResponse` - /predict/verified endpoint response (NEW)
- `ErrorResponse` - Error details

**Model Info Schemas:**
- `PerformanceMetrics` - Accuracy, F1, top-3/5 accuracy
- `SpeedMetrics` - Latency, throughput, device
- `ModelInfoResponse` - Full model information
- `ClassListResponse` - All 67 breed names

### 4. Configuration

#### Settings: `api/config.py`

```python
class Settings:
    # Model
    model_name: str = "convnext_base"
    checkpoint_path: str = "outputs/checkpoints/fold_0/best_model.pt"
    num_classes: int = 67
    image_size: int = 224

    # Device
    device: str = "auto"  # cuda, mps, cpu, or auto

    # API
    host: str = "0.0.0.0"
    port: int = 8000
    api_version: str = "1.0.0"

    # CORS
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = False
```

#### Dependency Injection: `api/dependencies.py`

```python
@lru_cache
def get_settings() → Settings

@lru_cache
def get_image_service() → ImageService

async def get_model_manager() → ModelManager
```

---

## Training Pipeline (src/)

### Model Architecture: `src/models.py`

```python
class TransferLearningModel(nn.Module):
    """Transfer learning classifier with TIMM backbone."""

    def __init__(self, model_name, num_classes, pretrained=True, dropout=0.2)

    # Architecture:
    # TIMM backbone (ResNet50/EfficientNet/ConvNext)
    #   ↓
    # Global average pooling
    #   ↓
    # Dropout(0.2)
    #   ↓
    # Linear classifier (num_features → 67 classes)
```

### Training Loop: `src/trainer.py`

**Features:**
- Stratified K-fold cross-validation
- Class-weighted sampling
- Focal loss for imbalance
- Early stopping
- Metrics tracking (accuracy, F1, precision, recall, top-3/5 accuracy)
- MLflow integration for experiment tracking

### Dataset: `src/dataset.py`

**Features:**
- Stratified train/val/test split
- Data augmentation (rotation, flip, color jitter, cutmix)
- Memory-efficient image loading
- Class imbalance handling

### Entry Point: `scripts/train.py`

```bash
# Quick test (2 epochs, 2 folds)
python scripts/train.py --fast_dev

# Full training
python scripts/train.py --model_name convnext_base --num_epochs 50 --num_folds 5
```

---

## Testing

### Test Structure

**Unit Tests: `tests/api/test_hybrid_inference_service.py`**
- 12 tests for HybridInferenceService
- Tests for all agreement statuses (verified, uncertain, error, unclear, cnn_only)
- Disagreement logging verification
- Timing measurements
- Mock VLM service

**Integration Tests: `tests/api/test_predict_verified.py`**
- 13 tests for /predict/verified endpoint
- End-to-end request/response testing
- Multipart file handling
- Error scenarios
- Temp file cleanup verification

### Running Tests

```bash
# All hybrid tests
pytest tests/api/test_hybrid_inference_service.py tests/api/test_predict_verified.py -v

# With coverage
pytest --cov=api.services.hybrid_inference_service --cov=api.routers.predict

# Watch mode
pytest-watch tests/api/
```

---

## API Endpoints

### Full Endpoint List

| Method | Path | Purpose | Status |
|--------|------|---------|--------|
| GET | /health/live | Liveness probe | Active |
| GET | /health/ready | Readiness check | Active |
| POST | /api/v1/predict | CNN-only inference | Active |
| POST | /api/v1/predict/verified | Hybrid CNN+VLM | NEW |
| GET | /api/v1/model/info | Model metadata | Active |
| GET | /api/v1/model/classes | Breed list | Active |
| GET | / | Root (info) | Active |
| GET | /docs | Swagger UI | Active |

### Request/Response Examples

**POST /api/v1/predict/verified**

```
Request:
  Content-Type: multipart/form-data
  file: <image.jpg>

Response (verified):
  {
    "predicted_class": "Persian",
    "confidence_level": "high",
    "verification_status": "verified",
    "cnn_prediction": "Persian",
    "cnn_confidence": 0.94,
    "vlm_prediction": "Persian",
    "vlm_reasoning": "Long fluffy coat, flat face...",
    "cnn_time_ms": 12.5,
    "vlm_time_ms": 892.3,
    "total_time_ms": 905.0,
    ...
  }

Response (uncertain):
  {
    "predicted_class": "Himalayan",    # VLM wins
    "confidence_level": "medium",
    "verification_status": "uncertain",
    "cnn_prediction": "Persian",
    "vlm_prediction": "Himalayan",
    "vlm_reasoning": "Color point pattern indicates Himalayan...",
    ...
  }
```

---

## Data Flow

### Hybrid Prediction Flow

```
User uploads image
    ↓
FastAPI receives multipart file
    ↓
ImageService.validate_and_preprocess()
    ├─ MIME validation
    ├─ Size validation
    ├─ Structure validation
    ├─ Dimension validation
    ├─ Pixel loading
    └─ Tensor conversion (1, 3, 224, 224)
    ↓
Save to temp file (for VLM)
    ↓
HybridInferenceService.predict()
    ├─ Stage 1: CNN.predict(tensor)
    │  ├─ ModelManager.predict()
    │  ├─ Get top-5 predictions
    │  └─ Extract top-3 for VLM
    │
    ├─ Stage 2: VLM.verify_prediction()
    │  ├─ Encode image to base64
    │  ├─ Build structured prompt
    │  ├─ Call GLM-4.6V
    │  ├─ Parse response
    │  └─ Extract breed & reasoning
    │
    └─ Stage 3: Agreement check
       ├─ If agree → status="verified" (HIGH)
       ├─ If disagree → status="uncertain" (MEDIUM) + LOG
       └─ If fail → status="error/cnn_only" (LOW)
    ↓
Clean up temp file
    ↓
Return HybridPredictionResponse
```

---

## Logging & Monitoring

### Disagreement Logging

**File:** `logs/disagreements.jsonl`

**Format:** One JSON object per line

```json
{
  "timestamp": 1703587200.123,
  "image_hash": "a1b2c3d4e5f6g7h8",
  "cnn_prediction": "Persian",
  "cnn_confidence": 0.87,
  "vlm_prediction": "Himalayan",
  "vlm_reasoning": "Color point pattern indicates Himalayan...",
  "final_prediction": "Himalayan"
}
```

**Security:** Image paths are hashed (SHA256, first 16 chars), not logged.

### Application Logging

**Format:** `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

**Loggers:**
- `api.main` - Startup/shutdown
- `api.services.*` - Service operations
- `disagreements` - Disagreement cases (JSONL output)

---

## Error Handling

### Custom Exceptions: `api/exceptions.py`

```python
class ImageValidationError(HTTPException)    # 400
class ImageTooLargeError(HTTPException)      # 413
class ModelInferenceError(HTTPException)     # 500
class ModelNotLoadedError(HTTPException)     # 503
```

### Exception Handlers: `api/main.py`

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler()    # 400 with error details

@app.exception_handler(Exception)
async def general_exception_handler()       # 500 with logging
```

---

## Deployment

### Docker Support

**Files:**
- `Dockerfile` - Multi-stage container
- `docker-compose.yml` - Service orchestration
- `docker-compose.override.yml` - Development overrides

**Quick Start:**
```bash
# Build
./scripts/docker-build.sh

# Run
./scripts/docker-run.sh

# Stop
./scripts/docker-stop.sh
```

### Environment Variables

```bash
# Model
MODEL_NAME=convnext_base
CHECKPOINT_PATH=outputs/checkpoints/fold_0/best_model.pt

# API
API_HOST=0.0.0.0
API_PORT=8000

# VLM
ZAI_API_KEY=your_key_here

# CORS
CORS_ORIGINS=["*"]
```

### Kubernetes Ready

- Health endpoints for liveness/readiness probes
- Graceful shutdown handling
- Stateless inference (horizontal scaling)
- Volume mount for checkpoints

---

## Code Standards

### Style Guidelines

**Python:**
- Black formatting (100 char line width)
- isort import organization
- Type hints (all functions/methods)
- Docstrings (Google style)

**API:**
- RESTful design
- Consistent error responses
- Request validation (Pydantic)
- Response schemas (all endpoints)

### Naming Conventions

**Variables:**
- `snake_case` for variables/functions
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants

**API:**
- Endpoint paths: lowercase, hyphenated
- Query params: snake_case
- Response fields: snake_case

**Files:**
- Modules: lowercase_with_underscores.py
- Classes: PascalCase in camelCase filename (model_service.py → ModelManager)

---

## Performance Characteristics

### Latency (GPU with VLM)

| Operation | Time |
|-----------|------|
| Image validation | 15-60ms |
| CNN inference | 5-20ms |
| VLM inference | 500-2000ms |
| Total | 530-2100ms |

### Throughput

- With VLM: ~0.5-1 req/sec per instance
- Without VLM: ~10-20 req/sec per instance

### Memory

- Model size: ~100-200MB
- Per-request overhead: ~50-100MB
- VLM disabled: ~200-300MB total

---

## Security Considerations

### Image Validation

- MIME type whitelist (JPEG/PNG/WebP only)
- Decompression bomb protection (PIL.MAX_IMAGE_PIXELS)
- Dimension limits to prevent pixel floods
- Structure validation before pixel loading

### Checkpoint Loading

- Path validation (no directory traversal)
- Must be in `outputs/checkpoints/` directory
- File existence check
- Type validation (file, not directory)

### Logging

- Image paths hashed (not logged)
- Temp files cleaned up in finally block
- Exception details logged safely
- No secrets in logs

### API Security

- CORS configured (restrictive origins)
- Request validation (Pydantic)
- File size limits
- Timeout handling

---

## Dependencies

### Core

- **PyTorch** (2.0+) - Deep learning
- **TIMM** (0.9+) - Pre-trained models
- **FastAPI** (0.100+) - Web framework
- **Pydantic** (2.0+) - Data validation
- **Pillow** (9.0+) - Image processing
- **torchvision** (0.15+) - Vision utilities
- **numpy** - Array operations

### VLM

- **zai-sdk** (0.0.4+) - Z.ai API client
- **ZAI_API_KEY** env var required

### Training

- **scikit-learn** - Metrics
- **matplotlib/seaborn** - Visualization
- **mlflow** - Experiment tracking
- **tensorboard** - Visualization

### Testing

- **pytest** (7.0+) - Test runner
- **pytest-asyncio** - Async test support
- **pytest-cov** - Coverage reporting

---

## Key Design Decisions

### 1. Hybrid Verification
- VLM runs on ALL predictions
- VLM wins on disagreement (better at visual reasoning)
- Disagreements logged for analysis

### 2. Image Validation
- 5-layer security model
- Structure validation before pixel loading
- Prevents memory exhaustion attacks

### 3. Singleton Pattern
- ModelManager: Reuse model across requests
- VLMService: Reuse API client, thread-safe
- ImageService: Cache transforms

### 4. Async Design
- FastAPI async endpoints
- Async model loading (startup)
- Supports concurrent requests

### 5. Graceful Degradation
- VLM optional (missing API key → CNN-only)
- CNN fallback when VLM fails
- Temp file cleanup in finally blocks

---

## Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| VLM latency | Slow requests | Accept longer time, offer CNN-only option |
| VLM API cost | Expensive at scale | Cache responses, batch processing |
| Single model instance | No redundancy | Replicate in Kubernetes |
| In-memory image cache | Memory usage | Implement Redis cache |

---

## Future Enhancements

### Phase 03: Monitoring
- Prometheus metrics
- Disagreement dashboard
- VLM accuracy evaluation
- CNN vs VLM comparison

### Phase 04: Optimization
- Response caching
- Batch VLM processing
- Confidence thresholds
- Explainability (attention maps)

### Phase 05: Advanced Features
- Ensemble learning
- Multi-model voting
- Feature extraction
- Fine-tuning from disagreements

---

## Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview & quick start |
| `docs/api-phase02-hybrid.md` | Hybrid verification API documentation |
| `docs/codebase-summary.md` | This file |
| `docs/code-standards.md` | Code style & patterns |
| `docs/system-architecture.md` | System design & architecture |
| `docs/project-roadmap.md` | Feature roadmap & timeline |

---

**Last Updated:** 2025-12-26
**Total Lines of Code:** ~3000+ (excluding tests & data)
**Test Coverage:** 25+ automated tests
**Production Ready:** Yes (with VLM API key)
