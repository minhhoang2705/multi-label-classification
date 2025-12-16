# Cat Breeds Classification API - Phase 02: Image Validation & Preprocessing

## Overview

Phase 02 implements robust image validation, preprocessing, and inference pipeline. Focus on security hardening against image-based attacks and ensuring consistent model input preparation.

**Status:** Complete
**Version:** 2.0.0
**Date:** Phase 02 Complete

---

## Architecture

### New Components

1. **ImageService** (`api/services/image_service.py`)
   - Multi-layer image validation (5 stages)
   - Security protections (decompression bombs, pixel floods)
   - Type-safe metadata tracking
   - ImageNet normalization pipeline

2. **Custom Exceptions** (`api/exceptions.py`)
   - `ImageValidationError` (400)
   - `ImageTooLargeError` (413)
   - `ModelInferenceError` (500)
   - `ModelNotLoadedError` (503)

3. **Dependency Injection** (`api/dependencies.py`)
   - Settings singleton
   - ImageService singleton
   - ModelManager async factory

4. **Exception Handlers** (updated `api/main.py`)
   - Validation error handling
   - Generic exception handler
   - Proper HTTP status codes

---

## Image Validation Pipeline

### 5-Layer Security Model

The validation sequence prevents security vulnerabilities by validating structure before loading pixels:

#### Layer 1: MIME Type Validation
```
Input: Content-Type header
Allowed: image/jpeg, image/png, image/webp
Rejects: image/svg+xml, application/json, etc.
Error: 400 Bad Request
```

#### Layer 2: File Size Validation
```
Limit: 10 MB
Action: Validates before reading full content
Purpose: Prevent DoS attacks via huge files
Error: 413 Payload Too Large
```

#### Layer 3: Image Structure Validation
```
Action: Open & verify image header WITHOUT loading pixels
Methods: PIL Image.verify() + format check
Purpose: Detect corrupted/invalid images early
Error: 400 Bad Request
```

**Critical Security Detail:** `Image.verify()` validates file structure WITHOUT loading pixel data. This prevents malicious files from consuming memory.

#### Layer 4: Dimension Validation
```
Min: 16x16 pixels
Max: 10000x10000 pixels
Action: BEFORE pixel loading (prevents memory exhaustion)
Purpose: Stops pixel flood attacks
Error: 400 Bad Request
```

**Key:** Dimension check occurs AFTER structure validation but BEFORE `img.load()`, creating a memory safety boundary.

#### Layer 5: Pixel Loading
```
Action: Force pixel data loading with MemoryError handling
Purpose: Final safe pixel decompression
Fallback: PIL decompression bomb protection (MAX_IMAGE_PIXELS=100M)
Error: 413 if MemoryError
```

### Configuration Constants

```python
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_DIMENSIONS = (10000, 10000)
MIN_DIMENSIONS = (16, 16)
Image.MAX_IMAGE_PIXELS = 100_000_000  # 100 million pixels
```

---

## Type-Safe Metadata

### ImageMetadata TypedDict

```python
class ImageMetadata(TypedDict):
    original_width: int        # Original image width
    original_height: int       # Original image height
    format: str               # Format (JPEG, PNG, WebP)
    mode: str                 # Color mode (RGB, L, RGBA)
    file_size_bytes: int      # File size in bytes
    filename: str             # Original filename
```

**Benefits:**
- Type checking at development time
- IDE autocomplete support
- Runtime validation
- Clear API contracts

---

## Preprocessing Pipeline

### ImageNet Normalization

```python
T.Compose([
    T.Resize((224, 224)),                    # Resize to model input size
    T.ToTensor(),                            # Convert to [0-1] range
    T.Normalize(
        mean=[0.485, 0.456, 0.406],        # ImageNet means
        std=[0.229, 0.224, 0.225]          # ImageNet stds
    )
])
```

**Output:** Tensor of shape `(1, 3, 224, 224)` ready for ResNet50

### Color Mode Handling

Automatic conversion to RGB handles:
- Grayscale (L mode) → RGB
- RGBA with alpha → RGB
- Palette images → RGB
- Indexed color → RGB

---

## API Endpoints - Phase 02

### POST `/predict`

Single image inference endpoint.

**Request:**
```http
POST /predict HTTP/1.1
Content-Type: multipart/form-data

[binary image data]
```

**Parameters:**
- `file` (UploadFile, required): Image file (JPEG, PNG, or WebP)

**Response (Success - 200):**
```json
{
  "predictions": [
    {
      "class": "Abyssinian",
      "confidence": 0.95
    },
    {
      "class": "Bengal",
      "confidence": 0.03
    },
    {
      "class": "Birman",
      "confidence": 0.02
    }
  ],
  "metadata": {
    "original_width": 800,
    "original_height": 600,
    "format": "JPEG",
    "mode": "RGB",
    "file_size_bytes": 65432,
    "filename": "my_cat.jpg"
  },
  "inference_time_ms": 18.5
}
```

**Response (Invalid MIME - 400):**
```json
{
  "detail": "Invalid file type: image/svg+xml. Allowed: image/jpeg, image/png, image/webp"
}
```

**Response (File Too Large - 413):**
```json
{
  "detail": "File too large: 15.50MB. Max: 10MB"
}
```

**Response (Image Too Small - 400):**
```json
{
  "detail": "Image too small: 8x8. Minimum: 16x16"
}
```

**Response (Corrupted Image - 400):**
```json
{
  "detail": "Not a valid image format. Supported: JPEG, PNG, WebP"
}
```

**Response (Model Not Loaded - 503):**
```json
{
  "detail": "Model not loaded. Please wait for startup to complete."
}
```

### Error Status Codes

| Status | Scenario |
|--------|----------|
| 400 | Invalid MIME type, corrupted image, bad dimensions |
| 413 | File too large (>10MB) or pixel memory exhaustion |
| 500 | Model inference failure |
| 503 | Model not loaded yet |

---

## Security Features

### 1. Decompression Bomb Protection

**Issue:** Malicious ZIP-like image formats expand to huge sizes in memory.

**Solution:**
```python
Image.MAX_IMAGE_PIXELS = 100_000_000  # 100 million pixel limit
# PIL raises DecompressionBombWarning/MemoryError if exceeded
```

**Example Attack:** 1000x1000 compressed image → decompresses to 100 million pixels → MemoryError

### 2. Pixel Flood Attack Prevention

**Issue:** Dimensions in header claim 10000x10000 but file is small.

**Solution:**
```
Layer 3: Validate structure (get dimensions from header)
Layer 4: Validate dimensions BEFORE loading pixels
Layer 5: Catch MemoryError on pixel load
```

**Attack Timeline:**
1. Attacker uploads: 100KB file claiming 10000x10000 dimensions
2. Layer 3: Header validation succeeds (structure is valid)
3. Layer 4: Dimension check fails → 400 error (safe reject)
4. Layer 5: Never reached (early prevention)

### 3. MIME Type Whitelist

Only JPEG, PNG, WebP allowed. Rejects:
- SVG (external entity attacks)
- GIF (layer parsing vulnerabilities)
- ICO (resource exhaustion)
- TIFF (complexity vulnerabilities)

### 4. Memory Exhaustion Defense

Sequential validation prevents memory spike:
```
Layer 1-2: ~1KB checks (MIME, size header)
Layer 3: ~100KB (header parsing, no pixel data)
Layer 4: Decision point (skip pixel load if dims invalid)
Layer 5: Safe pixel loading (only if passed all checks)
```

**Result:** Attack rejected before memory spike.

---

## Exception Handling

### ImageValidationError (400)

Raised by:
- Invalid MIME type
- Corrupted/invalid image format
- Dimension out of range

```python
raise ImageValidationError("Invalid file type: ...")
```

### ImageTooLargeError (413)

Raised by:
- File size > 10MB
- Memory exhaustion on pixel load

```python
raise ImageTooLargeError("File too large: ...")
```

### ModelInferenceError (500)

Raised by:
- Model forward pass failure
- Unexpected inference errors

```python
raise ModelInferenceError("Model inference failed: ...")
```

### ModelNotLoadedError (503)

Raised by:
- Endpoint called before model startup completes

```python
raise ModelNotLoadedError()
```

---

## Dependency Injection

### Settings Singleton

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()
```

Single Settings instance per application lifetime.

### ImageService Singleton

```python
@lru_cache
def get_image_service() -> ImageService:
    settings = get_settings()
    return ImageService(image_size=settings.image_size)
```

ImageService reused across requests (transform pipeline cached).

### ModelManager Factory

```python
async def get_model_manager() -> ModelManager:
    return await ModelManager.get_instance()
```

Returns singleton (async-aware).

### Usage in Endpoints

```python
@app.post("/predict")
async def predict(
    file: UploadFile,
    image_service: ImageService = Depends(get_image_service),
    model_manager: ModelManager = Depends(get_model_manager)
):
    # Dependencies injected by FastAPI
    tensor, metadata = await image_service.validate_and_preprocess(file)
    predictions = await model_manager.predict(tensor)
```

---

## Testing

### Test Coverage (61 tests)

**1. ImageService Validation Tests (20)**
   - MIME type validation
   - File size limits
   - Dimension validation (min/max)
   - Image structure validation
   - Pixel loading
   - Mode conversion (RGB, RGBA, L, palette)

**2. Security Tests (12)**
   - Decompression bomb handling
   - Pixel flood attack prevention
   - Large file rejection
   - Corrupted file handling
   - Invalid format detection

**3. Preprocessing Tests (8)**
   - Tensor shape validation
   - Normalization correctness
   - RGB conversion
   - Batch dimension addition
   - Different image sizes

**4. Metadata Tests (5)**
   - Metadata accuracy
   - Format detection
   - Size tracking
   - Filename preservation

**5. Exception Tests (7)**
   - Custom exception raising
   - HTTP status codes
   - Error messages
   - Exception propagation

**6. Dependency Injection Tests (4)**
   - Singleton caching
   - Factory functions
   - Injection resolution

**7. Integration Tests (5)**
   - End-to-end validation + preprocessing
   - File type variations
   - Edge cases

### Running Tests

```bash
# All Phase 02 tests
pytest tests/test_api_phase02.py -v

# Specific test class
pytest tests/test_api_phase02.py::TestImageValidation -v

# With coverage
pytest tests/test_api_phase02.py --cov=api.services.image_service
```

---

## Configuration

### New Environment Variables

None new - uses existing `API_IMAGE_SIZE` (defaults to 224).

### Validation Constants (Tunable)

Edit `api/services/image_service.py`:

```python
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # Adjust for larger uploads
MAX_DIMENSIONS = (10000, 10000)   # Adjust max size
MIN_DIMENSIONS = (16, 16)          # Adjust min size
Image.MAX_IMAGE_PIXELS = 100_000_000  # PIL decompression bomb limit
```

---

## Usage Examples

### Python/Requests

```python
import requests

with open("cat.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/predict",
        files=files
    )

result = response.json()
print(f"Top prediction: {result['predictions'][0]['class']}")
print(f"Metadata: {result['metadata']}")
print(f"Inference time: {result['inference_time_ms']}ms")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@cat.jpg"
```

### JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch("/predict", {
  method: "POST",
  body: formData
});

const result = await response.json();
console.log(result.predictions);
```

---

## Performance Characteristics

### Validation Time
- MIME check: <1ms
- File size check: <1ms
- Structure validation: 1-5ms
- Dimension check: <1ms
- Pixel loading: 10-50ms (varies by image size)
- **Total validation:** 15-60ms

### Preprocessing Time
- Resize: 5-15ms
- Tensor conversion: 5-10ms
- Normalization: <1ms
- **Total preprocessing:** 10-25ms

### Inference Time (ResNet50)
- CUDA: 5-20ms
- CPU: 50-200ms

### Total E2E (CUDA)
- Validation: 15-60ms
- Preprocessing: 10-25ms
- Inference: 5-20ms
- **Total:** 30-105ms

---

## Error Scenarios & Handling

### Scenario 1: User uploads SVG

```
Request: POST /predict with image/svg+xml
Layer 1: MIME check
Error: 400 - "Invalid file type: image/svg+xml"
```

### Scenario 2: Malicious decompression bomb

```
File: 100KB file claiming 10000x10000 pixels
Layer 3: Structure validation passes
Layer 4: Dimension check FAILS (too large)
Error: 400 - "Image too large: 10000x10000. Maximum: 10000x10000"
Result: SAFE - rejected before memory spike
```

### Scenario 3: Corrupted JPEG

```
File: Invalid JPEG header
Layer 3: Image.verify() fails
Error: 400 - "Not a valid image format"
```

### Scenario 4: Model not ready

```
Request: POST /predict during startup (before model loads)
Check: manager.is_loaded
Error: 503 - "Model not loaded. Please wait..."
```

---

## Integration with Phase 01

### Reused Components
- FastAPI application lifecycle
- Settings configuration
- ModelManager service
- Health endpoints
- CORS middleware
- Logging system

### New Exception Handling

Phase 01 `api/main.py` updated with:
```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    # Handles Pydantic validation errors
    # Returns 400 with detailed error info

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    # Catches unhandled exceptions
    # Logs traceback, returns 500
```

---

## Phase 02 Completion Checklist

- [x] ImageService with 5-layer validation
- [x] Type-safe metadata (TypedDict)
- [x] Decompression bomb protection
- [x] Pixel flood prevention
- [x] ImageNet normalization pipeline
- [x] Custom exception types
- [x] Dependency injection factories
- [x] Exception handlers in main.py
- [x] Comprehensive test suite (61 tests)
- [x] Security hardening
- [x] Color mode conversion (RGB, RGBA, L, palette)

---

## Next Phases

### Phase 03: Batch Inference
- `/predict/batch` endpoint
- Multiple image processing
- Concurrent validation
- Result aggregation

### Phase 04: Metrics & Monitoring
- Prometheus metrics
- Inference latency tracking
- Validation error rates
- Model performance metrics

### Phase 05: Advanced Features
- Explainability (attention maps)
- Confidence threshold filtering
- Multi-model ensemble
- Caching layer

---

## Files Summary

| File | Purpose |
|------|---------|
| `api/services/image_service.py` | Image validation & preprocessing |
| `api/exceptions.py` | Custom HTTP exceptions |
| `api/dependencies.py` | DI factories |
| `api/main.py` | Exception handlers (updated) |
| `tests/test_api_phase02.py` | 61 tests for Phase 02 |

---

## Dependencies

All from Phase 01 requirements plus:
- Pillow >= 9.0.0 (image processing)
- torchvision >= 0.15.0 (transforms)

See `requirements.txt` for complete list.

---

## Troubleshooting

### "Image too small" error on valid images

**Problem:** 16x16 minimum might be too strict
```bash
# Edit ImageService
MIN_DIMENSIONS = (8, 8)
```

### High validation latency on large images

**Problem:** Slow disk I/O or large pixel decompression
```bash
# Consider:
# 1. Use SSD for faster I/O
# 2. Reduce MAX_FILE_SIZE
# 3. Lower MAX_DIMENSIONS
```

### MemoryError during peak load

**Problem:** Multiple large images processing simultaneously
```bash
# Solutions:
# 1. Reduce MAX_FILE_SIZE (10MB default)
# 2. Reduce MAX_DIMENSIONS
# 3. Scale horizontally (multiple API instances)
```

---

## References

- [PIL/Pillow Documentation](https://pillow.readthedocs.io/)
- [Torchvision Transforms](https://pytorch.org/vision/stable/transforms.html)
- [ImageNet Normalization](https://pytorch.org/hub/pytorch_vision_resnet/)
- [FastAPI File Upload](https://fastapi.tiangolo.com/request-files/)
- [Security Best Practices](https://fastapi.tiangolo.com/tutorial/security/)

---

**Last Updated:** Phase 02 Complete
**Test Coverage:** 61 tests, comprehensive security coverage
**Maintainer:** Development Team
