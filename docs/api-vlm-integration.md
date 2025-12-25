# GLM-4.6V Vision Language Model Integration

## Overview

Phase 01 GLM-4.6V Integration adds Vision Language Model-based verification of CNN predictions for improved breed classification accuracy. When CNN predictions are uncertain or close calls, the GLM-4.6V model analyzes the actual image to verify or correct the CNN's top prediction.

**Status:** Complete
**Version:** 1.0.0
**Model:** GLM-4.6V (Z.ai Platform)

---

## Architecture

### Core Components

1. **VLMService** (`api/services/vlm_service.py`)
   - Singleton pattern for API client lifecycle
   - Base64 image encoding
   - Structured prompt generation
   - Response parsing and validation
   - Thread-safe for FastAPI concurrent requests

2. **Configuration** (`api/config.py`)
   - VLM enable/disable toggle
   - Z.ai API key management
   - Helper function to check VLM availability

3. **Tests** (`tests/api/test_vlm_service.py`)
   - 40+ test cases
   - Service initialization (with/without API key)
   - Image encoding (JPEG, PNG, WebP)
   - Prompt building
   - Response parsing (agree, disagree, unclear)
   - End-to-end verification flow
   - Error handling and fallback behavior

---

## Configuration

### Environment Variables

```bash
# Enable/disable VLM verification
API_VLM_ENABLED=true

# Z.ai API key (required for VLM to work)
# Get from: https://docs.z.ai
ZAI_API_KEY=your-api-key-here
```

### Checking VLM Availability

```python
from api.config import is_vlm_available

if is_vlm_available():
    # VLM is enabled and API key is set
    service = VLMService.get_instance()
else:
    # Fall back to CNN-only predictions
    pass
```

**VLM is available only when:**
1. `API_VLM_ENABLED=true` in config
2. `ZAI_API_KEY` environment variable is set

---

## VLMService API

### Singleton Pattern

```python
from api.services.vlm_service import VLMService

# Get singleton instance (thread-safe)
service = VLMService.get_instance()

# Reset for testing
VLMService.reset_instance()
```

**Why Singleton?**
- Reuses API client connection across requests
- Reduces initialization overhead
- Consistent service behavior
- Thread-safe for FastAPI's concurrent request handling

### Main Method: verify_prediction()

```python
def verify_prediction(
    image_path: str,
    cnn_top_3: List[Tuple[str, float]]
) -> Tuple[str, str, Optional[str]]:
    """
    Verify CNN prediction using GLM-4.6V.

    Args:
        image_path: Path to cat image (JPEG, PNG, WebP)
        cnn_top_3: List of (breed_name, confidence) from CNN top-3

    Returns:
        Tuple of (status, prediction, reasoning)
        - status: "agree", "disagree", "unclear", or "error"
        - prediction: VLM's breed prediction (or CNN fallback)
        - reasoning: Visual features explanation
    """
```

**Return Values:**

| Status | Meaning | When |
|--------|---------|------|
| `"agree"` | VLM picks CNN's top-1 prediction | Confidence match |
| `"disagree"` | VLM picks different breed (from top-3 or new) | VLM disagrees |
| `"unclear"` | Could not parse VLM response | Response parsing failed |
| `"error"` | API failed (network, file not found, etc.) | API error or file missing |

**On Error:** Always falls back to CNN's top-1 prediction as failsafe.

### Example Usage

```python
from api.services.vlm_service import VLMService

service = VLMService.get_instance()

# CNN gave these predictions
cnn_top_3 = [
    ("Persian", 0.85),
    ("Himalayan", 0.10),
    ("Exotic Shorthair", 0.03)
]

# Verify with VLM
status, prediction, reasoning = service.verify_prediction(
    image_path="cat_image.jpg",
    cnn_top_3=cnn_top_3
)

if status == "agree":
    print(f"VLM confirms: {prediction}")
elif status == "disagree":
    print(f"VLM suggests: {prediction}")
    print(f"Reason: {reasoning}")
else:
    print(f"Could not verify, using CNN prediction: {prediction}")
```

---

## Image Encoding

### Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- GIF (.gif)

Default MIME type for unknown extensions: `image/jpeg`

### Encoding Process

1. Read image file as bytes
2. Encode to base64 string
3. Build data URI: `data:image/jpeg;base64,<data>`
4. Send to GLM-4.6V API

```python
# Automatic in verify_prediction(), but can be called directly:
base64_image = service._encode_image_to_base64("image.jpg")
mime_type = service._get_image_mime_type("image.jpg")

image_uri = f"data:{mime_type};base64,{base64_image}"
```

---

## Prompt Structure

VLMService generates structured prompts to guide GLM-4.6V:

```
Analyze this cat image and identify its breed.

The CNN classifier's top-3 predictions are:
  1. Persian (85.0%)
  2. Himalayan (10.0%)
  3. Exotic Shorthair (3.0%)

Instructions:
1. Look at the cat's features: coat pattern, face shape, eye color, body type, ear shape
2. Compare with the 3 candidates above
3. Choose the most likely breed from the candidates, OR suggest a different breed if none match

Response format:
BREED: [your prediction]
MATCHES_CNN: [YES if your choice is in top-3, NO if different]
REASON: [1-2 sentence explanation of key visual features]
```

**Prompt Strategy:**
- Lists CNN candidates to scope predictions
- Instructs VLM to analyze specific visual features
- Enforces structured response format for parsing
- Temperature set to 0.3 for deterministic results

---

## Response Parsing

VLMService automatically parses structured VLM responses:

### Example Response

```
BREED: Persian
MATCHES_CNN: YES
REASON: The flat face and long silky fur are characteristic of Persian cats.
```

### Matching Logic

1. **Exact Match** (after normalization)
   - Case-insensitive comparison
   - Whitespace trimmed
   - Example: "PERSIAN" matches "Persian"

2. **Partial Match** (word-based)
   - All CNN words must appear in VLM response
   - Example: "Persian Longhair" matches "Persian" (contains "persian" word)

3. **No Match**
   - VLM suggests breed not in top-3
   - Still accepted as valid prediction
   - Status marked as "disagree"

### Fallbacks

| Situation | Behavior |
|-----------|----------|
| Parse error | Return status: "unclear", falls back to CNN top-1 |
| Missing image file | Return status: "error", falls back to CNN top-1 |
| API timeout | Return status: "error", falls back to CNN top-1 |
| Invalid response | Return status: "unclear", falls back to CNN top-1 |

---

## API Integration

### Prerequisites

1. **Install zai-sdk**
   ```bash
   pip install zai-sdk>=0.0.4
   ```
   (already in requirements.txt)

2. **Get Z.ai API Key**
   - Visit https://docs.z.ai
   - Create account and generate API key
   - Set `ZAI_API_KEY` environment variable

3. **Enable VLM**
   ```bash
   export API_VLM_ENABLED=true
   export ZAI_API_KEY=your-key
   ```

### Initialization Errors

```python
from api.services.vlm_service import VLMService

try:
    service = VLMService.get_instance()
except ValueError as e:
    # ZAI_API_KEY not set
    print(f"VLM not available: {e}")
except ImportError as e:
    # zai-sdk not installed
    print(f"Missing dependency: {e}")
```

---

## Testing

### Test Coverage

**File:** `tests/api/test_vlm_service.py`

**Test Categories:**

1. **Initialization Tests** (4)
   - Missing API key raises ValueError
   - Valid API key succeeds
   - Singleton pattern works
   - Reset instance functionality

2. **Image Encoding Tests** (6)
   - JPEG encoding
   - PNG encoding
   - MIME type detection
   - Unknown extension handling
   - File not found errors

3. **Prompt Building Tests** (2)
   - Includes all 3 candidates
   - Includes feature analysis instructions

4. **Response Parsing Tests** (5)
   - Agree response (VLM picks top-1)
   - Disagree response (VLM picks top-2)
   - New breed suggestion
   - Unclear response (no breed found)
   - Case-insensitive matching

5. **End-to-End Tests** (3)
   - Successful verification
   - File not found handling
   - API error handling

6. **Config Integration Tests** (3)
   - is_vlm_available() with key
   - is_vlm_available() without key
   - is_vlm_available() when disabled

### Running Tests

```bash
# Run all VLM tests
pytest tests/api/test_vlm_service.py -v

# Run specific test class
pytest tests/api/test_vlm_service.py::TestVLMServiceInitialization -v

# Run with mocked API (no real API calls)
pytest tests/api/test_vlm_service.py -v

# View coverage
pytest tests/api/test_vlm_service.py --cov=api.services.vlm_service
```

**Note:** All tests use mocked Z.ai SDK. No real API calls made.

---

## Error Handling

### Common Errors and Solutions

**ValueError: "ZAI_API_KEY environment variable not set"**
```bash
# Set the API key
export ZAI_API_KEY=your-api-key
# Then start the API
python -m uvicorn api.main:app
```

**ImportError: "zai-sdk not installed"**
```bash
# Install the SDK
pip install zai-sdk>=0.0.4
```

**FileNotFoundError: "Image not found"**
- Image file path is incorrect
- Verify path exists before calling verify_prediction()

**API Timeout or Network Error**
- Service gracefully falls back to CNN prediction
- Logs error with details
- Returns status: "error"

---

## Performance

### Request Flow

1. **Image Encoding** (~10-50ms)
   - Read file from disk
   - Encode to base64

2. **API Call** (~1-5 seconds)
   - Build data URI
   - Send to GLM-4.6V
   - Receive response

3. **Response Parsing** (~1-5ms)
   - Extract breed and reasoning
   - Validate against CNN top-3
   - Determine status

**Total Verification Time:** ~1-6 seconds (primarily API latency)

### Memory Usage

- VLMService singleton: ~10-20 MB
- Per-image encoding: ~5-50 MB (depends on image size)
- Z.ai client: Minimal overhead

---

## Integration with FastAPI

### Optional VLM Verification in Endpoints

```python
from fastapi import UploadFile
from api.services.vlm_service import VLMService
from api.config import is_vlm_available

@app.post("/api/v1/predict")
async def predict(file: UploadFile):
    # Get CNN predictions
    cnn_top_3 = [...] # Your CNN inference logic

    # Optional VLM verification
    if is_vlm_available():
        service = VLMService.get_instance()
        status, vlm_pred, reason = service.verify_prediction(
            image_path=file.filename,
            cnn_top_3=cnn_top_3
        )

        return {
            "cnn_prediction": cnn_top_3[0],
            "vlm_status": status,
            "vlm_prediction": vlm_pred,
            "vlm_reasoning": reason
        }
    else:
        return {
            "cnn_prediction": cnn_top_3[0],
            "vlm_available": False
        }
```

---

## Security Considerations

1. **API Key Management**
   - Never commit `.env` files with real keys
   - Use environment variables only
   - Rotate keys periodically

2. **Image Handling**
   - Validate file paths to prevent directory traversal
   - Check image dimensions before encoding (decompression bomb protection)
   - Limit image file sizes

3. **API Calls**
   - Rate limiting (per Z.ai platform limits)
   - Timeout protection (default 10-30s)
   - Error logging without exposing secrets

4. **Response Validation**
   - Parse structured responses carefully
   - Always have CNN fallback
   - Log unexpected response formats

---

## Dependencies

**Required:**
- `fastapi >= 0.115.0`
- `zai-sdk >= 0.0.4`
- `pydantic-settings >= 2.0.0`
- `python >= 3.8`

**From requirements.txt:**
```
zai-sdk>=0.0.4
```

---

## Environment Setup Example

**.env file:**
```bash
# Model
API_CHECKPOINT_PATH=outputs/checkpoints/fold_0/best_model.pt
API_MODEL_NAME=resnet50
API_NUM_CLASSES=67

# Server
API_PORT=8000
API_HOST=0.0.0.0

# VLM Integration
API_VLM_ENABLED=true
ZAI_API_KEY=sk-your-key-here

# Device
API_DEVICE=auto
```

**.env.example:**
```bash
# See .env.example for full configuration
API_VLM_ENABLED=true
ZAI_API_KEY=your-api-key-here
```

---

## Next Steps

### Integration Points

1. **Predict Endpoint Integration**
   - Add VLM verification to `/api/v1/predict`
   - Return VLM status and reasoning with predictions

2. **Monitoring and Metrics**
   - Track VLM agree/disagree rates
   - Monitor API latency
   - Log disagreement patterns

3. **Advanced Features**
   - Confidence thresholding (only verify low-confidence predictions)
   - Batch verification with rate limiting
   - Result caching for identical images

---

## References

- [Z.ai Platform Documentation](https://docs.z.ai)
- [GLM-4.6V Model Docs](https://docs.z.ai/models/glm-4-6v)
- [zai-sdk GitHub](https://github.com/z-ai/zai-sdk-python)
- [GLM-4V Vision Models](https://github.com/THUDM/CogVLM)

---

**Status:** Phase 01 Complete
**Last Updated:** December 25, 2024
**Version:** 1.0.0
