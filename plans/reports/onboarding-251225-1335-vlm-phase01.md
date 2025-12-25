# Onboarding Guide: VLM Phase 01 Integration

**Phase:** GLM-4.6V API Integration
**Date:** 2025-12-25
**Status:** Complete

## What Changed

New VLM (Vision Language Model) service added for cat breed verification using GLM-4.6V.

### Files Added
- `api/services/vlm_service.py` - VLM service implementation
- `tests/api/test_vlm_service.py` - Comprehensive unit tests

### Files Modified
- `api/config.py` - Added VLM configuration
- `requirements.txt` - Added zai-sdk dependency
- `.env.example` - Added VLM config section

## Setup Requirements

### 1. Install Dependencies

```bash
source .venv/bin/activate
uv pip install "zai-sdk>=0.0.4"
```

### 2. Get Z.ai API Key

1. Visit https://docs.z.ai
2. Sign up and create API key
3. Copy your API key

### 3. Configure Environment

Add to your `.env` file:

```bash
# VLM Configuration
API_VLM_ENABLED=true
ZAI_API_KEY=your-actual-api-key-here
```

**Important:** Never commit `.env` - API keys are sensitive!

### 4. Verify Setup

```bash
# Check VLM service imports correctly
python -c "from api.services.vlm_service import VLMService; print('✓ VLM service ready')"

# Check config
python -c "from api.config import is_vlm_available; print(f'VLM available: {is_vlm_available()}')"
```

Expected output when API key is set:
```
✓ VLM service ready
VLM available: True
```

## Usage Example

```python
from api.services.vlm_service import VLMService

# Get singleton instance
vlm = VLMService.get_instance()

# Verify CNN prediction
status, prediction, reasoning = vlm.verify_prediction(
    image_path="data/images/Persian/sample.jpg",
    cnn_top_3=[
        ("Persian", 0.92),
        ("Himalayan", 0.05),
        ("Exotic Shorthair", 0.02)
    ]
)

print(f"Status: {status}")  # "agree", "disagree", or "error"
print(f"VLM Prediction: {prediction}")
print(f"Reasoning: {reasoning}")
```

## Running Without VLM (Optional)

To disable VLM and use CNN only:

```bash
# In .env
API_VLM_ENABLED=false
```

Or remove `ZAI_API_KEY` from environment.

## Testing

```bash
# Run VLM service tests
python -m pytest tests/api/test_vlm_service.py -v

# Expected: 23/23 tests pass
```

## Troubleshooting

### "ZAI_API_KEY environment variable not set"
- Check `.env` file exists
- Verify `ZAI_API_KEY` is set correctly
- Restart API server after changing `.env`

### "ModuleNotFoundError: No module named 'zai'"
```bash
uv pip install "zai-sdk>=0.0.4"
```

### VLM always returns "error" status
- Check API key is valid
- Check network connectivity
- Review logs for detailed error messages

## Next Steps

Phase 02 will implement disagreement resolution strategy when VLM disagrees with CNN.

## Cost Considerations

Each VLM verification:
- Makes 1 API call to Z.ai GLM-4.6V
- Typical response time: 1-3 seconds
- Check Z.ai pricing at https://docs.z.ai/pricing

Consider:
- Caching results for same images
- Rate limiting for high-traffic scenarios
- Budget monitoring

## Security Notes

- API keys stored in environment variables (not code)
- `.env` file excluded via `.gitignore`
- Tests use mocked API (no real API calls)
- Thread-safe singleton for concurrent requests
