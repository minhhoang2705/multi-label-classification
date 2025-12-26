# API Quick Reference - Phase 02 Hybrid Verification

**Updated:** 2025-12-26
**New Endpoint:** POST /api/v1/predict/verified

---

## Quick Start

### Python Example (Requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict/verified",
    files={"file": open("cat.jpg", "rb")}
)
result = response.json()

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence_level']}")  # high/medium/low
print(f"Status: {result['verification_status']}")   # verified/uncertain/cnn_only
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/predict/verified" \
  -F "file=@cat.jpg" | jq '.'
```

---

## Response Interpretation

### Status Meanings

| Status | Meaning | Confidence | Prediction Source |
|--------|---------|-----------|-------------------|
| `verified` | CNN + VLM agree | **HIGH** | CNN |
| `uncertain` | CNN + VLM disagree | **MEDIUM** | **VLM** (wins!) |
| `cnn_only` | VLM disabled/unavailable | **LOW** | CNN |
| `error` | VLM API failed | **LOW** | CNN (fallback) |
| `unclear` | VLM response unparseable | **LOW** | CNN (fallback) |

### How to Use `verification_status`

```python
result = response.json()

if result['verification_status'] == 'verified':
    # High confidence - both models agree
    print(f"✓ {result['predicted_class']} (high confidence)")

elif result['verification_status'] == 'uncertain':
    # Medium confidence - disagreement, VLM wins
    print(f"? CNN: {result['cnn_prediction']}, VLM: {result['vlm_prediction']}")
    print(f"  Using VLM: {result['vlm_reasoning']}")

elif result['verification_status'] == 'cnn_only':
    # Low confidence - VLM unavailable
    print(f"! CNN only (VLM unavailable): {result['predicted_class']}")
```

---

## Response Fields

### Essential Fields

```python
result = {
    'predicted_class': 'Persian',           # Final prediction
    'confidence_level': 'high',             # high/medium/low
    'verification_status': 'verified',      # verified/uncertain/cnn_only/error
}
```

### CNN Details

```python
result = {
    'cnn_prediction': 'Persian',            # Top-1 prediction
    'cnn_confidence': 0.94,                 # 0.0-1.0
    'top_5_predictions': [                  # All top-5 with confidence
        {
            'rank': 1,
            'class_name': 'Persian',
            'class_id': 44,
            'confidence': 0.94
        },
        # ... 4 more
    ]
}
```

### VLM Details (if available)

```python
result = {
    'vlm_prediction': 'Persian',            # VLM's prediction
    'vlm_reasoning': 'Long fluffy coat...'  # Explanation
}
```

### Timing

```python
result = {
    'cnn_time_ms': 12.5,        # CNN inference
    'vlm_time_ms': 892.3,       # VLM inference (null if disabled)
    'total_time_ms': 905.0      # Total end-to-end
}
```

---

## Common Scenarios

### Scenario 1: Clear Agreement

```json
{
  "predicted_class": "Persian",
  "confidence_level": "high",
  "verification_status": "verified",
  "cnn_confidence": 0.94,
  "vlm_prediction": "Persian",
  "vlm_reasoning": "Flat face, long fluffy coat..."
}
```

**Interpretation:** ✅ Confident prediction

### Scenario 2: Disagreement (VLM Wins)

```json
{
  "predicted_class": "Himalayan",
  "confidence_level": "medium",
  "verification_status": "uncertain",
  "cnn_prediction": "Persian",
  "cnn_confidence": 0.87,
  "vlm_prediction": "Himalayan",
  "vlm_reasoning": "Color point pattern indicates Himalayan..."
}
```

**Interpretation:** ⚠️ Visual evidence suggests different breed, use VLM prediction

### Scenario 3: VLM Unavailable

```json
{
  "predicted_class": "Bengal",
  "confidence_level": "low",
  "verification_status": "cnn_only",
  "cnn_confidence": 0.78,
  "vlm_prediction": null,
  "vlm_reasoning": null,
  "vlm_time_ms": null
}
```

**Interpretation:** ⚠️ VLM not available, using CNN only (lower confidence)

---

## Error Handling

### Invalid Image (400)

```python
try:
    response = requests.post(url, files={"file": f})
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 400:
        # Invalid MIME type, corrupted, or bad dimensions
        print(f"Bad image: {e.response.json()['detail']}")
```

### Image Too Large (413)

```python
# File exceeds 10MB limit
print(f"Error: {response.json()['detail']}")  # "File too large..."
```

### Model Not Ready (503)

```python
# Called during startup before model loads
print(f"Model loading: {response.json()['detail']}")
```

---

## Configuration

### Enable/Disable VLM

```bash
# VLM enabled (default)
export ZAI_API_KEY=your_key_here

# VLM disabled (fallback to CNN-only)
# Unset ZAI_API_KEY, system gracefully degrades
```

### Accepted Image Formats

- ✅ JPEG (.jpg, .jpeg)
- ✅ PNG (.png)
- ✅ WebP (.webp)
- ❌ SVG, GIF, BMP, TIFF (rejected)

### Image Size Limits

- Min: 16x16 pixels
- Max: 10000x10000 pixels
- Max file: 10 MB

---

## Performance Tips

### Timing Budget

```
Image upload + validation:     15-60ms
CNN inference:                 5-20ms
VLM inference:                 500-2000ms
─────────────────────────────────────
Total with VLM:                530-2100ms
Total without VLM (fallback):  30-105ms
```

### For Production

1. **Accept VLM latency** - It's slower but more accurate
2. **Use CNN-only fallback** - If VLM API has issues
3. **Monitor disagreement rate** - Should be <10% on clean data
4. **Scale horizontally** - Multiple API instances

---

## Monitoring Disagreements

### Check Logs

```bash
# Watch live disagreements
tail -f logs/disagreements.jsonl | jq '.'

# Count disagreements
wc -l logs/disagreements.jsonl

# Analyze patterns
jq '.cnn_prediction' logs/disagreements.jsonl | sort | uniq -c
```

### Parse Example

```python
import json

with open('logs/disagreements.jsonl') as f:
    for line in f:
        disagreement = json.loads(line)
        print(f"CNN: {disagreement['cnn_prediction']}")
        print(f"VLM: {disagreement['vlm_prediction']}")
        print(f"Reason: {disagreement['vlm_reasoning']}")
        print()
```

---

## Comparison: /predict vs /predict/verified

| Feature | /predict | /predict/verified |
|---------|----------|------------------|
| Inference Time | 30-105ms | 530-2100ms |
| Verification | None | CNN+VLM |
| VLM Support | No | Yes |
| Accuracy | Standard | Higher |
| Cost | Low | Medium |
| Use Case | Speed | Accuracy |

**Recommendation:**
- Use **/predict** for fast API (CNN-only)
- Use **/predict/verified** for better accuracy (CNN+VLM)

---

## Common Questions

### Q: Why does /predict/verified take so long?

**A:** VLM (Vision Language Model) API call takes 500-2000ms. This is slower but more accurate. Use /predict for speed.

### Q: Can I retry if VLM fails?

**A:** No need - system automatically falls back to CNN with `status="error"`. If you want VLM verification, retry with same image.

### Q: How do I get better predictions?

**A:**
1. Use /predict/verified (better accuracy via VLM)
2. Monitor disagreement logs for patterns
3. Report problematic breeds for dataset improvement

### Q: What if VLM API key is missing?

**A:** System gracefully degrades to CNN-only. Set `ZAI_API_KEY` env var to enable VLM.

### Q: Can I use this for batch processing?

**A:** Currently no batch endpoint. Process images sequentially or call /predict/verified for each image. Phase 04 will add batch support.

---

## API Status Codes

```
200 OK                          Successful prediction
400 Bad Request                 Invalid image (MIME, size, format)
413 Payload Too Large          Image file or pixel data too large
500 Internal Server Error      Model inference failure
503 Service Unavailable        Model not loaded during startup
```

---

## Example Workflows

### Workflow 1: Accept Verified Predictions Only

```python
def get_verified_breed(image_file):
    response = requests.post(
        "http://localhost:8000/api/v1/predict/verified",
        files={"file": image_file}
    )
    result = response.json()

    if result['verification_status'] == 'verified':
        return result['predicted_class']
    else:
        return None  # Requires manual review
```

### Workflow 2: Use VLM Disagreements to Improve Training

```python
# Analyze disagreement log
import json
import pandas as pd

disagreements = []
with open('logs/disagreements.jsonl') as f:
    for line in f:
        disagreements.append(json.loads(line))

df = pd.DataFrame(disagreements)

# Which breeds have most disagreements?
problem_breeds = df['cnn_prediction'].value_counts().head(5)
print(f"Retrain on: {problem_breeds.index.tolist()}")
```

### Workflow 3: Implement Confidence Threshold

```python
def predict_with_threshold(image_file, threshold=0.90):
    response = requests.post(
        "http://localhost:8000/api/v1/predict/verified",
        files={"file": image_file}
    )
    result = response.json()

    # CNN confidence check
    if result['cnn_confidence'] < threshold:
        return result  # Flag as uncertain

    return result
```

---

## Troubleshooting

### "Model not loaded" (503)

```
Cause: Called during startup
Fix:   Wait for model loading to complete (~10-30 seconds)
```

### "Invalid file type" (400)

```
Cause: Image format not JPEG/PNG/WebP
Fix:   Convert image: ffmpeg -i input.gif -qscale:v 2 output.jpg
```

### "File too large" (413)

```
Cause: Image > 10MB
Fix:   Compress: convert input.jpg -resize 50% output.jpg
```

### VLM timeouts

```
Cause: Z.ai API slow or queue
Fix:   Retry with exponential backoff, or use CNN-only fallback
```

---

**Quick Links:**
- Full API Docs: `docs/api-phase02-hybrid.md`
- Codebase Overview: `docs/codebase-summary.md`
- Start API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- OpenAPI Docs: http://localhost:8000/docs
