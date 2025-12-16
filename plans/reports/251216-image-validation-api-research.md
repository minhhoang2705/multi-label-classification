# Image Validation & Processing for API Endpoints

## 1. Image Validation Strategy

**Multi-layer validation (CRITICAL):**
- Content-Type header check (not sufficient alone)
- Magic byte verification (file content)
- Dimension validation (max 10000x10000 to prevent pixel flood attacks)
- File size limits (max 50MB for content scanners)

```python
from fastapi import UploadFile, HTTPException
from PIL import Image
import io

ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_DIMENSIONS = (10000, 10000)
MIN_DIMENSIONS = (16, 16)

async def validate_image(file: UploadFile) -> Image.Image:
    # 1. Check MIME type
    if file.content_type not in ALLOWED_MIMES:
        raise HTTPException(400, "Invalid MIME type")

    # 2. Read & check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")

    # 3. Validate with PIL (verifies magic bytes + corruption)
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()  # Call immediately after open()

        # Reopen for actual processing (verify() closes file)
        img = Image.open(io.BytesIO(content))
        img.load()  # Force load to catch lazy-loading corruption
    except Exception as e:
        raise HTTPException(400, f"Invalid/corrupted image: {str(e)}")

    # 4. Validate dimensions
    width, height = img.size
    if not (MIN_DIMENSIONS[0] <= width <= MAX_DIMENSIONS[0] and
            MIN_DIMENSIONS[1] <= height <= MAX_DIMENSIONS[1]):
        raise HTTPException(400, "Image dimensions out of range")

    return img
```

## 2. PIL/Pillow Best Practices

**Error Handling:**
- Use `Image.verify()` immediately after `Image.open()`
- Call `img.load()` separately to catch deferred corruption
- Catches: IOError, SyntaxError, UnidentifiedImageError
- Handle PNG/JPEG differences (PNG verify() may miss some corruptions)

**Memory Management:**
- Use `io.BytesIO()` for file-like objects
- Call `.close()` or use context manager
- Load only needed frames for multi-frame images
- Convert to RGB early for consistency

```python
# Safe image loading pattern
from PIL import Image, UnidentifiedImageError
import io

try:
    with Image.open(io.BytesIO(content)) as img:
        img.verify()
except (IOError, SyntaxError, UnidentifiedImageError) as e:
    raise ValueError(f"Corrupted image: {e}")

# Re-open for processing
img = Image.open(io.BytesIO(content))
img = img.convert("RGB")  # Normalize format
img.load()  # Ensure data is loaded
```

## 3. PyTorch Transform Pipeline

**Standard preprocessing for inference:**
```python
import torchvision.transforms as T
from torchvision.transforms.functional import pil_to_tensor
import torch

# Option 1: Using Compose (recommended)
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),  # Converts (H,W,C) uint8 [0-255] → (C,H,W) float32 [0-1]
    T.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

# Option 2: Manual control (for custom pipelines)
img_tensor = pil_to_tensor(img)  # Shape: (C, H, W), dtype: uint8
img_tensor = img_tensor.float() / 255.0  # Normalize to [0-1]
img_tensor = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)(img_tensor)

# Add batch dimension
batch = img_tensor.unsqueeze(0)  # Shape: (1, C, H, W)
```

## 4. Multi-Format Support

**Format detection & handling:**
```python
def process_image(img: Image.Image) -> torch.Tensor:
    # Normalize all formats to RGB
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    elif img.mode == "L":  # Grayscale
        img = img.convert("RGB")  # Replicate across channels

    # Detect format from mime
    fmt = img.format  # "JPEG", "PNG", "WEBP"

    return preprocess(img)

# Supported: JPEG (lossy), PNG (lossless), WebP (both)
# Note: Animated formats need frame selection
```

## 5. Complete FastAPI Endpoint Example

```python
from fastapi import FastAPI, UploadFile, HTTPException, File
import torch

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Validate
        img = await validate_image(file)

        # 2. Preprocess
        img_tensor = preprocess(img)
        batch = img_tensor.unsqueeze(0)

        # 3. Inference
        with torch.no_grad():
            output = model(batch)

        return {"predictions": output.tolist()}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")
```

## 6. Security Checklist

- [ ] Validate MIME type (not sufficient alone)
- [ ] Check magic bytes (PIL does this via Image.open())
- [ ] Verify image integrity with `img.verify()` + `img.load()`
- [ ] Enforce dimension limits (prevent pixel floods)
- [ ] Enforce file size limits (max 50MB)
- [ ] Use whitelist for allowed formats
- [ ] Sanitize filenames (not needed for in-memory processing)
- [ ] Handle all PIL exceptions
- [ ] Test with corrupted images
- [ ] Monitor resource usage during load()

## 7. Error Handling Strategies

```python
# Comprehensive error handling
try:
    img = Image.open(io.BytesIO(content))
    img.verify()
    img = Image.open(io.BytesIO(content))
    img.load()
except FileNotFoundError:
    raise HTTPException(404, "Image file not found")
except UnidentifiedImageError:
    raise HTTPException(400, "Not a valid image format")
except IOError as e:
    raise HTTPException(400, f"Corrupted image: {e}")
except MemoryError:
    raise HTTPException(413, "Image too large to process")
except Exception as e:
    raise HTTPException(500, f"Unexpected error: {e}")
```

## Key Takeaways

1. **Never trust file extension** - validate actual content
2. **Use PIL.verify() + load()** - catches ~99% of corrupted images
3. **Enforce all limits** - dimensions, file size, aspect ratio
4. **Normalize formats** - convert to RGB early
5. **Use torchvision.transforms.Compose** - DRY, reusable, consistent
6. **Handle exceptions explicitly** - memory, corruption, format errors

---

## Sources

- [FastAPI File Upload Validation](https://medium.com/@jayhawk24/upload-files-in-fastapi-with-file-validation-787bd1a57658)
- [OWASP File Upload Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html)
- [Pillow Image Verification](https://github.com/python-pillow/Pillow/issues/6342)
- [TorchVision Transforms Documentation](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html)
- [File Upload Security Best Practices](https://betterstack.com/community/guides/scaling-python/uploading-files-using-fastapi/)
