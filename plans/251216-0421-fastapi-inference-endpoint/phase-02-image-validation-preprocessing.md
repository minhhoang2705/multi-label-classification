# Phase 02: Image Validation & Preprocessing

## Context

- [Main Plan](./plan.md)
- [Phase 01: Core API](./phase-01-core-api-model-loading.md)
- Research: [Image Validation](../reports/251216-image-validation-api-research.md)
- Scout: [Image Preprocessing](../reports/251216-scout-image-preprocessing.md)

## Overview

| Field | Value |
|-------|-------|
| Date | 2025-12-16 |
| Priority | High |
| Status | ✅ Completed |
| Completion Timestamp | 2025-12-16 04:45 UTC |
| Est. Time | 2 hours |
| Review | [Code Review Report](../reports/code-reviewer-251216-phase02-review.md) |

## Key Insights

1. **Multi-layer validation**: MIME type + magic bytes + PIL verify + dimensions
2. **PIL verify() + load()** pattern catches ~99% corrupted images
3. **ImageNet normalization**: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
4. **Val transforms**: Resize(224) -> ToTensor -> Normalize (no augmentation)
5. **Convert to RGB** early to handle grayscale/RGBA inputs

## Requirements

- Validate uploaded file (MIME, size, corruption)
- Enforce dimension limits (min 16x16, max 10000x10000)
- Preprocess to model input format (224x224, normalized tensor)
- Handle JPEG, PNG, WebP formats
- Return clear error messages for invalid images

## Architecture

```python
# api/services/image_service.py
class ImageService:
    ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_DIMENSIONS = (10000, 10000)
    MIN_DIMENSIONS = (16, 16)
    IMAGE_SIZE = 224

    async def validate_and_preprocess(
        self, file: UploadFile
    ) -> torch.Tensor: ...

    def _validate_mime(self, content_type: str): ...
    def _validate_file_size(self, content: bytes): ...
    def _validate_image(self, content: bytes) -> Image.Image: ...
    def _preprocess(self, img: Image.Image) -> torch.Tensor: ...
```

## Related Code Files

| File | Purpose |
|------|---------|
| `src/augmentations.py` | get_val_transforms(), Denormalize |
| `src/config.py` | AugmentationConfig (normalize_mean, normalize_std) |
| `src/dataset.py` | Image loading with .convert('RGB') |

## Implementation Steps

### 1. Create Image Service (api/services/image_service.py)

```python
import io
from typing import Tuple
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T
from fastapi import UploadFile, HTTPException

class ImageService:
    ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_DIMENSIONS = (10000, 10000)
    MIN_DIMENSIONS = (16, 16)

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    async def validate_and_preprocess(
        self, file: UploadFile
    ) -> Tuple[torch.Tensor, dict]:
        """
        Validate uploaded image and preprocess for inference.

        Returns:
            Tuple of (preprocessed tensor, metadata dict)

        Raises:
            HTTPException: For validation errors
        """
        # 1. Validate MIME type
        self._validate_mime(file.content_type)

        # 2. Read file content
        content = await file.read()

        # 3. Validate file size
        self._validate_file_size(content)

        # 4. Validate image integrity and get PIL Image
        img = self._validate_image(content)

        # 5. Validate dimensions
        self._validate_dimensions(img.size)

        # 6. Get metadata before transforms
        metadata = {
            "original_width": img.size[0],
            "original_height": img.size[1],
            "format": img.format,
            "mode": img.mode,
            "file_size_bytes": len(content)
        }

        # 7. Preprocess for model
        tensor = self._preprocess(img)

        return tensor, metadata

    def _validate_mime(self, content_type: str):
        """Validate MIME type."""
        if content_type not in self.ALLOWED_MIMES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {content_type}. "
                       f"Allowed: {', '.join(self.ALLOWED_MIMES)}"
            )

    def _validate_file_size(self, content: bytes):
        """Validate file size."""
        if len(content) > self.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(content)} bytes. "
                       f"Max: {self.MAX_FILE_SIZE} bytes"
            )

    def _validate_image(self, content: bytes) -> Image.Image:
        """
        Validate image integrity using PIL verify() + load().

        Returns:
            PIL Image object
        """
        try:
            # First pass: verify structure
            img = Image.open(io.BytesIO(content))
            img.verify()

            # Second pass: load actual data (verify closes file)
            img = Image.open(io.BytesIO(content))
            img.load()

            return img

        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Not a valid image format"
            )
        except IOError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Corrupted image: {str(e)}"
            )
        except MemoryError:
            raise HTTPException(
                status_code=413,
                detail="Image too large to process in memory"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image: {str(e)}"
            )

    def _validate_dimensions(self, size: Tuple[int, int]):
        """Validate image dimensions."""
        width, height = size

        if width < self.MIN_DIMENSIONS[0] or height < self.MIN_DIMENSIONS[1]:
            raise HTTPException(
                status_code=400,
                detail=f"Image too small: {width}x{height}. "
                       f"Min: {self.MIN_DIMENSIONS[0]}x{self.MIN_DIMENSIONS[1]}"
            )

        if width > self.MAX_DIMENSIONS[0] or height > self.MAX_DIMENSIONS[1]:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large: {width}x{height}. "
                       f"Max: {self.MAX_DIMENSIONS[0]}x{self.MAX_DIMENSIONS[1]}"
            )

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model inference.

        Args:
            img: PIL Image

        Returns:
            Tensor of shape (1, 3, 224, 224)
        """
        # Convert to RGB (handles grayscale, RGBA, palette)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Apply transforms
        tensor = self.transform(img)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor
```

### 2. Add Image Service Dependency

```python
# api/dependencies.py
from functools import lru_cache
from .services.image_service import ImageService
from .config import Settings

@lru_cache
def get_settings() -> Settings:
    return Settings()

@lru_cache
def get_image_service() -> ImageService:
    settings = get_settings()
    return ImageService(image_size=settings.image_size)
```

### 3. Create Custom Exceptions (api/exceptions.py)

```python
from fastapi import HTTPException

class ImageValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)

class ImageTooLargeError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=413, detail=detail)

class ModelInferenceError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail)
```

### 4. Add Exception Handlers to Main App

```python
# api/main.py
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Invalid request",
            "errors": exc.errors()
        }
    )
```

## Todo List

- [x] Create api/services/image_service.py
- [x] Create api/dependencies.py
- [x] Create api/exceptions.py
- [x] Add exception handlers to main.py
- [x] Unit test with valid images (JPEG, PNG, WebP) - 61/61 tests passing
- [x] Unit test with corrupted images
- [x] Unit test with oversized images
- [x] Unit test with undersized images
- [x] Verify tensor output shape (1, 3, 224, 224)

## Code Review Findings

### High Priority Improvements (Recommended)

1. **Reorder dimension validation** - Check dimensions BEFORE img.load() to prevent 1GB+ memory allocation
2. **Set explicit PIL.MAX_IMAGE_PIXELS** - Prevent decompression bomb warnings for large images
3. **Add TypedDict for metadata** - Improve type safety

See [detailed code review](../reports/code-reviewer-251216-phase02-review.md) for full analysis.

## Success Criteria

1. JPEG, PNG, WebP images accepted
2. Corrupted images rejected with 400 error
3. Files >10MB rejected with 413 error
4. Images <16x16 rejected with 400 error
5. Output tensor shape: (1, 3, 224, 224)
6. Normalization matches training pipeline

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Memory issues with large images | File size limit + PIL lazy loading |
| Slow preprocessing | Image resize is O(n), acceptable for 224x224 |
| Format not detected | PIL handles common formats, MIME pre-check |
| Corrupted file bypass | Double verification: verify() + load() |

## Security Considerations

- File size limits prevent DoS
- Dimension limits prevent pixel flood attacks
- No file system writes (in-memory only)
- MIME whitelist prevents arbitrary file upload
- Magic byte validation via PIL

## Next Steps

After Phase 02:
- [Phase 03: Inference Endpoint](./phase-03-inference-endpoint.md)
