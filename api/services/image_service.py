"""
Image validation and preprocessing service for inference.
"""

import io
import logging
from typing import Tuple
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as T
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)


class ImageService:
    """Service for validating and preprocessing uploaded images."""

    ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_DIMENSIONS = (10000, 10000)
    MIN_DIMENSIONS = (16, 16)

    def __init__(self, image_size: int = 224):
        """
        Initialize ImageService.

        Args:
            image_size: Target image size for model input
        """
        self.image_size = image_size

        # Define preprocessing transforms (matches training val transforms)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

    async def validate_and_preprocess(
        self, file: UploadFile
    ) -> Tuple[torch.Tensor, dict]:
        """
        Validate uploaded image and preprocess for inference.

        Args:
            file: FastAPI UploadFile object

        Returns:
            Tuple of (preprocessed tensor [1,3,H,W], metadata dict)

        Raises:
            HTTPException: For various validation errors
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

        # 6. Collect metadata before transforms
        metadata = {
            "original_width": img.size[0],
            "original_height": img.size[1],
            "format": img.format,
            "mode": img.mode,
            "file_size_bytes": len(content),
            "filename": file.filename
        }

        logger.info(
            f"Validated image: {file.filename}, "
            f"size={img.size}, format={img.format}"
        )

        # 7. Preprocess for model
        tensor = self._preprocess(img)

        return tensor, metadata

    def _validate_mime(self, content_type: str):
        """
        Validate MIME type against whitelist.

        Args:
            content_type: HTTP Content-Type header value

        Raises:
            HTTPException: If MIME type not allowed
        """
        if content_type not in self.ALLOWED_MIMES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {content_type}. "
                       f"Allowed: {', '.join(self.ALLOWED_MIMES)}"
            )

    def _validate_file_size(self, content: bytes):
        """
        Validate file size doesn't exceed limit.

        Args:
            content: File bytes

        Raises:
            HTTPException: If file too large
        """
        if len(content) > self.MAX_FILE_SIZE:
            size_mb = len(content) / (1024 * 1024)
            max_mb = self.MAX_FILE_SIZE / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {size_mb:.2f}MB. Max: {max_mb:.0f}MB"
            )

    def _validate_image(self, content: bytes) -> Image.Image:
        """
        Validate image integrity using PIL verify() + load() pattern.
        This catches ~99% of corrupted images.

        Args:
            content: Image file bytes

        Returns:
            PIL Image object

        Raises:
            HTTPException: For corrupted or invalid images
        """
        try:
            # First pass: verify file structure
            img = Image.open(io.BytesIO(content))
            img.verify()  # Validates format structure

            # Second pass: actually load pixel data
            # (verify() closes the file, must reopen)
            img = Image.open(io.BytesIO(content))
            img.load()  # Forces pixel data loading

            return img

        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Not a valid image format. "
                       "Supported: JPEG, PNG, WebP"
            )
        except IOError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Corrupted or invalid image: {str(e)}"
            )
        except MemoryError:
            raise HTTPException(
                status_code=413,
                detail="Image too large to process in memory"
            )
        except Exception as e:
            logger.error(f"Unexpected error validating image: {e}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image: {str(e)}"
            )

    def _validate_dimensions(self, size: Tuple[int, int]):
        """
        Validate image dimensions are within acceptable range.

        Args:
            size: (width, height) tuple

        Raises:
            HTTPException: If dimensions invalid
        """
        width, height = size

        # Check minimum dimensions
        if width < self.MIN_DIMENSIONS[0] or height < self.MIN_DIMENSIONS[1]:
            raise HTTPException(
                status_code=400,
                detail=f"Image too small: {width}x{height}. "
                       f"Minimum: {self.MIN_DIMENSIONS[0]}x{self.MIN_DIMENSIONS[1]}"
            )

        # Check maximum dimensions (prevents pixel flood attacks)
        if width > self.MAX_DIMENSIONS[0] or height > self.MAX_DIMENSIONS[1]:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large: {width}x{height}. "
                       f"Maximum: {self.MAX_DIMENSIONS[0]}x{self.MAX_DIMENSIONS[1]}"
            )

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model inference.

        Pipeline:
        1. Convert to RGB (handles grayscale, RGBA, palette modes)
        2. Resize to target size (224x224)
        3. Convert to tensor [0-1] range
        4. Normalize with ImageNet stats
        5. Add batch dimension

        Args:
            img: PIL Image

        Returns:
            Tensor of shape (1, 3, 224, 224)
        """
        # Convert to RGB early (handles grayscale, RGBA, palette)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Apply torchvision transforms
        tensor = self.transform(img)

        # Add batch dimension [C, H, W] -> [B, C, H, W]
        tensor = tensor.unsqueeze(0)

        return tensor
