"""
Test suite for ImageService - Validation and Preprocessing
"""

import pytest
from fastapi import HTTPException
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.services.image_service import ImageService


class TestImageServiceValidation:
    """Test image validation."""

    def test_validate_mime_jpeg(self, image_service):
        """Test JPEG MIME type accepted."""
        image_service._validate_mime("image/jpeg")  # Should not raise

    def test_validate_mime_png(self, image_service):
        """Test PNG MIME type accepted."""
        image_service._validate_mime("image/png")

    def test_validate_mime_webp(self, image_service):
        """Test WebP MIME type accepted."""
        image_service._validate_mime("image/webp")

    def test_validate_mime_invalid(self, image_service):
        """Test invalid MIME type rejected."""
        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_mime("application/pdf")
        assert exc_info.value.status_code == 400
        assert "Invalid file type" in exc_info.value.detail

    def test_validate_file_size_ok(self, image_service, valid_jpeg_bytes):
        """Test valid file size accepted."""
        image_service._validate_file_size(valid_jpeg_bytes)  # Should not raise

    def test_validate_file_size_too_large(self, image_service):
        """Test oversized file rejected."""
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_file_size(large_content)
        assert exc_info.value.status_code == 413

    def test_validate_image_structure_valid(self, image_service, valid_jpeg_bytes):
        """Test valid image passes structure verification."""
        img = image_service._validate_image_structure(valid_jpeg_bytes)
        assert img.size == (256, 256)

    def test_validate_image_structure_corrupted(self, image_service, corrupted_image_bytes):
        """Test corrupted image rejected."""
        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_image_structure(corrupted_image_bytes)
        assert exc_info.value.status_code == 400

    def test_validate_dimensions_ok(self, image_service):
        """Test valid dimensions accepted."""
        image_service._validate_dimensions((256, 256))

    def test_validate_dimensions_too_small(self, image_service):
        """Test undersized image rejected."""
        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_dimensions((10, 10))
        assert exc_info.value.status_code == 400
        assert "too small" in exc_info.value.detail

    def test_validate_dimensions_too_large(self, image_service):
        """Test oversized dimensions rejected."""
        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_dimensions((10001, 10001))
        assert exc_info.value.status_code == 400
        assert "too large" in exc_info.value.detail


class TestImageServicePreprocessing:
    """Test image preprocessing."""

    def test_preprocess_rgb(self, image_service, valid_jpeg_bytes):
        """Test RGB image preprocessing."""
        img = image_service._validate_image_structure(valid_jpeg_bytes)
        img = image_service._load_image_pixels(valid_jpeg_bytes)
        tensor = image_service._preprocess(img)

        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_preprocess_grayscale(self, image_service, grayscale_image_bytes):
        """Test grayscale image converted to RGB."""
        img = image_service._validate_image_structure(grayscale_image_bytes)
        img = image_service._load_image_pixels(grayscale_image_bytes)
        tensor = image_service._preprocess(img)

        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_rgba(self, image_service, rgba_image_bytes):
        """Test RGBA image converted to RGB."""
        img = image_service._validate_image_structure(rgba_image_bytes)
        img = image_service._load_image_pixels(rgba_image_bytes)
        tensor = image_service._preprocess(img)

        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_normalization(self, image_service, valid_jpeg_bytes):
        """Test ImageNet normalization applied."""
        img = image_service._validate_image_structure(valid_jpeg_bytes)
        img = image_service._load_image_pixels(valid_jpeg_bytes)
        tensor = image_service._preprocess(img)

        # After normalization, values should be roughly centered around 0
        # (depends on image content, but should not be in [0, 1] range)
        assert tensor.min() < 0 or tensor.max() > 1
