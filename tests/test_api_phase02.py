"""
Phase 02: Image Validation & Preprocessing - Comprehensive Test Suite

Tests:
1. ImageService MIME type validation
2. File size validation (max 10MB)
3. Image dimension validation (min 16x16, max 10000x10000)
4. Image corruption detection
5. RGB conversion (grayscale, RGBA, palette modes)
6. Preprocessing output shape (1, 3, 224, 224)
7. ImageNet normalization verification
8. Integration tests with various image formats (JPEG, PNG, WebP)
"""

import io
import asyncio
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import UploadFile
import torch
import numpy as np

# Adjust sys path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.services.image_service import ImageService


class TestImageServiceMimeValidation:
    """Test MIME type validation."""

    @pytest.fixture
    def image_service(self):
        """Create ImageService instance."""
        return ImageService(image_size=224)

    def test_valid_mime_jpeg(self, image_service):
        """Test JPEG MIME type is accepted."""
        # Should not raise
        image_service._validate_mime("image/jpeg")

    def test_valid_mime_png(self, image_service):
        """Test PNG MIME type is accepted."""
        # Should not raise
        image_service._validate_mime("image/png")

    def test_valid_mime_webp(self, image_service):
        """Test WebP MIME type is accepted."""
        # Should not raise
        image_service._validate_mime("image/webp")

    def test_invalid_mime_gif(self, image_service):
        """Test GIF MIME type is rejected."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_mime("image/gif")

        assert exc_info.value.status_code == 400
        assert "Invalid file type" in exc_info.value.detail

    def test_invalid_mime_tiff(self, image_service):
        """Test TIFF MIME type is rejected."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_mime("image/tiff")

        assert exc_info.value.status_code == 400

    def test_invalid_mime_text(self, image_service):
        """Test text MIME type is rejected."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_mime("text/plain")

        assert exc_info.value.status_code == 400

    def test_invalid_mime_empty(self, image_service):
        """Test empty MIME type is rejected."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_mime("")

        assert exc_info.value.status_code == 400


class TestImageServiceFileSizeValidation:
    """Test file size validation."""

    @pytest.fixture
    def image_service(self):
        """Create ImageService instance."""
        return ImageService(image_size=224)

    def test_valid_small_file(self, image_service):
        """Test small file size is accepted."""
        small_content = b"x" * 1024 * 100  # 100KB
        # Should not raise
        image_service._validate_file_size(small_content)

    def test_valid_medium_file(self, image_service):
        """Test medium file size is accepted."""
        medium_content = b"x" * 1024 * 1024 * 5  # 5MB
        # Should not raise
        image_service._validate_file_size(medium_content)

    def test_valid_max_file(self, image_service):
        """Test max allowed file size is accepted."""
        max_content = b"x" * (10 * 1024 * 1024)  # Exactly 10MB
        # Should not raise
        image_service._validate_file_size(max_content)

    def test_invalid_oversized_file(self, image_service):
        """Test oversized file is rejected with 413 status."""
        from fastapi import HTTPException

        oversized_content = b"x" * (10 * 1024 * 1024 + 1)  # 10MB + 1 byte

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_file_size(oversized_content)

        assert exc_info.value.status_code == 413
        assert "too large" in exc_info.value.detail.lower()

    def test_invalid_very_large_file(self, image_service):
        """Test very large file is rejected."""
        from fastapi import HTTPException

        large_content = b"x" * (100 * 1024 * 1024)  # 100MB

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_file_size(large_content)

        assert exc_info.value.status_code == 413


class TestImageServiceDimensionValidation:
    """Test image dimension validation."""

    @pytest.fixture
    def image_service(self):
        """Create ImageService instance."""
        return ImageService(image_size=224)

    def test_valid_small_dimensions(self, image_service):
        """Test small but valid dimensions."""
        # Should not raise for 16x16 (minimum)
        image_service._validate_dimensions((16, 16))

    def test_valid_medium_dimensions(self, image_service):
        """Test medium dimensions."""
        # Should not raise
        image_service._validate_dimensions((224, 224))

    def test_valid_large_dimensions(self, image_service):
        """Test large dimensions."""
        # Should not raise
        image_service._validate_dimensions((5000, 5000))

    def test_valid_max_dimensions(self, image_service):
        """Test maximum allowed dimensions."""
        # Should not raise for 10000x10000 (maximum)
        image_service._validate_dimensions((10000, 10000))

    def test_invalid_too_small_width(self, image_service):
        """Test image with width below minimum."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_dimensions((15, 100))

        assert exc_info.value.status_code == 400
        assert "too small" in exc_info.value.detail.lower()

    def test_invalid_too_small_height(self, image_service):
        """Test image with height below minimum."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_dimensions((100, 15))

        assert exc_info.value.status_code == 400
        assert "too small" in exc_info.value.detail.lower()

    def test_invalid_too_wide(self, image_service):
        """Test image width exceeds maximum."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_dimensions((10001, 1000))

        assert exc_info.value.status_code == 400
        assert "too large" in exc_info.value.detail.lower()

    def test_invalid_too_tall(self, image_service):
        """Test image height exceeds maximum."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_dimensions((1000, 10001))

        assert exc_info.value.status_code == 400
        assert "too large" in exc_info.value.detail.lower()

    def test_invalid_both_dimensions_too_small(self, image_service):
        """Test both dimensions below minimum."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_dimensions((1, 1))

        assert exc_info.value.status_code == 400


class TestImageServiceImageValidation:
    """Test image integrity validation."""

    @pytest.fixture
    def image_service(self):
        """Create ImageService instance."""
        return ImageService(image_size=224)

    def test_valid_jpeg_image(self, image_service):
        """Test valid JPEG image is accepted."""
        # Create a valid JPEG in memory
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        # Should not raise
        result = image_service._validate_image(content)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_valid_png_image(self, image_service):
        """Test valid PNG image is accepted."""
        img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        result = image_service._validate_image(content)
        assert isinstance(result, Image.Image)

    def test_valid_webp_image(self, image_service):
        """Test valid WebP image is accepted."""
        img = Image.new("RGB", (100, 100), color="green")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="WebP")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        result = image_service._validate_image(content)
        assert isinstance(result, Image.Image)

    def test_corrupted_image_truncated(self, image_service):
        """Test truncated/corrupted image is rejected."""
        from fastapi import HTTPException

        # Create a valid JPEG but truncate it
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        # Truncate the image data
        corrupted_content = content[:len(content) // 2]

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_image(corrupted_content)

        assert exc_info.value.status_code == 400
        assert "corrupted" in exc_info.value.detail.lower() or "invalid" in exc_info.value.detail.lower()

    def test_invalid_format_random_bytes(self, image_service):
        """Test random bytes are rejected."""
        from fastapi import HTTPException

        random_bytes = b"This is not an image file at all"

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_image(random_bytes)

        assert exc_info.value.status_code == 400

    def test_invalid_format_text_file(self, image_service):
        """Test text file is rejected."""
        from fastapi import HTTPException

        text_content = b"This is a text file pretending to be an image"

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_image(text_content)

        assert exc_info.value.status_code == 400

    def test_invalid_format_empty_file(self, image_service):
        """Test empty file is rejected."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            image_service._validate_image(b"")

        assert exc_info.value.status_code == 400


class TestImageServicePreprocessing:
    """Test image preprocessing pipeline."""

    @pytest.fixture
    def image_service(self):
        """Create ImageService instance."""
        return ImageService(image_size=224)

    def test_preprocess_output_shape_rgb(self, image_service):
        """Test preprocessed output has correct shape for RGB."""
        img = Image.new("RGB", (300, 300), color="red")

        tensor = image_service._preprocess(img)

        # Expected shape: (batch=1, channels=3, height=224, width=224)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_output_shape_grayscale(self, image_service):
        """Test preprocessed output has correct shape for grayscale converted to RGB."""
        img = Image.new("L", (300, 300), color=128)  # Grayscale

        tensor = image_service._preprocess(img)

        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_output_shape_rgba(self, image_service):
        """Test preprocessed output has correct shape for RGBA converted to RGB."""
        img = Image.new("RGBA", (300, 300), color=(255, 0, 0, 255))

        tensor = image_service._preprocess(img)

        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_converts_grayscale_to_rgb(self, image_service):
        """Test that grayscale image is converted to RGB mode."""
        img = Image.new("L", (300, 300), color=128)
        assert img.mode == "L"

        tensor = image_service._preprocess(img)

        # After preprocessing, we can't check the PIL image mode directly,
        # but we can verify the tensor has 3 channels
        assert tensor.shape[1] == 3  # 3 channels

    def test_preprocess_converts_rgba_to_rgb(self, image_service):
        """Test that RGBA image is converted to RGB mode."""
        img = Image.new("RGBA", (300, 300), color=(255, 0, 0, 255))
        assert img.mode == "RGBA"

        tensor = image_service._preprocess(img)

        assert tensor.shape[1] == 3  # 3 channels

    def test_preprocess_returns_torch_tensor(self, image_service):
        """Test preprocessing returns a torch tensor."""
        img = Image.new("RGB", (300, 300), color="blue")

        tensor = image_service._preprocess(img)

        assert isinstance(tensor, torch.Tensor)

    def test_preprocess_tensor_dtype(self, image_service):
        """Test tensor has float32 dtype."""
        img = Image.new("RGB", (300, 300), color="green")

        tensor = image_service._preprocess(img)

        assert tensor.dtype == torch.float32

    def test_preprocess_tensor_range(self, image_service):
        """Test tensor values are normalized (not in 0-255 range)."""
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))

        tensor = image_service._preprocess(img)

        # After normalization with ImageNet stats, values should be roughly in range [-2, 2]
        # and definitely not in [0, 255]
        assert tensor.min() < 2
        assert tensor.max() < 4

    def test_preprocess_different_input_sizes(self, image_service):
        """Test preprocessing works with different input image sizes."""
        sizes = [(50, 50), (100, 100), (500, 500), (1000, 1000)]

        for size in sizes:
            img = Image.new("RGB", size, color="red")
            tensor = image_service._preprocess(img)

            # Output should always be (1, 3, 224, 224) regardless of input
            assert tensor.shape == (1, 3, 224, 224)


class TestImageServiceNormalization:
    """Test ImageNet normalization is applied correctly."""

    @pytest.fixture
    def image_service(self):
        """Create ImageService instance."""
        return ImageService(image_size=224)

    def test_normalization_applied_to_tensor(self, image_service):
        """Test that normalization is applied (values are normalized, not raw)."""
        # Create a uniform color image
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))

        tensor = image_service._preprocess(img)

        # After converting 128/255 and normalizing, the value should be significantly different
        # from the raw 128/255 = 0.5 value
        normalized_value = tensor[0, 0, 0, 0].item()  # Sample a pixel

        # With ImageNet normalization:
        # raw_value = 128/255 = 0.5
        # (0.5 - mean) / std for each channel varies, but should be in range [-2, 2] roughly
        assert -3 < normalized_value < 3

    def test_imagenet_mean_std_applied(self, image_service):
        """Test ImageNet mean and std are used in normalization."""
        # ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

        # Create an image with known pixel values
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))

        tensor = image_service._preprocess(img)

        # For white pixels (normalized to 1.0), after subtracting mean and dividing by std:
        # (1.0 - 0.485) / 0.229 ≈ 2.25 for red channel
        # (1.0 - 0.456) / 0.224 ≈ 2.43 for green channel
        # (1.0 - 0.406) / 0.225 ≈ 2.64 for blue channel

        sample_pixel = tensor[0, :, 0, 0]  # Get first pixel all channels

        # All three channels should be in range [2, 3] for white pixels
        assert all(1.5 < val < 3.0 for val in sample_pixel.tolist())

    def test_black_pixel_normalization(self, image_service):
        """Test normalization of black pixels."""
        # Create a black image
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        tensor = image_service._preprocess(img)

        # For black pixels (normalized to 0.0), after subtracting mean and dividing by std:
        # (0.0 - 0.485) / 0.229 ≈ -2.11 for red channel
        # (0.0 - 0.456) / 0.224 ≈ -2.04 for green channel
        # (0.0 - 0.406) / 0.225 ≈ -1.80 for blue channel

        sample_pixel = tensor[0, :, 0, 0]

        # All three channels should be in range [-2.5, -1.5] for black pixels
        assert all(-2.5 < val < -1.5 for val in sample_pixel.tolist())


class TestImageServiceIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def image_service(self):
        """Create ImageService instance."""
        return ImageService(image_size=224)

    @pytest.mark.asyncio
    async def test_full_pipeline_valid_jpeg(self, image_service):
        """Test complete pipeline with valid JPEG."""
        # Create a valid JPEG
        img = Image.new("RGB", (300, 300), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        # Mock UploadFile
        mock_file = AsyncMock(spec=UploadFile)
        mock_file.content_type = "image/jpeg"
        mock_file.filename = "test.jpg"
        mock_file.read = AsyncMock(return_value=content)

        tensor, metadata = await image_service.validate_and_preprocess(mock_file)

        assert tensor.shape == (1, 3, 224, 224)
        assert metadata["filename"] == "test.jpg"
        assert metadata["format"] == "JPEG"
        assert metadata["original_width"] == 300
        assert metadata["original_height"] == 300

    @pytest.mark.asyncio
    async def test_full_pipeline_valid_png(self, image_service):
        """Test complete pipeline with valid PNG."""
        img = Image.new("RGB", (400, 400), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        mock_file = AsyncMock(spec=UploadFile)
        mock_file.content_type = "image/png"
        mock_file.filename = "test.png"
        mock_file.read = AsyncMock(return_value=content)

        tensor, metadata = await image_service.validate_and_preprocess(mock_file)

        assert tensor.shape == (1, 3, 224, 224)
        assert metadata["format"] == "PNG"

    @pytest.mark.asyncio
    async def test_full_pipeline_invalid_mime(self, image_service):
        """Test pipeline rejects invalid MIME type."""
        from fastapi import HTTPException

        # Create any file content
        content = b"fake image data"

        mock_file = AsyncMock(spec=UploadFile)
        mock_file.content_type = "image/gif"  # Not allowed
        mock_file.filename = "test.gif"
        mock_file.read = AsyncMock(return_value=content)

        with pytest.raises(HTTPException) as exc_info:
            await image_service.validate_and_preprocess(mock_file)

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_full_pipeline_oversized_file(self, image_service):
        """Test pipeline rejects oversized files."""
        from fastapi import HTTPException

        # Create oversized content
        oversized_content = b"x" * (10 * 1024 * 1024 + 1)

        mock_file = AsyncMock(spec=UploadFile)
        mock_file.content_type = "image/jpeg"
        mock_file.filename = "huge.jpg"
        mock_file.read = AsyncMock(return_value=oversized_content)

        with pytest.raises(HTTPException) as exc_info:
            await image_service.validate_and_preprocess(mock_file)

        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    async def test_full_pipeline_corrupted_image(self, image_service):
        """Test pipeline rejects corrupted images."""
        from fastapi import HTTPException

        # Create a valid JPEG then truncate it
        img = Image.new("RGB", (300, 300), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()
        corrupted_content = content[:len(content) // 2]

        mock_file = AsyncMock(spec=UploadFile)
        mock_file.content_type = "image/jpeg"
        mock_file.filename = "corrupted.jpg"
        mock_file.read = AsyncMock(return_value=corrupted_content)

        with pytest.raises(HTTPException) as exc_info:
            await image_service.validate_and_preprocess(mock_file)

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_full_pipeline_undersized_image(self, image_service):
        """Test pipeline rejects undersized images."""
        from fastapi import HTTPException

        # Create a very small image (smaller than minimum)
        img = Image.new("RGB", (10, 10), color="red")  # 10x10 < 16x16 minimum
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        mock_file = AsyncMock(spec=UploadFile)
        mock_file.content_type = "image/jpeg"
        mock_file.filename = "tiny.jpg"
        mock_file.read = AsyncMock(return_value=content)

        with pytest.raises(HTTPException) as exc_info:
            await image_service.validate_and_preprocess(mock_file)

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_full_pipeline_oversized_dimensions(self, image_service):
        """Test pipeline rejects oversized dimensions."""
        from fastapi import HTTPException

        # Mock an image that would exceed max dimensions
        with patch.object(
            image_service, "_validate_image"
        ) as mock_validate:
            # Create a mock PIL Image with large dimensions
            mock_img = MagicMock()
            mock_img.mode = "RGB"
            mock_img.size = (10001, 1000)  # Width exceeds max
            mock_img.format = "JPEG"
            mock_validate.return_value = mock_img

            content = b"dummy"

            mock_file = AsyncMock(spec=UploadFile)
            mock_file.content_type = "image/jpeg"
            mock_file.filename = "huge_dims.jpg"
            mock_file.read = AsyncMock(return_value=content)

            with pytest.raises(HTTPException) as exc_info:
                await image_service.validate_and_preprocess(mock_file)

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_full_pipeline_grayscale_conversion(self, image_service):
        """Test pipeline converts grayscale to RGB."""
        # Create a grayscale image
        img = Image.new("L", (300, 300), color=128)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        mock_file = AsyncMock(spec=UploadFile)
        mock_file.content_type = "image/png"
        mock_file.filename = "grayscale.png"
        mock_file.read = AsyncMock(return_value=content)

        tensor, metadata = await image_service.validate_and_preprocess(mock_file)

        # Verify output is RGB (3 channels)
        assert tensor.shape == (1, 3, 224, 224)
        assert metadata["mode"] == "L"  # Original was grayscale

    @pytest.mark.asyncio
    async def test_full_pipeline_rgba_conversion(self, image_service):
        """Test pipeline converts RGBA to RGB."""
        # Create an RGBA image
        img = Image.new("RGBA", (300, 300), color=(255, 0, 0, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        mock_file = AsyncMock(spec=UploadFile)
        mock_file.content_type = "image/png"
        mock_file.filename = "rgba.png"
        mock_file.read = AsyncMock(return_value=content)

        tensor, metadata = await image_service.validate_and_preprocess(mock_file)

        # Verify output is RGB (3 channels)
        assert tensor.shape == (1, 3, 224, 224)
        assert metadata["mode"] == "RGBA"  # Original was RGBA


class TestImageServiceEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def image_service(self):
        """Create ImageService instance."""
        return ImageService(image_size=224)

    def test_minimum_valid_dimensions_16x16(self, image_service):
        """Test minimum valid dimensions (16x16)."""
        img = Image.new("RGB", (16, 16), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        # Should not raise
        result = image_service._validate_image(content)
        assert result.size == (16, 16)

    def test_maximum_valid_dimensions_10000x10000(self, image_service):
        """Test maximum valid dimensions (10000x10000)."""
        # Don't actually create the image (too large), just test the validation
        image_service._validate_dimensions((10000, 10000))
        # Should not raise

    def test_non_square_image_landscape(self, image_service):
        """Test non-square landscape image."""
        img = Image.new("RGB", (800, 400), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        tensor = Image.open(io.BytesIO(content))
        tensor = image_service._preprocess(tensor)

        assert tensor.shape == (1, 3, 224, 224)

    def test_non_square_image_portrait(self, image_service):
        """Test non-square portrait image."""
        img = Image.new("RGB", (400, 800), color="green")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        tensor = Image.open(io.BytesIO(content))
        tensor = image_service._preprocess(tensor)

        assert tensor.shape == (1, 3, 224, 224)

    def test_palette_mode_image_conversion(self, image_service):
        """Test palette mode (indexed color) image conversion."""
        # Create a palette mode image
        img = Image.new("P", (300, 300))
        # Add a simple palette (palette values must be in range 0-255)
        palette = [i % 256 for i in range(768)]  # 256 colors * 3 channels
        img.putpalette(palette)

        # Fill with color
        img.paste(0, (0, 0, 300, 300))

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        content = img_bytes.getvalue()

        loaded_img = image_service._validate_image(content)
        tensor = image_service._preprocess(loaded_img)

        assert tensor.shape == (1, 3, 224, 224)


class TestImageServiceConfiguration:
    """Test ImageService configuration."""

    def test_default_image_size(self):
        """Test default image size is 224."""
        service = ImageService()
        assert service.image_size == 224

    def test_custom_image_size(self):
        """Test custom image size configuration."""
        service = ImageService(image_size=256)
        assert service.image_size == 256

    def test_custom_image_size_preprocessing(self):
        """Test preprocessing respects custom image size."""
        service = ImageService(image_size=256)

        img = Image.new("RGB", (300, 300), color="red")
        tensor = service._preprocess(img)

        # With custom size, should be (1, 3, 256, 256)
        assert tensor.shape == (1, 3, 256, 256)

    def test_mime_types_allowed(self):
        """Test allowed MIME types are correct."""
        service = ImageService()

        assert "image/jpeg" in service.ALLOWED_MIMES
        assert "image/png" in service.ALLOWED_MIMES
        assert "image/webp" in service.ALLOWED_MIMES

    def test_max_file_size_is_10mb(self):
        """Test max file size is 10MB."""
        service = ImageService()
        assert service.MAX_FILE_SIZE == 10 * 1024 * 1024

    def test_min_dimensions_is_16x16(self):
        """Test minimum dimensions are 16x16."""
        service = ImageService()
        assert service.MIN_DIMENSIONS == (16, 16)

    def test_max_dimensions_is_10000x10000(self):
        """Test maximum dimensions are 10000x10000."""
        service = ImageService()
        assert service.MAX_DIMENSIONS == (10000, 10000)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
