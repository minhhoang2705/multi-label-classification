"""
Test fixtures for API tests.
"""

import pytest
from pathlib import Path
from PIL import Image
import io
import numpy as np
import torch
import sys

# Adjust sys path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient
from api.main import app
from api.services.image_service import ImageService
from api.config import Settings


# Test image paths
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def client():
    """Create test client with lifespan context."""
    # TestClient with lifespan support requires context manager
    with TestClient(app) as client:
        yield client


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def image_service():
    """Create image service."""
    return ImageService(image_size=224)


@pytest.fixture
def valid_jpeg_bytes():
    """Create valid JPEG image bytes."""
    img = Image.new("RGB", (256, 256), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def valid_png_bytes():
    """Create valid PNG image bytes."""
    img = Image.new("RGB", (256, 256), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def corrupted_image_bytes():
    """Create corrupted image bytes."""
    return b"not an image content"


@pytest.fixture
def tiny_image_bytes():
    """Create 1x1 image (too small)."""
    img = Image.new("RGB", (1, 1), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def grayscale_image_bytes():
    """Create grayscale image."""
    img = Image.new("L", (256, 256), color=128)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def rgba_image_bytes():
    """Create RGBA image with transparency."""
    img = Image.new("RGBA", (256, 256), color=(255, 0, 0, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def mock_probabilities():
    """Create mock probability array."""
    probs = np.zeros((1, 67))
    probs[0, 0] = 0.8   # Abyssinian
    probs[0, 1] = 0.1   # American Bobtail
    probs[0, 2] = 0.05  # American Curl
    probs[0, 3] = 0.03  # American Shorthair
    probs[0, 4] = 0.02  # American Wirehair
    return probs


@pytest.fixture
def mock_class_names():
    """Create mock class names list."""
    return [
        "Abyssinian", "American Bobtail", "American Curl", "American Shorthair",
        "American Wirehair", "Applehead Siamese", "Balinese", "Bengal"
    ] + [f"Breed_{i}" for i in range(8, 67)]  # Fill remaining
