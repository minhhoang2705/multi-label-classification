"""
Test suite for Predict Endpoint
"""

import pytest
from io import BytesIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPredictEndpoint:
    """Test prediction endpoint."""

    def test_predict_valid_image(self, client, valid_jpeg_bytes):
        """Test prediction with valid image."""
        files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict", files=files)

        assert response.status_code == 200
        data = response.json()

        assert "predicted_class" in data
        assert "confidence" in data
        assert "top_5_predictions" in data
        assert "inference_time_ms" in data
        assert "image_metadata" in data

        assert len(data["top_5_predictions"]) == 5
        assert data["confidence"] >= 0.0
        assert data["confidence"] <= 1.0

    def test_predict_invalid_mime(self, client):
        """Test rejection of invalid MIME type."""
        files = {"file": ("file.txt", BytesIO(b"not an image"), "text/plain")}
        response = client.post("/api/v1/predict", files=files)

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_predict_corrupted_image(self, client, corrupted_image_bytes):
        """Test rejection of corrupted image."""
        files = {"file": ("bad.jpg", BytesIO(corrupted_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict", files=files)

        assert response.status_code == 400

    def test_predict_tiny_image(self, client, tiny_image_bytes):
        """Test rejection of undersized image."""
        files = {"file": ("tiny.png", BytesIO(tiny_image_bytes), "image/png")}
        response = client.post("/api/v1/predict", files=files)

        assert response.status_code == 400
        assert "too small" in response.json()["detail"]

    def test_predict_png_image(self, client, valid_png_bytes):
        """Test prediction with PNG image."""
        files = {"file": ("cat.png", BytesIO(valid_png_bytes), "image/png")}
        response = client.post("/api/v1/predict", files=files)

        assert response.status_code == 200

    def test_predict_response_schema(self, client, valid_jpeg_bytes):
        """Test response matches expected schema."""
        files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict", files=files)

        data = response.json()

        # Check top_5_predictions structure
        for pred in data["top_5_predictions"]:
            assert "rank" in pred
            assert "class_name" in pred
            assert "class_id" in pred
            assert "confidence" in pred
            assert 1 <= pred["rank"] <= 5
            assert 0 <= pred["class_id"] <= 66

        # Check image_metadata
        meta = data["image_metadata"]
        assert "original_width" in meta
        assert "original_height" in meta
        assert "file_size_bytes" in meta

    def test_predict_top_prediction_matches(self, client, valid_jpeg_bytes):
        """Test top prediction matches first in top_5."""
        files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict", files=files)

        data = response.json()

        # Top prediction should match first in top_5_predictions
        assert data["predicted_class"] == data["top_5_predictions"][0]["class_name"]
        assert data["confidence"] == data["top_5_predictions"][0]["confidence"]

    def test_predict_inference_time_positive(self, client, valid_jpeg_bytes):
        """Test inference time is positive."""
        files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict", files=files)

        data = response.json()
        assert data["inference_time_ms"] > 0

    def test_predict_grayscale_image(self, client, grayscale_image_bytes):
        """Test prediction with grayscale image."""
        files = {"file": ("gray.png", BytesIO(grayscale_image_bytes), "image/png")}
        response = client.post("/api/v1/predict", files=files)

        # Grayscale images should be converted to RGB and processed
        assert response.status_code == 200

    def test_predict_rgba_image(self, client, rgba_image_bytes):
        """Test prediction with RGBA image."""
        files = {"file": ("rgba.png", BytesIO(rgba_image_bytes), "image/png")}
        response = client.post("/api/v1/predict", files=files)

        # RGBA images should be converted to RGB and processed
        assert response.status_code == 200
