"""
Test suite for /predict/verified endpoint.

These are integration tests that verify the hybrid CNN + VLM endpoint
works correctly end-to-end. We mock the VLM service to avoid real API calls.
"""

import pytest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, Mock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPredictVerifiedEndpoint:
    """Test /predict/verified endpoint with VLM verification."""

    def test_verified_endpoint_exists(self, client):
        """Endpoint should exist and accept POST requests."""
        # Test with OPTIONS to check endpoint exists
        response = client.options("/api/v1/predict/verified")
        # Should return 200 or 405 (method not allowed for OPTIONS)
        assert response.status_code in [200, 405]

    @patch('api.services.vlm_service.VLMService.get_instance')
    def test_verified_prediction_with_vlm_agreement(
        self,
        mock_get_vlm,
        client,
        valid_jpeg_bytes
    ):
        """
        Test prediction when CNN and VLM agree.
        Should return 'verified' status with high confidence.
        """
        # Mock VLM service to agree with CNN
        mock_vlm = Mock()
        mock_vlm.verify_prediction = Mock(
            return_value=("agree", "Abyssinian", "Ticked coat pattern")
        )
        mock_get_vlm.return_value = mock_vlm

        files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict/verified", files=files)

        assert response.status_code == 200
        data = response.json()

        # Check agreement fields
        assert data["verification_status"] == "verified"
        assert data["confidence_level"] == "high"
        assert data["vlm_prediction"] is not None
        assert data["vlm_reasoning"] is not None

    @patch('api.services.vlm_service.VLMService.get_instance')
    def test_verified_prediction_with_vlm_disagreement(
        self,
        mock_get_vlm,
        client,
        valid_jpeg_bytes
    ):
        """
        Test prediction when CNN and VLM disagree.
        Should return 'uncertain' status and use VLM prediction.
        """
        # Mock VLM service to disagree with CNN
        mock_vlm = Mock()
        mock_vlm.verify_prediction = Mock(
            return_value=("disagree", "Bengal", "Spotted pattern, not ticked")
        )
        mock_get_vlm.return_value = mock_vlm

        files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict/verified", files=files)

        assert response.status_code == 200
        data = response.json()

        # Check disagreement handling
        assert data["verification_status"] == "uncertain"
        assert data["confidence_level"] == "medium"
        # VLM prediction should be used as final
        assert data["predicted_class"] == data["vlm_prediction"]
        assert data["predicted_class"] != data["cnn_prediction"]

    @patch('api.services.vlm_service.VLMService.get_instance')
    def test_verified_prediction_vlm_disabled(
        self,
        mock_get_vlm,
        client,
        valid_jpeg_bytes
    ):
        """
        Test prediction when VLM is not available.
        Should return 'cnn_only' status and fallback to CNN.
        """
        # Mock VLM service not available
        mock_get_vlm.side_effect = ValueError("ZAI_API_KEY not set")

        files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict/verified", files=files)

        assert response.status_code == 200
        data = response.json()

        # Should fallback to CNN-only
        assert data["verification_status"] == "cnn_only"
        assert data["confidence_level"] == "low"
        assert data["vlm_prediction"] is None
        assert data["vlm_time_ms"] is None

    def test_verified_response_schema(self, client, valid_jpeg_bytes):
        """Test response matches HybridPredictionResponse schema."""
        with patch('api.services.vlm_service.VLMService.get_instance') as mock_get_vlm:
            mock_vlm = Mock()
            mock_vlm.verify_prediction = Mock(
                return_value=("agree", "Abyssinian", "Test reasoning")
            )
            mock_get_vlm.return_value = mock_vlm

            files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
            response = client.post("/api/v1/predict/verified", files=files)

            data = response.json()

            # Check all required fields are present
            assert "predicted_class" in data
            assert "confidence_level" in data
            assert "verification_status" in data
            assert "cnn_prediction" in data
            assert "cnn_confidence" in data
            assert "top_5_predictions" in data
            assert "vlm_prediction" in data
            assert "vlm_reasoning" in data
            assert "cnn_time_ms" in data
            assert "vlm_time_ms" in data
            assert "total_time_ms" in data
            assert "image_metadata" in data
            assert "model_info" in data

    def test_verified_invalid_image(self, client, corrupted_image_bytes):
        """Test rejection of corrupted image."""
        files = {"file": ("bad.jpg", BytesIO(corrupted_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict/verified", files=files)

        assert response.status_code == 400

    def test_verified_timing_metrics(self, client, valid_jpeg_bytes):
        """Test timing metrics are present and valid."""
        with patch('api.services.vlm_service.VLMService.get_instance') as mock_get_vlm:
            mock_vlm = Mock()
            mock_vlm.verify_prediction = Mock(
                return_value=("agree", "Abyssinian", "Test reasoning")
            )
            mock_get_vlm.return_value = mock_vlm

            files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
            response = client.post("/api/v1/predict/verified", files=files)

            data = response.json()

            # Check timing metrics are positive numbers
            assert data["cnn_time_ms"] > 0
            assert data["vlm_time_ms"] > 0
            assert data["total_time_ms"] > 0
            # Total should be greater than individual times
            assert data["total_time_ms"] >= data["cnn_time_ms"]

    def test_verified_cnn_results_always_present(self, client, valid_jpeg_bytes):
        """CNN results should always be present regardless of VLM status."""
        with patch('api.services.vlm_service.VLMService.get_instance') as mock_get_vlm:
            # VLM disabled
            mock_get_vlm.side_effect = ValueError("Not available")

            files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
            response = client.post("/api/v1/predict/verified", files=files)

            data = response.json()

            # CNN results should always be present
            assert data["cnn_prediction"] is not None
            assert data["cnn_confidence"] > 0
            assert len(data["top_5_predictions"]) == 5

    def test_verified_model_info(self, client, valid_jpeg_bytes):
        """Test model_info includes both CNN and VLM information."""
        with patch('api.services.vlm_service.VLMService.get_instance') as mock_get_vlm:
            mock_vlm = Mock()
            mock_vlm.verify_prediction = Mock(
                return_value=("agree", "Abyssinian", "Test")
            )
            mock_get_vlm.return_value = mock_vlm

            files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
            response = client.post("/api/v1/predict/verified", files=files)

            data = response.json()

            # Check model info
            model_info = data["model_info"]
            assert "cnn_model" in model_info
            assert "vlm_model" in model_info
            assert "device" in model_info
            assert model_info["vlm_model"] == "glm-4.6v"

    @patch('api.services.vlm_service.VLMService.get_instance')
    def test_verified_top_5_structure(
        self,
        mock_get_vlm,
        client,
        valid_jpeg_bytes
    ):
        """Test top_5_predictions have correct structure."""
        mock_vlm = Mock()
        mock_vlm.verify_prediction = Mock(
            return_value=("agree", "Abyssinian", "Test")
        )
        mock_get_vlm.return_value = mock_vlm

        files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict/verified", files=files)

        data = response.json()

        # Check top_5_predictions structure
        for pred in data["top_5_predictions"]:
            assert "rank" in pred
            assert "class_name" in pred
            assert "class_id" in pred
            assert "confidence" in pred
            assert 1 <= pred["rank"] <= 5
            assert 0 <= pred["class_id"] <= 66

    @patch('api.services.vlm_service.VLMService.get_instance')
    def test_verified_confidence_levels(
        self,
        mock_get_vlm,
        client,
        valid_jpeg_bytes
    ):
        """Test confidence levels are correct for different statuses."""
        # Test 1: Agreement -> high confidence
        mock_vlm = Mock()
        mock_vlm.verify_prediction = Mock(
            return_value=("agree", "Abyssinian", "Test")
        )
        mock_get_vlm.return_value = mock_vlm

        files = {"file": ("cat.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict/verified", files=files)
        assert response.json()["confidence_level"] == "high"

        # Test 2: Disagreement -> medium confidence
        mock_vlm.verify_prediction = Mock(
            return_value=("disagree", "Bengal", "Test")
        )
        files = {"file": ("cat2.jpg", BytesIO(valid_jpeg_bytes), "image/jpeg")}
        response = client.post("/api/v1/predict/verified", files=files)
        assert response.json()["confidence_level"] == "medium"

    def test_verified_accepts_png(self, client, valid_png_bytes):
        """Test endpoint accepts PNG images."""
        with patch('api.services.vlm_service.VLMService.get_instance') as mock_get_vlm:
            mock_vlm = Mock()
            mock_vlm.verify_prediction = Mock(
                return_value=("agree", "Abyssinian", "Test")
            )
            mock_get_vlm.return_value = mock_vlm

            files = {"file": ("cat.png", BytesIO(valid_png_bytes), "image/png")}
            response = client.post("/api/v1/predict/verified", files=files)

            assert response.status_code == 200

    def test_verified_tiny_image_rejected(self, client, tiny_image_bytes):
        """Test rejection of undersized image."""
        files = {"file": ("tiny.png", BytesIO(tiny_image_bytes), "image/png")}
        response = client.post("/api/v1/predict/verified", files=files)

        assert response.status_code == 400
        assert "too small" in response.json()["detail"]
