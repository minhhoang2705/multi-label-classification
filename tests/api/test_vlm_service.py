"""
Test suite for VLMService.

These tests verify the VLM service works correctly WITHOUT making real API calls.
We mock the Z.ai SDK to simulate different response scenarios.

Key test scenarios:
1. Service initialization (with/without API key)
2. Image encoding to base64
3. Response parsing (agree, disagree, unclear)
4. Error handling (network failures, parsing errors)
"""

import pytest
import base64
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestVLMServiceInitialization:
    """Test VLM service initialization."""

    def test_init_without_api_key_raises(self):
        """Service should fail if ZAI_API_KEY not set."""
        # Ensure no API key in environment
        with patch.dict("os.environ", {}, clear=True):
            # Need to reimport to get fresh state
            from api.services.vlm_service import VLMService
            VLMService.reset_instance()

            with pytest.raises(ValueError, match="ZAI_API_KEY"):
                VLMService()

    def test_init_with_api_key_succeeds(self):
        """Service should initialize when API key is set."""
        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key"}):
            # Mock the ZaiClient import
            with patch("api.services.vlm_service.VLMService.__init__") as mock_init:
                mock_init.return_value = None
                from api.services.vlm_service import VLMService
                VLMService.reset_instance()
                service = VLMService()
                # If we get here without exception, init succeeded

    def test_singleton_pattern(self):
        """get_instance should return same instance."""
        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key"}):
            with patch("zai.ZaiClient"):
                from api.services.vlm_service import VLMService
                VLMService.reset_instance()

                instance1 = VLMService.get_instance()
                instance2 = VLMService.get_instance()

                assert instance1 is instance2

    def test_reset_instance(self):
        """reset_instance should clear singleton."""
        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key"}):
            with patch("zai.ZaiClient"):
                from api.services.vlm_service import VLMService
                VLMService.reset_instance()

                instance1 = VLMService.get_instance()
                VLMService.reset_instance()
                instance2 = VLMService.get_instance()

                assert instance1 is not instance2


class TestImageEncoding:
    """Test image encoding functionality."""

    @pytest.fixture
    def vlm_service(self):
        """Create VLM service with mocked client."""
        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key"}):
            with patch("zai.ZaiClient"):
                from api.services.vlm_service import VLMService
                VLMService.reset_instance()
                return VLMService()

    def test_encode_jpeg_image(self, vlm_service, valid_jpeg_bytes):
        """Should encode JPEG to base64."""
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(valid_jpeg_bytes)
            temp_path = f.name

        try:
            encoded = vlm_service._encode_image_to_base64(temp_path)

            # Verify it's valid base64
            decoded = base64.b64decode(encoded)
            assert decoded == valid_jpeg_bytes
        finally:
            Path(temp_path).unlink()

    def test_encode_png_image(self, vlm_service, valid_png_bytes):
        """Should encode PNG to base64."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(valid_png_bytes)
            temp_path = f.name

        try:
            encoded = vlm_service._encode_image_to_base64(temp_path)
            decoded = base64.b64decode(encoded)
            assert decoded == valid_png_bytes
        finally:
            Path(temp_path).unlink()

    def test_encode_nonexistent_file_raises(self, vlm_service):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            vlm_service._encode_image_to_base64("/nonexistent/path.jpg")

    def test_get_mime_type_jpeg(self, vlm_service):
        """Should return correct MIME type for JPEG."""
        assert vlm_service._get_image_mime_type("image.jpg") == "image/jpeg"
        assert vlm_service._get_image_mime_type("image.jpeg") == "image/jpeg"

    def test_get_mime_type_png(self, vlm_service):
        """Should return correct MIME type for PNG."""
        assert vlm_service._get_image_mime_type("image.png") == "image/png"

    def test_get_mime_type_unknown_defaults_to_jpeg(self, vlm_service):
        """Should default to JPEG for unknown extensions."""
        assert vlm_service._get_image_mime_type("image.bmp") == "image/jpeg"


class TestPromptBuilding:
    """Test prompt generation."""

    @pytest.fixture
    def vlm_service(self):
        """Create VLM service with mocked client."""
        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key"}):
            with patch("zai.ZaiClient"):
                from api.services.vlm_service import VLMService
                VLMService.reset_instance()
                return VLMService()

    def test_build_prompt_with_top_3(self, vlm_service):
        """Prompt should include all 3 candidates."""
        cnn_top_3 = [
            ("Persian", 0.85),
            ("Himalayan", 0.10),
            ("Exotic Shorthair", 0.03)
        ]

        prompt = vlm_service._build_prompt(cnn_top_3)

        assert "Persian (85.0%)" in prompt
        assert "Himalayan (10.0%)" in prompt
        assert "Exotic Shorthair (3.0%)" in prompt
        assert "BREED:" in prompt
        assert "MATCHES_CNN:" in prompt
        assert "REASON:" in prompt

    def test_build_prompt_includes_instructions(self, vlm_service):
        """Prompt should include feature analysis instructions."""
        prompt = vlm_service._build_prompt([("Test", 0.5), ("Test2", 0.3), ("Test3", 0.2)])

        assert "coat pattern" in prompt.lower()
        assert "face shape" in prompt.lower()


class TestResponseParsing:
    """Test VLM response parsing."""

    @pytest.fixture
    def vlm_service(self):
        """Create VLM service with mocked client."""
        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key"}):
            with patch("zai.ZaiClient"):
                from api.services.vlm_service import VLMService
                VLMService.reset_instance()
                return VLMService()

    def test_parse_agree_response(self, vlm_service):
        """Should return 'agree' when VLM picks top-1 CNN prediction."""
        content = """BREED: Persian
MATCHES_CNN: YES
REASON: The flat face and long silky fur are characteristic of Persian cats."""

        cnn_top_3 = [("Persian", 0.85), ("Himalayan", 0.10), ("Exotic", 0.03)]

        status, prediction, reason = vlm_service._parse_response(content, cnn_top_3)

        assert status == "agree"
        assert prediction == "Persian"
        assert "flat face" in reason.lower()

    def test_parse_disagree_picks_top2(self, vlm_service):
        """Should return 'disagree' when VLM picks 2nd candidate."""
        content = """BREED: Himalayan
MATCHES_CNN: YES
REASON: The color points suggest Himalayan rather than pure Persian."""

        cnn_top_3 = [("Persian", 0.85), ("Himalayan", 0.10), ("Exotic", 0.03)]

        status, prediction, reason = vlm_service._parse_response(content, cnn_top_3)

        assert status == "disagree"
        assert prediction == "Himalayan"

    def test_parse_disagree_new_breed(self, vlm_service):
        """Should return 'disagree' when VLM suggests new breed."""
        content = """BREED: Ragdoll
MATCHES_CNN: NO
REASON: The blue eyes and color pattern indicate Ragdoll, not Persian."""

        cnn_top_3 = [("Persian", 0.85), ("Himalayan", 0.10), ("Exotic", 0.03)]

        status, prediction, reason = vlm_service._parse_response(content, cnn_top_3)

        assert status == "disagree"
        assert prediction == "Ragdoll"

    def test_parse_unclear_no_breed(self, vlm_service):
        """Should return 'unclear' when breed not parseable."""
        content = "I cannot determine the breed from this image."

        cnn_top_3 = [("Persian", 0.85), ("Himalayan", 0.10), ("Exotic", 0.03)]

        status, prediction, reason = vlm_service._parse_response(content, cnn_top_3)

        assert status == "unclear"
        # Falls back to CNN top-1
        assert prediction == "Persian"

    def test_parse_case_insensitive_matching(self, vlm_service):
        """Should match breeds case-insensitively."""
        content = """BREED: PERSIAN
MATCHES_CNN: YES
REASON: Classic Persian features."""

        cnn_top_3 = [("Persian", 0.85), ("Himalayan", 0.10), ("Exotic", 0.03)]

        status, prediction, reason = vlm_service._parse_response(content, cnn_top_3)

        assert status == "agree"
        # Should normalize to exact CNN name
        assert prediction == "Persian"


class TestVerifyPrediction:
    """Test end-to-end verification."""

    @pytest.fixture
    def vlm_service_with_mock_api(self):
        """Create VLM service with mocked API client."""
        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key"}):
            mock_client = MagicMock()
            with patch("zai.ZaiClient", return_value=mock_client):
                from api.services.vlm_service import VLMService
                VLMService.reset_instance()
                service = VLMService()
                return service, mock_client

    def test_verify_prediction_success(self, vlm_service_with_mock_api, valid_jpeg_bytes):
        """Should return parsed result on successful API call."""
        service, mock_client = vlm_service_with_mock_api

        # Mock API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """BREED: Persian
MATCHES_CNN: YES
REASON: Flat face and long fur typical of Persian."""

        mock_client.chat.completions.create.return_value = mock_response

        # Create temp image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(valid_jpeg_bytes)
            temp_path = f.name

        try:
            cnn_top_3 = [("Persian", 0.85), ("Himalayan", 0.10), ("Exotic", 0.03)]
            status, prediction, reason = service.verify_prediction(temp_path, cnn_top_3)

            assert status == "agree"
            assert prediction == "Persian"
            assert "flat face" in reason.lower()
        finally:
            Path(temp_path).unlink()

    def test_verify_prediction_file_not_found(self, vlm_service_with_mock_api):
        """Should handle missing file gracefully."""
        service, mock_client = vlm_service_with_mock_api

        cnn_top_3 = [("Persian", 0.85), ("Himalayan", 0.10), ("Exotic", 0.03)]
        status, prediction, reason = service.verify_prediction(
            "/nonexistent/image.jpg",
            cnn_top_3
        )

        assert status == "error"
        # Falls back to CNN top-1
        assert prediction == "Persian"
        assert "not found" in reason.lower() or "No such file" in reason

    def test_verify_prediction_api_error(self, vlm_service_with_mock_api, valid_jpeg_bytes):
        """Should handle API errors gracefully."""
        service, mock_client = vlm_service_with_mock_api

        # Mock API failure
        mock_client.chat.completions.create.side_effect = Exception("API timeout")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(valid_jpeg_bytes)
            temp_path = f.name

        try:
            cnn_top_3 = [("Persian", 0.85), ("Himalayan", 0.10), ("Exotic", 0.03)]
            status, prediction, reason = service.verify_prediction(temp_path, cnn_top_3)

            assert status == "error"
            # Falls back to CNN top-1
            assert prediction == "Persian"
            assert "timeout" in reason.lower()
        finally:
            Path(temp_path).unlink()


class TestConfigIntegration:
    """Test config module integration."""

    def test_is_vlm_available_with_key(self):
        """Should return True when key is set and enabled."""
        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key"}):
            from api.config import is_vlm_available, settings
            # Ensure vlm_enabled is True
            settings.vlm_enabled = True
            assert is_vlm_available() is True

    def test_is_vlm_available_without_key(self):
        """Should return False when key is not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove ZAI_API_KEY if present
            import os
            if "ZAI_API_KEY" in os.environ:
                del os.environ["ZAI_API_KEY"]
            from api.config import is_vlm_available
            assert is_vlm_available() is False

    def test_is_vlm_available_disabled(self):
        """Should return False when disabled in config."""
        with patch.dict("os.environ", {"ZAI_API_KEY": "test-key"}):
            from api.config import is_vlm_available, settings
            original = settings.vlm_enabled
            settings.vlm_enabled = False
            try:
                assert is_vlm_available() is False
            finally:
                settings.vlm_enabled = original
