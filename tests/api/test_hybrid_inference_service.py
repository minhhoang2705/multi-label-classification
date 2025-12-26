"""
Test suite for HybridInferenceService.

These tests verify the hybrid CNN + VLM prediction logic WITHOUT making
real API calls. We mock both the CNN model and VLM service to test the
disagreement detection logic.

Key test scenarios:
1. Agreement (CNN == VLM) → verified, high confidence
2. Disagreement (CNN != VLM) → uncertain, VLM wins, medium confidence
3. VLM disabled → cnn_only, low confidence
4. VLM error → error status, CNN fallback, low confidence
5. Unclear VLM response → unclear status, CNN fallback
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.services.hybrid_inference_service import (
    HybridInferenceService,
    HybridPrediction
)
from api.models import PredictionItem


class TestHybridInferenceService:
    """Test hybrid inference service combining CNN + VLM."""

    @pytest.fixture
    def mock_model_manager(self, mock_class_names):
        """Create mock ModelManager."""
        manager = Mock()
        manager.class_names = mock_class_names
        manager.device = Mock(type="cpu")

        # Mock predict to return probabilities
        probs = np.zeros((1, 67))
        probs[0, 0] = 0.85  # Abyssinian (CNN top prediction)
        probs[0, 1] = 0.10  # American Bobtail
        probs[0, 2] = 0.03  # American Curl
        manager.predict = Mock(return_value=(probs, None))

        return manager

    @pytest.fixture
    def mock_vlm_service_agree(self):
        """Mock VLM service that agrees with CNN."""
        service = Mock()
        # VLM agrees: returns "agree" status with same breed as CNN
        service.verify_prediction = Mock(
            return_value=("agree", "Abyssinian", "Long ears and ticked coat")
        )
        return service

    @pytest.fixture
    def mock_vlm_service_disagree(self):
        """Mock VLM service that disagrees with CNN."""
        service = Mock()
        # VLM disagrees: returns "disagree" status with different breed
        service.verify_prediction = Mock(
            return_value=("disagree", "Bengal", "Spotted coat pattern, not ticked")
        )
        return service

    @pytest.fixture
    def mock_vlm_service_unclear(self):
        """Mock VLM service that returns unclear response."""
        service = Mock()
        service.verify_prediction = Mock(
            return_value=("unclear", None, "Could not parse breed")
        )
        return service

    @pytest.fixture
    def mock_vlm_service_error(self):
        """Mock VLM service that raises exception."""
        service = Mock()
        service.verify_prediction = Mock(
            side_effect=Exception("API timeout")
        )
        return service

    @pytest.fixture
    def temp_image_path(self, valid_jpeg_bytes):
        """Create temporary image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(valid_jpeg_bytes)
            temp_path = f.name
        yield temp_path
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def mock_image_tensor(self):
        """Create mock image tensor."""
        import torch
        return torch.randn(1, 3, 224, 224)


class TestAgreementScenario(TestHybridInferenceService):
    """Test when CNN and VLM agree."""

    @pytest.mark.asyncio
    async def test_agreement_returns_verified_status(
        self,
        mock_model_manager,
        mock_vlm_service_agree,
        mock_image_tensor,
        temp_image_path
    ):
        """When CNN and VLM agree, should return 'verified' status."""
        service = HybridInferenceService(
            mock_model_manager,
            mock_vlm_service_agree,
            vlm_enabled=True
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        assert result.status == "verified"
        assert result.final_prediction == "Abyssinian"
        assert result.final_confidence == "high"

    @pytest.mark.asyncio
    async def test_agreement_includes_vlm_reasoning(
        self,
        mock_model_manager,
        mock_vlm_service_agree,
        mock_image_tensor,
        temp_image_path
    ):
        """Should include VLM reasoning when agreeing."""
        service = HybridInferenceService(
            mock_model_manager,
            mock_vlm_service_agree,
            vlm_enabled=True
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        assert result.vlm_prediction == "Abyssinian"
        assert "Long ears" in result.vlm_reasoning
        assert result.vlm_time_ms is not None


class TestDisagreementScenario(TestHybridInferenceService):
    """Test when CNN and VLM disagree."""

    @pytest.mark.asyncio
    async def test_disagreement_returns_uncertain_status(
        self,
        mock_model_manager,
        mock_vlm_service_disagree,
        mock_image_tensor,
        temp_image_path
    ):
        """When disagreeing, should return 'uncertain' status."""
        service = HybridInferenceService(
            mock_model_manager,
            mock_vlm_service_disagree,
            vlm_enabled=True
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        assert result.status == "uncertain"
        assert result.final_confidence == "medium"

    @pytest.mark.asyncio
    async def test_disagreement_vlm_wins(
        self,
        mock_model_manager,
        mock_vlm_service_disagree,
        mock_image_tensor,
        temp_image_path
    ):
        """When disagreeing, VLM prediction should be the final result."""
        service = HybridInferenceService(
            mock_model_manager,
            mock_vlm_service_disagree,
            vlm_enabled=True
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        # VLM wins on disagreement
        assert result.cnn_prediction == "Abyssinian"
        assert result.vlm_prediction == "Bengal"
        assert result.final_prediction == "Bengal"  # VLM prediction

    @pytest.mark.asyncio
    @patch('api.services.hybrid_inference_service.disagreement_logger')
    async def test_disagreement_logged(
        self,
        mock_logger,
        mock_model_manager,
        mock_vlm_service_disagree,
        mock_image_tensor,
        temp_image_path
    ):
        """Disagreements should be logged for analysis."""
        service = HybridInferenceService(
            mock_model_manager,
            mock_vlm_service_disagree,
            vlm_enabled=True
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        # Should have logged the disagreement
        assert mock_logger.info.called


class TestVLMDisabledScenario(TestHybridInferenceService):
    """Test when VLM is disabled or unavailable."""

    @pytest.mark.asyncio
    async def test_vlm_disabled_returns_cnn_only(
        self,
        mock_model_manager,
        mock_image_tensor,
        temp_image_path
    ):
        """When VLM disabled, should return 'cnn_only' status."""
        # Create service with VLM disabled
        service = HybridInferenceService(
            mock_model_manager,
            vlm_service=None,
            vlm_enabled=False
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        assert result.status == "cnn_only"
        assert result.final_prediction == "Abyssinian"  # CNN prediction
        assert result.final_confidence == "low"  # Lower without VLM
        assert result.vlm_prediction is None
        assert result.vlm_time_ms is None

    @pytest.mark.asyncio
    async def test_vlm_service_none_but_enabled_flag_true(
        self,
        mock_model_manager,
        mock_image_tensor,
        temp_image_path
    ):
        """When VLM service is None but flag is True, should still disable VLM."""
        service = HybridInferenceService(
            mock_model_manager,
            vlm_service=None,
            vlm_enabled=True  # Flag says enabled but service is None
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        # Should treat as disabled
        assert result.status == "cnn_only"


class TestVLMErrorScenario(TestHybridInferenceService):
    """Test error handling when VLM fails."""

    @pytest.mark.asyncio
    async def test_vlm_error_returns_error_status(
        self,
        mock_model_manager,
        mock_vlm_service_error,
        mock_image_tensor,
        temp_image_path
    ):
        """When VLM fails, should return 'error' status and fallback to CNN."""
        service = HybridInferenceService(
            mock_model_manager,
            mock_vlm_service_error,
            vlm_enabled=True
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        assert result.status == "error"
        assert result.final_prediction == "Abyssinian"  # CNN fallback
        assert result.final_confidence == "low"
        assert "API timeout" in result.vlm_reasoning


class TestUnclearVLMResponse(TestHybridInferenceService):
    """Test when VLM response is unclear."""

    @pytest.mark.asyncio
    async def test_unclear_response_falls_back_to_cnn(
        self,
        mock_model_manager,
        mock_vlm_service_unclear,
        mock_image_tensor,
        temp_image_path
    ):
        """When VLM response unclear, should fallback to CNN."""
        service = HybridInferenceService(
            mock_model_manager,
            mock_vlm_service_unclear,
            vlm_enabled=True
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        assert result.status == "unclear"
        assert result.final_prediction == "Abyssinian"  # CNN fallback
        assert result.final_confidence == "low"


class TestTimingMetrics(TestHybridInferenceService):
    """Test timing metrics are captured correctly."""

    @pytest.mark.asyncio
    async def test_cnn_timing_recorded(
        self,
        mock_model_manager,
        mock_vlm_service_agree,
        mock_image_tensor,
        temp_image_path
    ):
        """Should record CNN inference time."""
        service = HybridInferenceService(
            mock_model_manager,
            mock_vlm_service_agree,
            vlm_enabled=True
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        assert result.cnn_time_ms > 0
        assert isinstance(result.cnn_time_ms, float)

    @pytest.mark.asyncio
    async def test_vlm_timing_recorded(
        self,
        mock_model_manager,
        mock_vlm_service_agree,
        mock_image_tensor,
        temp_image_path
    ):
        """Should record VLM verification time."""
        service = HybridInferenceService(
            mock_model_manager,
            mock_vlm_service_agree,
            vlm_enabled=True
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        assert result.vlm_time_ms is not None
        assert result.vlm_time_ms > 0


class TestCNNResults(TestHybridInferenceService):
    """Test CNN results are always included."""

    @pytest.mark.asyncio
    async def test_cnn_top_5_always_present(
        self,
        mock_model_manager,
        mock_vlm_service_agree,
        mock_image_tensor,
        temp_image_path
    ):
        """Should always include CNN's top-5 predictions."""
        service = HybridInferenceService(
            mock_model_manager,
            mock_vlm_service_agree,
            vlm_enabled=True
        )

        result = await service.predict(mock_image_tensor, temp_image_path)

        assert len(result.cnn_top_5) == 5
        assert result.cnn_top_5[0].class_name == "Abyssinian"
        assert result.cnn_confidence == result.cnn_top_5[0].confidence
