"""
Test suite for InferenceService
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.services.inference_service import InferenceService


class TestInferenceService:
    """Test inference service."""

    def test_get_top_k_predictions(self, mock_probabilities, mock_class_names):
        """Test top-K prediction extraction."""
        predictions = InferenceService.get_top_k_predictions(
            probs=mock_probabilities,
            class_names=mock_class_names,
            k=5
        )

        assert len(predictions) == 5
        assert predictions[0].rank == 1
        assert predictions[0].class_name == "Abyssinian"
        assert predictions[0].confidence == pytest.approx(0.8)
        assert predictions[0].class_id == 0

    def test_get_top_k_ordering(self, mock_probabilities, mock_class_names):
        """Test predictions are ordered by confidence."""
        predictions = InferenceService.get_top_k_predictions(
            probs=mock_probabilities,
            class_names=mock_class_names,
            k=5
        )

        confidences = [p.confidence for p in predictions]
        assert confidences == sorted(confidences, reverse=True)

    def test_get_top_k_all_predictions(self, mock_probabilities, mock_class_names):
        """Test getting all 67 predictions."""
        predictions = InferenceService.get_top_k_predictions(
            probs=mock_probabilities,
            class_names=mock_class_names,
            k=67
        )

        assert len(predictions) == 67
        assert predictions[0].rank == 1
        assert predictions[66].rank == 67

    def test_synchronize_device_cuda(self):
        """Test CUDA synchronization called when device is CUDA."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Should not raise
            InferenceService.synchronize_device(device)

    def test_synchronize_device_cpu(self):
        """Test CPU synchronization is no-op."""
        device = torch.device("cpu")
        # Should not raise
        InferenceService.synchronize_device(device)
