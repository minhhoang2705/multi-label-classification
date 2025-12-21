"""
Inference service for running model predictions and formatting results.
"""

import numpy as np
import torch
from typing import List

from ..models import PredictionItem


class InferenceService:
    """Service for running model inference and formatting predictions."""

    @staticmethod
    def get_top_k_predictions(
        probs: np.ndarray,
        class_names: List[str],
        k: int = 5
    ) -> List[PredictionItem]:
        """
        Get top-K predictions from probability array.

        Args:
            probs: Probability array of shape (1, num_classes)
            class_names: List of class names
            k: Number of top predictions to return

        Returns:
            List of PredictionItem sorted by confidence (highest first)
        """
        # probs shape: (1, 67) -> squeeze to (67,)
        probs_1d = probs.squeeze()

        # Get top-k indices (argsort descending)
        top_k_indices = np.argsort(probs_1d)[::-1][:k]

        predictions = []
        for rank, idx in enumerate(top_k_indices, start=1):
            predictions.append(PredictionItem(
                rank=rank,
                class_name=class_names[idx],
                class_id=int(idx),
                confidence=float(probs_1d[idx])
            ))

        return predictions

    @staticmethod
    def synchronize_device(device: torch.device):
        """
        Synchronize CUDA device for accurate timing.

        Args:
            device: torch.device to synchronize
        """
        if device.type == "cuda":
            torch.cuda.synchronize()
