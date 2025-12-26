"""
Hybrid CNN + VLM inference service with disagreement detection.

This module combines predictions from both the CNN classifier and VLM (Vision
Language Model) to provide more accurate breed identification, especially for
edge cases where the CNN might be confident but wrong.

How it works:
1. CNN predicts top-5 breeds with confidence scores
2. VLM analyzes the same image given CNN's top-3 candidates
3. If they agree → high confidence result
4. If they disagree → VLM's prediction is trusted (VLM better at visual reasoning)
5. If VLM fails → fallback to CNN-only

Why this approach?
- CNNs can misclassify visually similar breeds (Persian vs Himalayan)
- VLMs can reason about specific visual features (coat pattern, face shape)
- Disagreements indicate uncertain cases that need human review
"""

import logging
import time
import json
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .model_service import ModelManager
from .vlm_service import VLMService
from .inference_service import InferenceService

logger = logging.getLogger(__name__)

# Separate logger for disagreement cases (outputs to JSONL file)
disagreement_logger = logging.getLogger('disagreements')


@dataclass
class HybridPrediction:
    """
    Result of hybrid CNN + VLM prediction.

    This combines outputs from both models and includes metadata about
    whether they agreed or disagreed.
    """
    # CNN results
    cnn_prediction: str
    cnn_confidence: float
    cnn_top_5: list

    # VLM results (may be None if VLM disabled/failed)
    vlm_prediction: Optional[str]
    vlm_reasoning: Optional[str]

    # Agreement status
    # - "verified": CNN and VLM agree → high confidence
    # - "uncertain": CNN and VLM disagree → VLM wins, medium confidence
    # - "cnn_only": VLM disabled/failed → CNN only, low confidence
    # - "unclear": VLM response unparseable → CNN fallback, low confidence
    # - "error": VLM errored → CNN fallback, low confidence
    status: str

    # Final recommendation (VLM wins on disagreement)
    final_prediction: str
    final_confidence: str  # "high", "medium", "low"

    # Timing breakdown
    cnn_time_ms: float
    vlm_time_ms: Optional[float]


class HybridInferenceService:
    """
    Service combining CNN + VLM predictions with disagreement detection.

    This service runs both models and intelligently combines their outputs,
    preferring VLM predictions when there's disagreement because VLMs are
    better at visual reasoning for edge cases.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        vlm_service: Optional[VLMService] = None,
        vlm_enabled: bool = True
    ):
        """
        Initialize hybrid inference service.

        Args:
            model_manager: CNN model manager (always required)
            vlm_service: VLM service instance (optional, may be None if disabled)
            vlm_enabled: Whether to use VLM verification
        """
        self.model_manager = model_manager
        self.vlm_service = vlm_service
        # VLM only enabled if both flag is True AND service provided
        self.vlm_enabled = vlm_enabled and vlm_service is not None

    async def predict(
        self,
        image_tensor,
        image_path: str
    ) -> HybridPrediction:
        """
        Run hybrid CNN + VLM prediction.

        Decision logic:
        - When agreeing → use CNN prediction with "high" confidence
        - When disagreeing → use VLM prediction with "medium" confidence
        - When VLM fails → use CNN prediction with "low" confidence

        Args:
            image_tensor: Preprocessed image tensor for CNN
            image_path: Path to original image file for VLM

        Returns:
            HybridPrediction with combined results and agreement status
        """
        # Stage 1: CNN prediction (always runs)
        cnn_start = time.perf_counter()
        probs, _ = self.model_manager.predict(image_tensor)
        InferenceService.synchronize_device(self.model_manager.device)
        cnn_time_ms = (time.perf_counter() - cnn_start) * 1000

        # Get top-5 predictions from CNN
        top_5 = InferenceService.get_top_k_predictions(
            probs=probs,
            class_names=self.model_manager.class_names,
            k=5
        )

        cnn_prediction = top_5[0].class_name
        cnn_confidence = top_5[0].confidence

        # Prepare top-3 for VLM prompt (don't overwhelm VLM with too many options)
        cnn_top_3: List[Tuple[str, float]] = [
            (p.class_name, p.confidence) for p in top_5[:3]
        ]

        # Stage 2: VLM verification (if enabled)
        if not self.vlm_enabled:
            # VLM disabled - return CNN-only result
            return HybridPrediction(
                cnn_prediction=cnn_prediction,
                cnn_confidence=cnn_confidence,
                cnn_top_5=top_5,
                vlm_prediction=None,
                vlm_reasoning=None,
                status="cnn_only",
                final_prediction=cnn_prediction,
                final_confidence="low",  # Lower confidence without VLM verification
                cnn_time_ms=cnn_time_ms,
                vlm_time_ms=None
            )

        # Run VLM verification
        vlm_start = time.perf_counter()
        try:
            status, vlm_pred, reasoning = self.vlm_service.verify_prediction(
                image_path=image_path,
                cnn_top_3=cnn_top_3
            )
            vlm_time_ms = (time.perf_counter() - vlm_start) * 1000
        except Exception as e:
            # VLM call failed - log and fallback to CNN
            logger.error(f"VLM verification failed: {e}")
            return HybridPrediction(
                cnn_prediction=cnn_prediction,
                cnn_confidence=cnn_confidence,
                cnn_top_5=top_5,
                vlm_prediction=None,
                vlm_reasoning=str(e),
                status="error",
                final_prediction=cnn_prediction,
                final_confidence="low",
                cnn_time_ms=cnn_time_ms,
                vlm_time_ms=None
            )

        # Stage 3: Agreement check - decide final prediction
        if status == "agree":
            # CNN and VLM agree - high confidence result
            result = HybridPrediction(
                cnn_prediction=cnn_prediction,
                cnn_confidence=cnn_confidence,
                cnn_top_5=top_5,
                vlm_prediction=vlm_pred,
                vlm_reasoning=reasoning,
                status="verified",
                final_prediction=cnn_prediction,
                final_confidence="high",
                cnn_time_ms=cnn_time_ms,
                vlm_time_ms=vlm_time_ms
            )
        elif status == "disagree":
            # VLM disagrees - USE VLM PREDICTION AS FINAL
            # This is the key decision: trust VLM over CNN on disagreement
            result = HybridPrediction(
                cnn_prediction=cnn_prediction,
                cnn_confidence=cnn_confidence,
                cnn_top_5=top_5,
                vlm_prediction=vlm_pred,
                vlm_reasoning=reasoning,
                status="uncertain",
                final_prediction=vlm_pred,  # VLM wins!
                final_confidence="medium",
                cnn_time_ms=cnn_time_ms,
                vlm_time_ms=vlm_time_ms
            )
            # Log disagreement for later analysis
            self._log_disagreement(result, image_path)
        else:
            # Unclear VLM response - fallback to CNN
            result = HybridPrediction(
                cnn_prediction=cnn_prediction,
                cnn_confidence=cnn_confidence,
                cnn_top_5=top_5,
                vlm_prediction=vlm_pred,
                vlm_reasoning=reasoning,
                status="unclear",
                final_prediction=cnn_prediction,
                final_confidence="low",
                cnn_time_ms=cnn_time_ms,
                vlm_time_ms=vlm_time_ms
            )

        return result

    def _log_disagreement(self, result: HybridPrediction, image_path: str):
        """
        Log disagreement case to JSONL file for post-hoc analysis.

        Why log disagreements?
        - Helps identify systematic CNN errors
        - Enables VLM accuracy evaluation
        - Provides data for model improvement

        Args:
            result: HybridPrediction with disagreement
            image_path: Path to the image (not logged for security)
        """
        try:
            import hashlib
            # Hash the path instead of logging it (avoid temp path exposure)
            path_hash = hashlib.sha256(image_path.encode()).hexdigest()[:16]

            log_entry = {
                "timestamp": time.time(),
                "image_hash": path_hash,  # Hash instead of path (security)
                "cnn_prediction": result.cnn_prediction,
                "cnn_confidence": result.cnn_confidence,
                "vlm_prediction": result.vlm_prediction,
                "vlm_reasoning": result.vlm_reasoning,
                "final_prediction": result.final_prediction
            }
            # Log as JSON (one line per disagreement)
            disagreement_logger.info(json.dumps(log_entry))
        except Exception as e:
            # Don't crash on logging failure
            logger.warning(f"Failed to log disagreement: {e}")
