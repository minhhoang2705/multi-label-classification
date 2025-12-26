"""
Prediction endpoint for cat breed classification.
"""

import os
import tempfile
import time
import logging
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException

logger = logging.getLogger(__name__)

from ..models import (
    PredictionResponse,
    HybridPredictionResponse,
    ImageMetadata,
    ErrorResponse
)
from ..services.model_service import ModelManager
from ..services.image_service import ImageService
from ..services.inference_service import InferenceService
from ..services.vlm_service import VLMService
from ..services.hybrid_inference_service import HybridInferenceService
from ..dependencies import get_image_service

router = APIRouter()


async def get_model_manager() -> ModelManager:
    """
    Dependency to get model manager instance.

    Returns:
        ModelManager singleton instance

    Raises:
        HTTPException: If model is not loaded (503)
    """
    manager = await ModelManager.get_instance()
    if not manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    return manager


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        413: {"model": ErrorResponse, "description": "Image too large"},
        503: {"model": ErrorResponse, "description": "Model not ready"}
    }
)
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    image_service: ImageService = Depends(get_image_service),
    model_manager: ModelManager = Depends(get_model_manager)
) -> PredictionResponse:
    """
    Predict cat breed from uploaded image.

    Returns top prediction with confidence score and top-5 predictions.

    Args:
        file: Uploaded image file
        image_service: Image validation service (injected)
        model_manager: Model manager singleton (injected)

    Returns:
        PredictionResponse with predictions, metadata, and timing

    Raises:
        HTTPException: Various validation and processing errors
    """
    # 1. Validate and preprocess image
    tensor, metadata = await image_service.validate_and_preprocess(file)

    # 2. Run inference with timing
    start_time = time.perf_counter()

    probs, _ = model_manager.predict(tensor)

    # Synchronize for accurate timing
    InferenceService.synchronize_device(model_manager.device)

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    # 3. Get top-5 predictions
    top_5 = InferenceService.get_top_k_predictions(
        probs=probs,
        class_names=model_manager.class_names,
        k=5
    )

    # 4. Build response
    return PredictionResponse(
        predicted_class=top_5[0].class_name,
        confidence=top_5[0].confidence,
        top_5_predictions=top_5,
        inference_time_ms=round(inference_time_ms, 3),
        image_metadata=ImageMetadata(**metadata),
        model_info={
            "model_name": model_manager.model_name,
            "device": str(model_manager.device),
            "num_classes": len(model_manager.class_names)
        }
    )


@router.post(
    "/predict/verified",
    response_model=HybridPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        413: {"model": ErrorResponse, "description": "Image too large"},
        503: {"model": ErrorResponse, "description": "Model not ready"}
    }
)
async def predict_verified(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    image_service: ImageService = Depends(get_image_service),
    model_manager: ModelManager = Depends(get_model_manager)
) -> HybridPredictionResponse:
    """
    Predict cat breed with VLM verification.

    This endpoint combines CNN predictions with VLM (Vision Language Model)
    verification to catch edge cases where the CNN might be wrong.

    How it works:
    1. CNN predicts top-5 breeds
    2. VLM analyzes image given CNN's top-3 candidates
    3. Returns prediction with agreement status:
       - "verified": CNN and VLM agree → high confidence
       - "uncertain": CNN and VLM disagree → uses VLM prediction (medium confidence)
       - "cnn_only": VLM disabled/failed → fallback to CNN (low confidence)

    Args:
        file: Uploaded image file
        image_service: Image validation service (injected)
        model_manager: Model manager singleton (injected)

    Returns:
        HybridPredictionResponse with combined predictions and metadata

    Raises:
        HTTPException: Various validation and processing errors
    """
    start_time = time.perf_counter()

    # Save temp file for VLM (VLM needs file path, not tensor)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Reset file pointer for CNN preprocessing
    await file.seek(0)

    try:
        # 1. Preprocess image for CNN
        tensor, metadata = await image_service.validate_and_preprocess(file)

        # 2. Get VLM service (may be None if disabled/not configured)
        try:
            vlm_service = VLMService.get_instance()
        except (ValueError, ImportError) as e:
            # VLM not available (missing API key or SDK not installed)
            vlm_service = None

        # 3. Run hybrid inference
        hybrid_service = HybridInferenceService(model_manager, vlm_service)
        result = await hybrid_service.predict(tensor, tmp_path)

        total_time_ms = (time.perf_counter() - start_time) * 1000

        # 4. Build response
        return HybridPredictionResponse(
            predicted_class=result.final_prediction,
            confidence_level=result.final_confidence,
            verification_status=result.status,
            cnn_prediction=result.cnn_prediction,
            cnn_confidence=result.cnn_confidence,
            top_5_predictions=result.cnn_top_5,
            vlm_prediction=result.vlm_prediction,
            vlm_reasoning=result.vlm_reasoning,
            cnn_time_ms=round(result.cnn_time_ms, 3),
            vlm_time_ms=round(result.vlm_time_ms, 3) if result.vlm_time_ms else None,
            total_time_ms=round(total_time_ms, 3),
            image_metadata=ImageMetadata(**metadata),
            model_info={
                "cnn_model": model_manager.model_name,
                "vlm_model": "glm-4.6v" if vlm_service else None,
                "device": str(model_manager.device)
            }
        )
    finally:
        # Always cleanup temp file
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            # File already deleted (ok)
            pass
        except (PermissionError, OSError) as e:
            # Log but don't crash on cleanup failure
            logger.warning(f"Failed to cleanup temp file {tmp_path}: {e}")
