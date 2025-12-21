"""
Prediction endpoint for cat breed classification.
"""

import time
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException

from ..models import PredictionResponse, ImageMetadata, ErrorResponse
from ..services.model_service import ModelManager
from ..services.image_service import ImageService
from ..services.inference_service import InferenceService
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
