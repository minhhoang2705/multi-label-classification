"""
Model information and metadata endpoints.
"""

import json
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException

from ..models import (
    ModelInfoResponse,
    ClassListResponse,
    ClassInfo,
    PerformanceMetrics,
    SpeedMetrics
)
from ..services.model_service import ModelManager
from ..config import settings

logger = logging.getLogger(__name__)

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


def load_test_metrics(checkpoint_path: str) -> dict:
    """
    Load pre-computed test metrics if available.

    Derives metrics path from checkpoint path:
    outputs/checkpoints/fold_0/best_model.pt
    -> outputs/test_results/fold_0/val/test_metrics.json

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        dict with metrics data, or empty dict if not found
    """
    try:
        cp_path = Path(checkpoint_path)
        fold = cp_path.parent.name  # e.g., "fold_0"
        metrics_path = Path("outputs/test_results") / fold / "val" / "test_metrics.json"

        if metrics_path.exists():
            logger.info(f"Loading test metrics from {metrics_path}")
            with open(metrics_path) as f:
                return json.load(f)
        else:
            logger.warning(f"Test metrics not found at {metrics_path}")
    except Exception as e:
        logger.error(f"Error loading test metrics: {e}")

    return {}


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(
    model_manager: ModelManager = Depends(get_model_manager)
) -> ModelInfoResponse:
    """
    Get model information and performance metrics.

    Returns model metadata, validation performance, and inference speed.

    Returns:
        ModelInfoResponse with all model details
    """
    # Load test metrics
    test_data = load_test_metrics(settings.checkpoint_path)

    performance_metrics = None
    speed_metrics = None

    # Extract performance metrics
    if test_data.get("metrics"):
        m = test_data["metrics"]
        performance_metrics = PerformanceMetrics(
            accuracy=m.get("accuracy", 0.0),
            balanced_accuracy=m.get("balanced_accuracy", 0.0),
            f1_macro=m.get("f1_macro", 0.0),
            f1_weighted=m.get("f1_weighted", 0.0),
            top_3_accuracy=m.get("top_3_accuracy", 0.0),
            top_5_accuracy=m.get("top_5_accuracy", 0.0)
        )

    # Extract speed metrics
    if test_data.get("speed_metrics"):
        s = test_data["speed_metrics"]
        speed_metrics = SpeedMetrics(
            avg_time_per_sample_ms=s.get("avg_time_per_sample_ms", 0.0),
            throughput_samples_per_sec=s.get("throughput_samples_per_sec", 0.0),
            device=s.get("device", str(model_manager.device))
        )

    return ModelInfoResponse(
        model_name=model_manager.model_name,
        num_classes=len(model_manager.class_names),
        image_size=224,  # From config
        checkpoint_path=settings.checkpoint_path,
        device=str(model_manager.device),
        is_loaded=model_manager.is_loaded,
        performance_metrics=performance_metrics,
        speed_metrics=speed_metrics,
        class_names=model_manager.class_names
    )


@router.get("/model/classes", response_model=ClassListResponse)
async def list_classes(
    model_manager: ModelManager = Depends(get_model_manager)
) -> ClassListResponse:
    """
    List all supported cat breed classes.

    Returns class IDs and names for reference.

    Returns:
        ClassListResponse with all 67 breeds
    """
    classes = [
        ClassInfo(id=i, name=name)
        for i, name in enumerate(model_manager.class_names)
    ]

    return ClassListResponse(
        num_classes=len(classes),
        classes=classes
    )
