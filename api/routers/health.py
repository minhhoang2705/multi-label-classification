"""
Health check endpoints for monitoring and readiness probes.
"""

from fastapi import APIRouter
from ..services.model_service import ModelManager


router = APIRouter()


@router.get("/health/live")
async def liveness():
    """
    Liveness probe endpoint.
    Returns 200 if application is running.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness():
    """
    Readiness probe endpoint.
    Returns model status and device information.
    """
    manager = await ModelManager.get_instance()

    return {
        "status": "ready" if manager.is_loaded else "not_ready",
        "model_loaded": manager.is_loaded,
        "model_name": manager.model_name if manager.is_loaded else None,
        "device": str(manager.device) if manager.device else None,
        "num_classes": len(manager.class_names) if manager.is_loaded else 0
    }
