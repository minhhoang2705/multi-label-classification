"""
Dependency injection factories for FastAPI.
"""

from functools import lru_cache
from .services.image_service import ImageService
from .services.model_service import ModelManager
from .config import Settings


@lru_cache
def get_settings() -> Settings:
    """Get settings singleton."""
    return Settings()


@lru_cache
def get_image_service() -> ImageService:
    """Get image service singleton."""
    settings = get_settings()
    return ImageService(image_size=settings.image_size)


async def get_model_manager() -> ModelManager:
    """Get model manager singleton."""
    return await ModelManager.get_instance()
