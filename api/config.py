"""
API configuration using Pydantic Settings.

This module centralizes all configuration for the API server, including:
- Model settings (checkpoint path, architecture)
- Server settings (host, port, CORS)
- VLM settings (enable/disable, API keys)

All settings can be overridden via environment variables with API_ prefix.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """API configuration with environment variable support."""

    # Model configuration
    checkpoint_path: str = "outputs/checkpoints/fold_0/best_model.pt"
    model_name: str = "resnet50"
    num_classes: int = 67
    image_size: int = 224

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Device configuration (auto, cuda, mps, cpu)
    device: str = "auto"

    # CORS configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    cors_allow_credentials: bool = True

    # API metadata
    api_title: str = "Cat Breeds Classification API"
    api_version: str = "1.0.0"
    api_description: str = "Multi-class classification API for 67 cat breeds using ResNet50/EfficientNet"

    # VLM (Vision Language Model) configuration
    # When enabled, predictions are verified by GLM-4.6V for higher accuracy
    vlm_enabled: bool = True
    # Z.ai API key - get from https://docs.z.ai
    # Note: This is read from ZAI_API_KEY env var, not API_ZAI_API_KEY
    zai_api_key: Optional[str] = None

    class Config:
        env_prefix = "API_"
        case_sensitive = False
        # Allow reading ZAI_API_KEY without the API_ prefix
        extra = "allow"


# Global settings instance
settings = Settings()


def is_vlm_available() -> bool:
    """
    Check if VLM verification is available.

    VLM requires both:
    1. vlm_enabled = True in config
    2. ZAI_API_KEY environment variable set

    Returns:
        True if VLM can be used, False otherwise
    """
    import os
    return settings.vlm_enabled and bool(os.getenv("ZAI_API_KEY"))
