"""
API configuration using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from typing import List


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

    class Config:
        env_prefix = "API_"
        case_sensitive = False


# Global settings instance
settings = Settings()
