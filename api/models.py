"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionItem(BaseModel):
    """Single prediction with class name and confidence."""
    rank: int = Field(..., ge=1, le=67, description="Prediction rank (1-67)")
    class_name: str = Field(..., description="Predicted cat breed name")
    class_id: int = Field(..., ge=0, le=66, description="Class index (0-66)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class ImageMetadata(BaseModel):
    """Metadata about the uploaded image."""
    original_width: int
    original_height: int
    format: str
    mode: str
    file_size_bytes: int
    filename: str


class PredictionResponse(BaseModel):
    """Response from the prediction endpoint."""
    # Top prediction
    predicted_class: str = Field(..., description="Top predicted breed")
    confidence: float = Field(..., description="Confidence of top prediction")

    # Top-5 predictions
    top_5_predictions: List[PredictionItem] = Field(
        ..., description="Top 5 predictions with confidence scores"
    )

    # Performance
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")

    # Metadata
    image_metadata: ImageMetadata
    model_info: dict = Field(
        default_factory=dict,
        description="Model information (name, device)"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str
    errors: Optional[List[dict]] = None
