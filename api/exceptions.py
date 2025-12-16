"""
Custom exceptions for the API.
"""

from fastapi import HTTPException


class ImageValidationError(HTTPException):
    """Raised when image validation fails."""

    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)


class ImageTooLargeError(HTTPException):
    """Raised when image file size exceeds limit."""

    def __init__(self, detail: str):
        super().__init__(status_code=413, detail=detail)


class ModelInferenceError(HTTPException):
    """Raised when model inference fails."""

    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail)


class ModelNotLoadedError(HTTPException):
    """Raised when model is not loaded yet."""

    def __init__(self, detail: str = "Model not loaded. Please wait for startup to complete."):
        super().__init__(status_code=503, detail=detail)
