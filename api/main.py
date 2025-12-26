"""
FastAPI application with model loading and inference endpoints.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .config import settings
from .services.model_service import ModelManager
from .routers import health, predict, model
from .middleware import VersionHeaderMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure disagreement logger (outputs to JSONL file)
disagreement_logger = logging.getLogger('disagreements')
disagreement_logger.setLevel(logging.INFO)
# Create logs directory if it doesn't exist
import os
os.makedirs('logs', exist_ok=True)
# Add file handler for JSONL output
disagreement_handler = logging.FileHandler('logs/disagreements.jsonl')
disagreement_handler.setLevel(logging.INFO)
# Simple format - just the message (already JSON)
disagreement_handler.setFormatter(logging.Formatter('%(message)s'))
disagreement_logger.addHandler(disagreement_handler)
# Don't propagate to root logger (avoid double logging)
disagreement_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Handles startup and shutdown events.
    """
    # Startup: Load model
    logger.info("="*60)
    logger.info("Starting Cat Breeds Classification API")
    logger.info("="*60)

    manager = await ModelManager.get_instance()
    try:
        await manager.load_model(
            checkpoint_path=settings.checkpoint_path,
            model_name=settings.model_name,
            num_classes=settings.num_classes,
            device=settings.device
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise

    logger.info("="*60)
    logger.info(f"API ready on http://{settings.host}:{settings.port}")
    logger.info("="*60)

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title="Cat Breeds Classification API",
    description="""
## Cat Breed Image Classification API

Predict cat breeds from uploaded images using deep learning.

### Features
- **67 cat breeds** supported
- **Top-5 predictions** with confidence scores
- **Fast inference** (~0.9ms/image on GPU)
- **Image validation** (JPEG, PNG, WebP)

### Model Information
- Architecture: ResNet50 / EfficientNet (TIMM)
- Input size: 224x224
- Normalization: ImageNet statistics

### Usage
1. Upload an image via POST /api/v1/predict
2. Receive predicted breed with confidence scores
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middlewares
# Version headers (add first to be outermost)
app.add_middleware(VersionHeaderMiddleware)

# CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Invalid request",
            "errors": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error"
        }
    )


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(model.router, prefix="/api/v1", tags=["Model"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Cat Breeds Classification API",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health/live"
    }
