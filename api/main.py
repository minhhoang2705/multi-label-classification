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
from .routers import health, predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan
)

# Add CORS middleware with restricted origins
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


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Cat Breeds Classification API",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health/live"
    }
