# =============================================================================
# Stage 1: Builder
# Purpose: Install Python dependencies with native extensions
# =============================================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS builder

WORKDIR /build

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# Purpose: Production-optimized execution environment
# =============================================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy installed packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser api/ /app/api/
COPY --chown=appuser:appuser src/ /app/src/

# Create directories for volumes
RUN mkdir -p /app/outputs/checkpoints /app/logs /home/appuser/.cache && \
    chown -R appuser:appuser /app/outputs /app/logs /home/appuser/.cache

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    API_CHECKPOINT_PATH=outputs/checkpoints/fold_0/best_model.pt \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Verify GPU access (build-time check)
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Expose port
EXPOSE 8000

# Run application
# Development: Single Uvicorn worker
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Production alternative (uncomment for Gunicorn):
# CMD ["gunicorn", "api.main:app", \
#      "--workers", "4", \
#      "--worker-class", "uvicorn.workers.UvicornWorker", \
#      "--bind", "0.0.0.0:8000", \
#      "--timeout", "120", \
#      "--graceful-timeout", "30", \
#      "--access-logfile", "-", \
#      "--error-logfile", "-"]
