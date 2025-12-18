#!/bin/bash
# Run API tests

set -e

echo "================================================="
echo "Running API Test Suite - Phase 05"
echo "================================================="
echo ""

# Unit tests (fast, no model loading)
echo "1. Running unit tests (ImageService, InferenceService)..."
pytest tests/api/test_image_service.py tests/api/test_inference_service.py -v

echo ""
echo "================================================="
echo ""

# Integration tests (requires model)
echo "2. Running integration tests (Health, Predict, Model endpoints)..."
pytest tests/api/test_health.py tests/api/test_predict.py tests/api/test_model.py -v

echo ""
echo "================================================="
echo ""

# Coverage report
echo "3. Running with coverage..."
pytest tests/api/ --cov=api --cov-report=term-missing --cov-report=html

echo ""
echo "================================================="
echo "Test suite completed!"
echo "================================================="
echo ""
echo "Coverage report: htmlcov/index.html"
