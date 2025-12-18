"""
Test suite for Health Endpoints
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_liveness(self, client):
        """Test liveness endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_liveness_response_structure(self, client):
        """Test liveness response has required fields."""
        response = client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert isinstance(data["status"], str)

    def test_readiness(self, client):
        """Test readiness endpoint."""
        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ["ready", "not_ready"]
        assert "model_loaded" in data

    def test_readiness_model_loaded_field(self, client):
        """Test readiness returns model_loaded boolean."""
        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data["model_loaded"], bool)
