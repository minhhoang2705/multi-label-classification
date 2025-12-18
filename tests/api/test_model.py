"""
Test suite for Model Endpoints
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModelEndpoints:
    """Test model information endpoints."""

    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "model_name" in data
        assert "num_classes" in data
        assert "device" in data
        assert "is_loaded" in data

    def test_model_info_num_classes(self, client):
        """Test model info returns correct number of classes."""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200

        data = response.json()
        if data["is_loaded"]:
            assert data["num_classes"] == 67

    def test_model_info_class_names(self, client):
        """Test model info includes class names."""
        response = client.get("/api/v1/model/info")
        assert response.status_code == 200

        data = response.json()
        assert "class_names" in data
        assert isinstance(data["class_names"], list)

        if data["is_loaded"]:
            assert len(data["class_names"]) == 67

    def test_model_classes(self, client):
        """Test model classes endpoint."""
        response = client.get("/api/v1/model/classes")
        assert response.status_code == 200

        data = response.json()
        assert "num_classes" in data
        assert "classes" in data

    def test_model_classes_structure(self, client):
        """Test model classes response structure."""
        response = client.get("/api/v1/model/classes")
        assert response.status_code == 200

        data = response.json()

        if data["num_classes"] > 0:
            assert len(data["classes"]) == data["num_classes"]

            # Check first class structure
            first_class = data["classes"][0]
            assert "id" in first_class
            assert "name" in first_class
            assert first_class["id"] == 0
            assert first_class["name"] == "Abyssinian"

    def test_model_classes_all_67_breeds(self, client):
        """Test model classes returns all 67 breeds."""
        response = client.get("/api/v1/model/classes")
        assert response.status_code == 200

        data = response.json()

        # Only check if model is loaded
        if data["num_classes"] == 67:
            assert len(data["classes"]) == 67

            # Check all IDs are unique
            ids = [c["id"] for c in data["classes"]]
            assert len(set(ids)) == 67

            # Check all names are unique
            names = [c["name"] for c in data["classes"]]
            assert len(set(names)) == 67

            # Check ID range
            assert min(ids) == 0
            assert max(ids) == 66
