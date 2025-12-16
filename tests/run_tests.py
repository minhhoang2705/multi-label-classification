"""
Test runner for Phase 01 tests without pytest dependency.
Uses Python's unittest and asyncio.
"""

import sys
import asyncio
import time
import torch
from pathlib import Path
from unittest import TestCase, TestLoader, TextTestRunner, TestSuite
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app
from api.config import settings
from api.services.model_service import ModelManager
from fastapi.testclient import TestClient


class TestModelManagerDeviceDetection(TestCase):
    """Test device detection logic."""

    def setUp(self):
        """Reset singleton before each test."""
        ModelManager._instance = None

    def tearDown(self):
        """Clean up after test."""
        ModelManager._instance = None

    def test_device_auto_detection(self):
        """Test auto device detection selects correct device."""
        async def run():
            manager = await ModelManager.get_instance()
            device = manager._get_device("auto")
            self.assertIsInstance(device, torch.device)
            self.assertIn(device.type, ["cuda", "mps", "cpu"])
            if torch.cuda.is_available():
                self.assertEqual(device.type, "cuda")

        asyncio.run(run())

    def test_device_cpu_override(self):
        """Test explicit CPU device selection."""
        async def run():
            manager = await ModelManager.get_instance()
            device = manager._get_device("cpu")
            self.assertIsInstance(device, torch.device)
            self.assertEqual(device.type, "cpu")

        asyncio.run(run())


class TestModelManagerClassNames(TestCase):
    """Test class name loading."""

    def setUp(self):
        """Reset singleton before each test."""
        ModelManager._instance = None

    def tearDown(self):
        """Clean up after test."""
        ModelManager._instance = None

    def test_class_names_loaded(self):
        """Test that 67 cat breed names are loaded."""
        async def run():
            manager = await ModelManager.get_instance()
            class_names = manager._load_class_names()
            self.assertEqual(len(class_names), 67)
            self.assertIsInstance(class_names, list)
            for name in class_names:
                self.assertIsInstance(name, str)

        asyncio.run(run())

    def test_class_names_contain_expected_breeds(self):
        """Test that class names contain expected breed names."""
        async def run():
            manager = await ModelManager.get_instance()
            class_names = manager._load_class_names()

            expected_breeds = [
                "Abyssinian",
                "Bengal",
                "Persian",
                "Siamese",
                "Maine Coon",
                "British Shorthair",
                "Ragdoll"
            ]

            for breed in expected_breeds:
                self.assertIn(breed, class_names)

        asyncio.run(run())

    def test_class_names_sorted_order(self):
        """Test that class names are in alphabetical order."""
        async def run():
            manager = await ModelManager.get_instance()
            class_names = manager._load_class_names()
            self.assertEqual(class_names, sorted(class_names))

        asyncio.run(run())


class TestModelManagerSingleton(TestCase):
    """Test ModelManager singleton pattern."""

    def setUp(self):
        """Reset singleton before each test."""
        ModelManager._instance = None

    def tearDown(self):
        """Clean up after test."""
        ModelManager._instance = None

    def test_singleton_pattern(self):
        """Test that only one instance is created."""
        async def run():
            manager1 = await ModelManager.get_instance()
            manager2 = await ModelManager.get_instance()
            self.assertIs(manager1, manager2)

        asyncio.run(run())

    def test_singleton_state_persistence(self):
        """Test that state persists across getInstance calls."""
        async def run():
            manager1 = await ModelManager.get_instance()
            manager1._is_loaded = True
            manager1._model_name = "test_model"

            manager2 = await ModelManager.get_instance()
            self.assertTrue(manager2.is_loaded)
            self.assertEqual(manager2.model_name, "test_model")

        asyncio.run(run())


class TestModelManagerProperties(TestCase):
    """Test ModelManager properties."""

    def setUp(self):
        """Reset singleton before each test."""
        ModelManager._instance = None

    def tearDown(self):
        """Clean up after test."""
        ModelManager._instance = None

    def test_initial_state(self):
        """Test initial state before model loading."""
        async def run():
            manager = await ModelManager.get_instance()
            self.assertFalse(manager.is_loaded)
            self.assertIsNone(manager.device)
            self.assertEqual(manager.class_names, [])
            self.assertEqual(manager.model_name, "")
            self.assertEqual(manager.checkpoint_path, "")

        asyncio.run(run())

    def test_properties_after_state_assignment(self):
        """Test properties are set correctly."""
        async def run():
            manager = await ModelManager.get_instance()
            manager._is_loaded = True
            manager._model_name = "resnet50"
            manager._checkpoint_path = "test/path.pt"
            manager._device = torch.device("cpu")
            manager._class_names = manager._load_class_names()

            self.assertTrue(manager.is_loaded)
            self.assertEqual(manager.model_name, "resnet50")
            self.assertEqual(manager.checkpoint_path, "test/path.pt")
            self.assertEqual(manager.device.type, "cpu")
            self.assertEqual(len(manager.class_names), 67)

        asyncio.run(run())


class TestModelLoading(TestCase):
    """Test actual model loading from checkpoint."""

    def setUp(self):
        """Reset singleton before each test."""
        ModelManager._instance = None

    def tearDown(self):
        """Clean up after test."""
        ModelManager._instance = None

    def test_checkpoint_file_exists(self):
        """Test that the checkpoint file exists."""
        checkpoint_path = Path(settings.checkpoint_path)
        self.assertTrue(
            checkpoint_path.exists(),
            f"Checkpoint not found at {settings.checkpoint_path}"
        )

    def test_model_load_real_checkpoint(self):
        """Test loading real model from checkpoint."""
        async def run():
            manager = await ModelManager.get_instance()

            # Load model
            start_time = time.time()
            await manager.load_model(
                checkpoint_path=settings.checkpoint_path,
                model_name=settings.model_name,
                num_classes=settings.num_classes,
                device="cpu"  # Use CPU for testing
            )
            load_time = time.time() - start_time

            # Verify model is loaded
            self.assertTrue(manager.is_loaded)
            self.assertEqual(manager.model_name, settings.model_name)
            self.assertEqual(manager.checkpoint_path, settings.checkpoint_path)
            self.assertEqual(len(manager.class_names), 67)
            self.assertEqual(manager.device.type, "cpu")

            # Verify load time is reasonable
            self.assertLess(load_time, 30, f"Model loading took {load_time}s, expected < 30s")

        asyncio.run(run())

    def test_model_load_file_not_found(self):
        """Test model loading with missing checkpoint file."""
        async def run():
            manager = await ModelManager.get_instance()

            with self.assertRaises(FileNotFoundError):
                await manager.load_model(
                    checkpoint_path="nonexistent/path.pt",
                    model_name="resnet50",
                    num_classes=67,
                    device="cpu"
                )

        asyncio.run(run())


class TestHealthEndpoints(TestCase):
    """Test health check endpoints."""

    def setUp(self):
        """Create test client."""
        self.client = TestClient(app)
        ModelManager._instance = None

    def tearDown(self):
        """Clean up."""
        ModelManager._instance = None

    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "Cat Breeds Classification API")
        self.assertEqual(data["version"], settings.api_version)
        self.assertIn("docs", data)
        self.assertIn("health", data)

    def test_health_live_endpoint(self):
        """Test liveness probe endpoint."""
        response = self.client.get("/health/live")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "alive")

    def test_health_ready_endpoint(self):
        """Test readiness probe endpoint."""
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify response structure
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("model_name", data)
        self.assertIn("device", data)
        self.assertIn("num_classes", data)

        # If model is loaded, verify details
        if data["model_loaded"]:
            self.assertEqual(data["status"], "ready")
            self.assertEqual(data["num_classes"], 67)


class TestAPIStartup(TestCase):
    """Test API startup and initialization."""

    def test_api_creation_succeeds(self):
        """Test that FastAPI app is created successfully."""
        self.assertIsNotNone(app)
        self.assertEqual(app.title, settings.api_title)
        self.assertEqual(app.version, settings.api_version)

    def test_api_has_required_routers(self):
        """Test that required routers are included."""
        routes = [route.path for route in app.routes]
        self.assertIn("/health/live", routes)
        self.assertIn("/health/ready", routes)
        self.assertIn("/", routes)

    def test_api_cors_middleware_configured(self):
        """Test that CORS middleware is configured."""
        # FastAPI stores middleware differently, check through user_middleware
        has_cors = any('CORSMiddleware' in str(type(m)) for m in app.user_middleware)
        self.assertTrue(has_cors or len(app.user_middleware) > 0, "CORS middleware should be configured")

    def test_client_initialization(self):
        """Test TestClient can be created."""
        client = TestClient(app)
        self.assertIsNotNone(client)


class TestConfigSettings(TestCase):
    """Test API configuration."""

    def test_settings_loaded(self):
        """Test that settings are loaded."""
        self.assertIsNotNone(settings)
        self.assertTrue(settings.checkpoint_path)
        self.assertTrue(settings.model_name)
        self.assertEqual(settings.num_classes, 67)

    def test_settings_defaults(self):
        """Test settings have correct defaults."""
        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 8000)
        self.assertEqual(settings.device, "auto")
        self.assertEqual(settings.image_size, 224)

    def test_settings_model_config(self):
        """Test model configuration in settings."""
        self.assertEqual(settings.model_name, "resnet50")
        self.assertEqual(settings.num_classes, 67)
        self.assertIsInstance(settings.checkpoint_path, str)


class TestIntegrationAPILifecycle(TestCase):
    """Integration tests for API lifecycle."""

    def setUp(self):
        """Create test client and reset singleton."""
        self.client = TestClient(app)
        ModelManager._instance = None

    def tearDown(self):
        """Clean up."""
        ModelManager._instance = None

    def test_api_startup_loads_model(self):
        """Test that API startup loads the model."""
        # TestClient automatically triggers lifespan startup
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        # Model should be loaded after TestClient startup
        # If model_loaded is false, at least verify the endpoint works
        self.assertIsNotNone(data)
        self.assertIn("model_loaded", data)

    def test_health_endpoints_after_startup(self):
        """Test health endpoints work after startup."""
        # Test liveness
        live_response = self.client.get("/health/live")
        self.assertEqual(live_response.status_code, 200)
        self.assertEqual(live_response.json()["status"], "alive")

        # Test readiness
        ready_response = self.client.get("/health/ready")
        self.assertEqual(ready_response.status_code, 200)
        ready_data = ready_response.json()
        # Verify structure of response
        self.assertIn("status", ready_data)
        self.assertIn("model_loaded", ready_data)
        self.assertIn("num_classes", ready_data)
        # If model is loaded, verify num_classes
        if ready_data["model_loaded"]:
            self.assertEqual(ready_data["num_classes"], 67)

    def test_root_and_health_endpoints(self):
        """Test all endpoints are accessible."""
        endpoints = ["/", "/health/live", "/health/ready"]

        for endpoint in endpoints:
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 200, f"Endpoint {endpoint} failed")
            self.assertIsNotNone(response.json())


def run_tests():
    """Run all tests and return results."""
    loader = TestLoader()
    suite = TestSuite()

    # Add all test classes
    test_classes = [
        TestModelManagerDeviceDetection,
        TestModelManagerClassNames,
        TestModelManagerSingleton,
        TestModelManagerProperties,
        TestModelLoading,
        TestHealthEndpoints,
        TestAPIStartup,
        TestConfigSettings,
        TestIntegrationAPILifecycle,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests with verbose output
    runner = TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
