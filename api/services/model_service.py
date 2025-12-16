"""
Model management service with singleton pattern for inference.
"""

import logging
import torch
import torch.nn as nn
import timm
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for model loading and inference.
    Ensures single model instance across application lifecycle.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self):
        """Initialize ModelManager with empty state."""
        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None
        self._class_names: List[str] = []
        self._is_loaded: bool = False
        self._model_name: str = ""
        self._checkpoint_path: str = ""

    @classmethod
    async def get_instance(cls) -> "ModelManager":
        """
        Get singleton instance of ModelManager.

        Returns:
            ModelManager instance
        """
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def _get_device(self, device_str: str) -> torch.device:
        """
        Auto-detect or parse device string.

        Args:
            device_str: Device string (auto, cuda, mps, cpu)

        Returns:
            torch.device instance
        """
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    def _load_class_names(self) -> List[str]:
        """
        Load 67 cat breed class names in sorted order.

        Returns:
            List of breed names
        """
        return [
            "Abyssinian",
            "American Bobtail",
            "American Curl",
            "American Shorthair",
            "American Wirehair",
            "Applehead Siamese",
            "Balinese",
            "Bengal",
            "Birman",
            "Bombay",
            "British Shorthair",
            "Burmese",
            "Burmilla",
            "Calico",
            "Canadian Hairless",
            "Chartreux",
            "Chausie",
            "Chinchilla",
            "Cornish Rex",
            "Cymric",
            "Devon Rex",
            "Dilute Calico",
            "Dilute Tortoiseshell",
            "Domestic Long Hair",
            "Domestic Medium Hair",
            "Domestic Short Hair",
            "Egyptian Mau",
            "Exotic Shorthair",
            "Extra-Toes Cat - Hemingway Polydactyl",
            "Havana",
            "Himalayan",
            "Japanese Bobtail",
            "Javanese",
            "Korat",
            "LaPerm",
            "Maine Coon",
            "Manx",
            "Munchkin",
            "Nebelung",
            "Norwegian Forest Cat",
            "Ocicat",
            "Oriental Long Hair",
            "Oriental Short Hair",
            "Oriental Tabby",
            "Persian",
            "Pixiebob",
            "Ragamuffin",
            "Ragdoll",
            "Russian Blue",
            "Scottish Fold",
            "Selkirk Rex",
            "Siamese",
            "Siberian",
            "Silver",
            "Singapura",
            "Snowshoe",
            "Somali",
            "Sphynx - Hairless Cat",
            "Tabby",
            "Tiger",
            "Tonkinese",
            "Torbie",
            "Tortoiseshell",
            "Turkish Angora",
            "Turkish Van",
            "Tuxedo",
            "York Chocolate"
        ]

    def _validate_checkpoint_path(self, checkpoint_path: str) -> Path:
        """
        Validate checkpoint path to prevent path traversal attacks.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Validated Path object

        Raises:
            ValueError: If path is invalid or attempts traversal
            FileNotFoundError: If file doesn't exist
        """
        # Convert to absolute path
        checkpoint_file = Path(checkpoint_path).resolve()

        # Define allowed base directory
        allowed_base = Path("outputs/checkpoints").resolve()

        # Check if path is within allowed directory
        try:
            checkpoint_file.relative_to(allowed_base)
        except ValueError:
            raise ValueError(
                f"Invalid checkpoint path. Must be within outputs/checkpoints/ directory. "
                f"Got: {checkpoint_path}"
            )

        # Check file exists
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        # Check it's a file (not directory)
        if not checkpoint_file.is_file():
            raise ValueError(
                f"Checkpoint path must be a file: {checkpoint_path}"
            )

        return checkpoint_file

    async def load_model(
        self,
        checkpoint_path: str,
        model_name: str,
        num_classes: int = 67,
        device: str = "auto"
    ) -> None:
        """
        Load model from checkpoint using TransferLearningModel pattern.

        Args:
            checkpoint_path: Path to checkpoint file
            model_name: TIMM model name (e.g., resnet50, efficientnet_b0)
            num_classes: Number of output classes
            device: Device string (auto, cuda, mps, cpu)

        Raises:
            ValueError: If checkpoint path is invalid
            FileNotFoundError: If checkpoint path doesn't exist
            RuntimeError: If model loading fails
        """
        # Validate checkpoint path (prevents path traversal)
        checkpoint_file = self._validate_checkpoint_path(checkpoint_path)

        # Set device
        self._device = self._get_device(device)
        logger.info(f"Using device: {self._device}")

        # Create model architecture (same as src/models.py TransferLearningModel)
        logger.info(f"Creating model: {model_name}")
        backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_rate=0.2
        )
        num_features = backbone.num_features

        # Build model matching TransferLearningModel structure with backbone + classifier
        class APIModel(nn.Module):
            def __init__(self, backbone, num_features, num_classes):
                super().__init__()
                self.backbone = backbone
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(num_features, num_classes)
                )

            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)

        self._model = APIModel(backbone, num_features, num_classes)

        # Load checkpoint
        logger.info(f"Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(str(checkpoint_file), map_location=self._device)

        # Handle checkpoint structure
        if 'model_state_dict' in checkpoint:
            self._model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self._model.load_state_dict(checkpoint)

        # Move to device and set to eval mode
        self._model.to(self._device)
        self._model.eval()

        # Load class names
        self._class_names = self._load_class_names()

        # Update state
        self._is_loaded = True
        self._model_name = model_name
        self._checkpoint_path = str(checkpoint_file)

        logger.info(f"Model loaded successfully: {len(self._class_names)} classes")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def device(self) -> Optional[torch.device]:
        """Get current device."""
        return self._device

    @property
    def class_names(self) -> List[str]:
        """Get list of class names."""
        return self._class_names

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def checkpoint_path(self) -> str:
        """Get checkpoint path."""
        return self._checkpoint_path

    def predict(self, tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on input tensor.

        Args:
            tensor: Input tensor [B, C, H, W] or [C, H, W]

        Returns:
            Tuple of (probabilities, logits) as numpy arrays

        Raises:
            RuntimeError: If model not loaded
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Add batch dimension if needed
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            tensor = tensor.to(self._device)
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy(), logits.cpu().numpy()
