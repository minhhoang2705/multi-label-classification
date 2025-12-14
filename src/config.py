"""
Configuration dataclasses for the training pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "data/images"
    csv_path: Optional[str] = "data/clean_metadata.csv"
    image_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True

    # Stratified split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # Basic augmentations
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_degrees: int = 15

    # Color augmentations
    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1

    # Advanced augmentations
    random_erasing: bool = True
    random_erasing_prob: float = 0.25

    # AutoAugment / RandAugment
    use_autoaugment: bool = False
    use_randaugment: bool = False
    randaugment_n: int = 2
    randaugment_m: int = 9

    # Mixup / CutMix
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0

    # Normalization (ImageNet stats)
    normalize_mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    """Model configuration."""
    # Model architecture
    # resnet50, efficientnet_b0-b7, vit_b_16, convnext_tiny
    name: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = 67

    # Transfer learning
    freeze_backbone: bool = False
    freeze_epochs: int = 0  # Number of epochs to keep backbone frozen

    # Dropout
    dropout: float = 0.2
    drop_path_rate: float = 0.0  # For models that support it


@dataclass
class LossConfig:
    """Loss function configuration."""
    loss_type: str = "focal"  # "focal", "weighted_ce", "ce"

    # Focal loss parameters
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = 0.25

    # Class weights
    use_class_weights: bool = True

    # Label smoothing
    label_smoothing: float = 0.0


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    optimizer_type: str = "adamw"  # "adam", "adamw", "sgd"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    momentum: float = 0.9  # For SGD

    # Learning rate scheduler
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau", "none"
    scheduler_warmup_epochs: int = 5
    scheduler_min_lr: float = 1e-6

    # Step scheduler
    step_size: int = 10
    step_gamma: float = 0.1

    # Plateau scheduler
    plateau_patience: int = 5
    plateau_factor: float = 0.5

    # Gradient clipping
    grad_clip: Optional[float] = 1.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training
    batch_size: int = 32
    num_epochs: int = 50
    accumulation_steps: int = 1  # Gradient accumulation

    # Cross-validation
    num_folds: int = 5
    fold_to_train: Optional[int] = None  # None means train all folds

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    # "macro_f1", "val_loss", "balanced_acc"
    early_stopping_metric: str = "macro_f1"
    early_stopping_mode: str = "max"  # "max" or "min"

    # Mixed precision training
    use_amp: bool = True

    # Device
    device: str = "cuda"  # "cuda", "cpu", "mps"

    # Reproducibility
    seed: int = 42
    deterministic: bool = False


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    # MLflow
    use_mlflow: bool = True
    mlflow_experiment_name: str = "cat_breeds_classification"
    mlflow_tracking_uri: str = "mlruns"

    # Checkpointing
    checkpoint_dir: str = "outputs/checkpoints"
    save_best_only: bool = False
    save_last: bool = True

    # Logging frequency
    log_every_n_steps: int = 50

    # Artifacts
    save_confusion_matrix: bool = True
    save_per_class_metrics: bool = True


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(
        default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create directories
        Path(self.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.mlflow_tracking_uri).mkdir(
            parents=True, exist_ok=True)

        # Validate ratios
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        assert abs(
            total_ratio - 1.0) < 1e-6, f"Train/val/test ratios must sum to 1.0, got {total_ratio}"

        # Validate loss type
        assert self.loss.loss_type in ["focal", "weighted_ce", "ce"], \
            f"Invalid loss type: {self.loss.loss_type}"

        # Validate early stopping
        if self.training.early_stopping:
            assert self.training.early_stopping_mode in ["max", "min"], \
                f"Invalid early stopping mode: {self.training.early_stopping_mode}"

    def to_dict(self):
        """Convert config to dictionary for logging."""
        return {
            "data": self.data.__dict__,
            "augmentation": self.augmentation.__dict__,
            "model": self.model.__dict__,
            "loss": self.loss.__dict__,
            "optimizer": self.optimizer.__dict__,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
        }


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_fast_dev_config() -> Config:
    """Get configuration for fast development/debugging."""
    config = Config()
    config.training.num_epochs = 2
    config.training.num_folds = 2
    config.data.num_workers = 2
    config.logging.log_every_n_steps = 10
    return config
