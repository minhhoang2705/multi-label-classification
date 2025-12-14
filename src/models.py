"""
Model factory for transfer learning with various architectures.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional

from .config import ModelConfig


class TransferLearningModel(nn.Module):
    """Transfer learning model wrapper."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize transfer learning model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Create backbone using timm
        self.backbone = timm.create_model(
            config.name,
            pretrained=config.pretrained,
            num_classes=0,  # Remove classifier head
            drop_rate=config.dropout,
            drop_path_rate=config.drop_path_rate
        )
        
        # Get number of features from backbone
        self.num_features = self.backbone.num_features
        
        # Create custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(self.num_features, config.num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
        
        # Freeze backbone if specified
        if config.freeze_backbone:
            self.freeze_backbone()
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Backbone frozen: {self.config.name}")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"Backbone unfrozen: {self.config.name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Logits tensor [B, num_classes]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(config: ModelConfig) -> nn.Module:
    """
    Create model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        PyTorch model
    """
    model = TransferLearningModel(config)
    
    print(f"\nModel: {config.name}")
    print(f"Pretrained: {config.pretrained}")
    print(f"Num classes: {config.num_classes}")
    print(f"Total parameters: {model.get_num_total_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    return model


def get_available_models() -> list:
    """
    Get list of available models from timm.
    
    Returns:
        List of model names
    """
    # Popular models for image classification
    recommended_models = [
        # ResNet family
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d',
        'wide_resnet50_2', 'wide_resnet101_2',
        
        # EfficientNet family
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
        'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
        'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l',
        
        # Vision Transformer
        'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224',
        'vit_large_patch16_224',
        
        # ConvNeXt
        'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
        
        # MobileNet
        'mobilenetv3_small_100', 'mobilenetv3_large_100',
        
        # DenseNet
        'densenet121', 'densenet169', 'densenet201',
        
        # RegNet
        'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008',
        'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008',
        
        # Swin Transformer
        'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
        'swin_base_patch4_window7_224',
    ]
    
    return recommended_models


def print_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224)):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    try:
        from torchinfo import summary
        summary(model, input_size=input_size)
    except ImportError:
        print("Install torchinfo for detailed model summary: pip install torchinfo")
        print(f"\nModel architecture:\n{model}")


class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        """
        Initialize ensemble model.
        
        Args:
            models: List of PyTorch models
            weights: Optional weights for each model (must sum to 1.0)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
            self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted average of model predictions
        """
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            outputs.append(output * weight)
        
        return torch.stack(outputs).sum(dim=0)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cuda',
    strict: bool = True
) -> dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint
        strict: Whether to strictly enforce key matching
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    return checkpoint


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        checkpoint_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")

