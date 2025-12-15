"""
Loss functions for imbalanced classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import LossConfig


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Focusing parameter for modulating loss (gamma >= 0)
            reduction: 'none' | 'mean' | 'sum'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs: Predicted logits [B, C]
            targets: Ground truth labels [B]

        Returns:
            Loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # Compute probabilities
        p = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = (1 - p) ** self.gamma * ce_loss

        # Apply alpha weighting (for multi-class, alpha is typically not used
        # or should be a per-class weight tensor, not a single scalar)
        # For now, we disable alpha weighting for multi-class classification
        # to avoid incorrect gradient direction
        # if self.alpha is not None:
        #     alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        #     focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with class weights."""
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Initialize weighted cross-entropy loss.
        
        Args:
            weight: Class weights tensor [C]
            label_smoothing: Label smoothing factor
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits [B, C]
            targets: Ground truth labels [B]
            
        Returns:
            Loss value
        """
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing cross-entropy.
        
        Args:
            smoothing: Label smoothing factor
        """
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits [B, C]
            targets: Ground truth labels [B]
            
        Returns:
            Loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # One-hot encode targets
        num_classes = inputs.size(-1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        
        # Apply label smoothing
        targets_smooth = targets_one_hot * self.confidence + \
                        (1 - targets_one_hot) * self.smoothing / (num_classes - 1)
        
        loss = (-targets_smooth * log_probs).sum(dim=-1)
        
        return loss.mean()


class MixupLoss(nn.Module):
    """Loss function for mixup/cutmix training."""
    
    def __init__(self, base_loss: nn.Module):
        """
        Initialize mixup loss.
        
        Args:
            base_loss: Base loss function (e.g., CrossEntropyLoss, FocalLoss)
        """
        super().__init__()
        self.base_loss = base_loss
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """
        Forward pass for mixup.
        
        Args:
            inputs: Predicted logits [B, C]
            targets_a: First set of targets [B]
            targets_b: Second set of targets [B]
            lam: Mixup lambda value
            
        Returns:
            Mixed loss value
        """
        return lam * self.base_loss(inputs, targets_a) + \
               (1 - lam) * self.base_loss(inputs, targets_b)


def create_loss_function(
    config: LossConfig,
    class_weights: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> nn.Module:
    """
    Create loss function from configuration.
    
    Args:
        config: Loss configuration
        class_weights: Optional class weights
        device: Device to move weights to
        
    Returns:
        Loss function
    """
    # Move class weights to device if provided
    if class_weights is not None and config.use_class_weights:
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    
    if config.loss_type == 'focal':
        loss_fn = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing
        )
        print(f"Using Focal Loss (gamma={config.focal_gamma}, alpha={config.focal_alpha})")
    
    elif config.loss_type == 'weighted_ce':
        loss_fn = WeightedCrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.label_smoothing
        )
        print(f"Using Weighted Cross-Entropy Loss")
        if class_weights is not None:
            print(f"  Weight range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")
    
    elif config.loss_type == 'ce':
        loss_fn = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        print(f"Using Cross-Entropy Loss")
    
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")
    
    if config.label_smoothing > 0:
        print(f"  Label smoothing: {config.label_smoothing}")
    
    return loss_fn


class BalancedBatchSampler:
    """
    Sampler that creates balanced batches for imbalanced datasets.
    Each batch contains approximately equal samples from each class.
    """
    
    def __init__(
        self,
        labels: torch.Tensor,
        batch_size: int,
        num_classes: int
    ):
        """
        Initialize balanced batch sampler.
        
        Args:
            labels: Dataset labels
            batch_size: Batch size
            num_classes: Number of classes
        """
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Group indices by class
        self.class_indices = {}
        for class_idx in range(num_classes):
            self.class_indices[class_idx] = torch.where(labels == class_idx)[0]
    
    def __iter__(self):
        """Generate balanced batches."""
        samples_per_class = self.batch_size // self.num_classes
        
        # Shuffle indices for each class
        shuffled_indices = {}
        for class_idx in range(self.num_classes):
            indices = self.class_indices[class_idx]
            shuffled_indices[class_idx] = indices[torch.randperm(len(indices))]
        
        # Generate batches
        batch = []
        class_pointers = {i: 0 for i in range(self.num_classes)}
        
        while True:
            for class_idx in range(self.num_classes):
                indices = shuffled_indices[class_idx]
                pointer = class_pointers[class_idx]
                
                # Check if we need to reshuffle
                if pointer + samples_per_class > len(indices):
                    shuffled_indices[class_idx] = indices[torch.randperm(len(indices))]
                    pointer = 0
                    class_pointers[class_idx] = 0
                
                # Add samples to batch
                batch.extend(
                    shuffled_indices[class_idx][pointer:pointer + samples_per_class].tolist()
                )
                class_pointers[class_idx] += samples_per_class
                
                if len(batch) >= self.batch_size:
                    yield batch[:self.batch_size]
                    batch = batch[self.batch_size:]
    
    def __len__(self):
        """Return number of batches."""
        total_samples = sum(len(indices) for indices in self.class_indices.values())
        return total_samples // self.batch_size

