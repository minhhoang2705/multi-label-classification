"""
Utility functions for training pipeline.
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from typing import Optional
import json


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
    
    print(f"Random seed set to: {seed}")


def get_device(device_str: str = 'cuda') -> torch.device:
    """
    Get torch device.
    
    Args:
        device_str: Device string ('cuda', 'cpu', 'mps')
        
    Returns:
        Torch device
    """
    if device_str == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    elif device_str == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("MPS not available, using CPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> tuple:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def save_json(data: dict, filepath: str):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved JSON to: {filepath}")


def load_json(filepath: str) -> dict:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(
        self,
        patience: int = 10,
        mode: str = 'max',
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            mode: 'max' for maximizing metric, 'min' for minimizing
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda score, best: score > best + min_delta
        else:
            self.is_better = lambda score, best: score < best - min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"Validation metric improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
                return True
        
        return False


class GradualWarmupScheduler:
    """Gradually warm up learning rate."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_lr: float,
        after_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            base_lr: Base learning rate
            after_scheduler: Scheduler to use after warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.current_epoch + 1
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if self.after_scheduler is not None:
                self.after_scheduler.step()
    
    def get_last_lr(self):
        """Get last learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def state_dict(self):
        """Return the state of the scheduler as a dict."""
        state = {
            'warmup_epochs': self.warmup_epochs,
            'base_lr': self.base_lr,
            'current_epoch': self.current_epoch
        }
        if self.after_scheduler is not None:
            state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load the scheduler state."""
        self.warmup_epochs = state_dict['warmup_epochs']
        self.base_lr = state_dict['base_lr']
        self.current_epoch = state_dict['current_epoch']
        if self.after_scheduler is not None and 'after_scheduler' in state_dict:
            self.after_scheduler.load_state_dict(state_dict['after_scheduler'])


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Apply mixup augmentation.
    
    Args:
        x: Input images
        y: Labels
        alpha: Mixup alpha parameter
        
    Returns:
        Tuple of (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    Apply cutmix augmentation.
    
    Args:
        x: Input images
        y: Labels
        alpha: Cutmix alpha parameter
        
    Returns:
        Tuple of (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get random box
    _, _, h, w = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Apply cutmix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


class AverageMeterGroup:
    """Group of average meters."""
    
    def __init__(self):
        """Initialize meter group."""
        self.meters = {}
    
    def update(self, data: dict, n: int = 1):
        """
        Update meters.
        
        Args:
            data: Dictionary of values
            n: Number of items
        """
        for key, value in data.items():
            if key not in self.meters:
                from .metrics import AverageMeter
                self.meters[key] = AverageMeter(key)
            self.meters[key].update(value, n)
    
    def __getitem__(self, key: str):
        """Get meter by key."""
        return self.meters[key]
    
    def __str__(self) -> str:
        """String representation."""
        return " | ".join([str(meter) for meter in self.meters.values()])
    
    def summary(self) -> dict:
        """Get summary of all meters."""
        return {name: meter.avg for name, meter in self.meters.items()}


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9
) -> torch.optim.Optimizer:
    """
    Create optimizer.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
        momentum: Momentum (for SGD)
        
    Returns:
        PyTorch optimizer
    """
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    print(f"Optimizer: {optimizer_type.upper()}, LR: {learning_rate}, WD: {weight_decay}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_epochs: int,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        num_epochs: Total number of epochs
        **kwargs: Additional scheduler parameters
        
    Returns:
        PyTorch scheduler or None
    """
    if scheduler_type.lower() == 'none':
        return None
    
    elif scheduler_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=kwargs.get('min_lr', 1e-6)
        )
        print(f"Scheduler: CosineAnnealingLR")
    
    elif scheduler_type.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
        print(f"Scheduler: StepLR")
    
    elif scheduler_type.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=kwargs.get('patience', 5),
            factor=kwargs.get('factor', 0.5)
        )
        print(f"Scheduler: ReduceLROnPlateau")
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

