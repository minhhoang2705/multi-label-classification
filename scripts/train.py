"""
Main training script with CLI interface.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set Hugging Face cache directory to avoid permission issues
os.environ['HF_HOME'] = str(Path(__file__).parent.parent / '.cache' / 'huggingface')
os.environ['TORCH_HOME'] = str(Path(__file__).parent.parent / '.cache' / 'torch')

import argparse
from dataclasses import asdict
from src.config import Config, get_default_config, get_fast_dev_config
from src.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Cat Breeds Classification Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/images',
                        help='Path to images directory')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                        help='Model architecture name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone initially')
    parser.add_argument('--freeze_epochs', type=int, default=0,
                        help='Number of epochs to keep backbone frozen')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds for cross-validation (1 for single split)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train (None for all folds)')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['focal', 'weighted_ce', 'ce'],
                        help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor')
    
    # Augmentation arguments
    parser.add_argument('--use_autoaugment', action='store_true',
                        help='Use AutoAugment')
    parser.add_argument('--use_randaugment', action='store_true',
                        help='Use RandAugment')
    parser.add_argument('--use_mixup', action='store_true',
                        help='Use Mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Mixup alpha parameter')
    parser.add_argument('--use_cutmix', action='store_true',
                        help='Use CutMix augmentation')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                        help='CutMix alpha parameter')
    
    # Training settings
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='Use early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--early_stopping_metric', type=str, default='macro_f1',
                        choices=['macro_f1', 'val_loss', 'balanced_acc'],
                        help='Metric for early stopping')
    
    # Logging arguments
    parser.add_argument('--use_mlflow', action='store_true', default=True,
                        help='Use MLflow for experiment tracking')
    parser.add_argument('--experiment_name', type=str, default='cat_breeds_classification',
                        help='MLflow experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints',
                        help='Directory to save checkpoints')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--fast_dev', action='store_true',
                        help='Use fast development config (2 epochs, 2 folds)')
    
    return parser.parse_args()


def update_config_from_args(config: Config, args: argparse.Namespace) -> Config:
    """
    Update configuration from command line arguments.
    
    Args:
        config: Base configuration
        args: Parsed arguments
        
    Returns:
        Updated configuration
    """
    # Data config
    config.data.data_dir = args.data_dir
    config.data.image_size = args.image_size
    config.data.num_workers = args.num_workers
    
    # Model config
    config.model.name = args.model_name
    config.model.pretrained = args.pretrained
    config.model.freeze_backbone = args.freeze_backbone
    config.model.freeze_epochs = args.freeze_epochs
    config.model.dropout = args.dropout
    
    # Training config
    config.data.batch_size = args.batch_size
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.num_epochs
    config.training.num_folds = args.num_folds
    config.training.fold_to_train = args.fold
    config.training.use_amp = args.use_amp
    config.training.early_stopping = args.early_stopping
    config.training.early_stopping_patience = args.early_stopping_patience
    config.training.early_stopping_metric = args.early_stopping_metric
    config.training.device = args.device
    config.training.seed = args.seed
    
    # Optimizer config
    config.optimizer.optimizer_type = args.optimizer
    config.optimizer.learning_rate = args.lr
    config.optimizer.weight_decay = args.weight_decay
    config.optimizer.scheduler_type = args.scheduler
    config.optimizer.scheduler_warmup_epochs = args.warmup_epochs
    
    # Loss config
    config.loss.loss_type = args.loss_type
    config.loss.focal_gamma = args.focal_gamma
    config.loss.focal_alpha = args.focal_alpha
    config.loss.use_class_weights = args.use_class_weights
    config.loss.label_smoothing = args.label_smoothing
    
    # Augmentation config
    config.augmentation.use_autoaugment = args.use_autoaugment
    config.augmentation.use_randaugment = args.use_randaugment
    config.augmentation.use_mixup = args.use_mixup
    config.augmentation.mixup_alpha = args.mixup_alpha
    config.augmentation.use_cutmix = args.use_cutmix
    config.augmentation.cutmix_alpha = args.cutmix_alpha
    
    # Logging config
    config.logging.use_mlflow = args.use_mlflow
    config.logging.mlflow_experiment_name = args.experiment_name
    config.logging.checkpoint_dir = args.checkpoint_dir
    
    return config


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Get base config
    if args.fast_dev:
        config = get_fast_dev_config()
        print("\n⚡ Using fast development config (2 epochs, 2 folds)")
    else:
        config = get_default_config()
    
    # Update config from args
    config = update_config_from_args(config, args)
    
    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"\nModel: {config.model.name}")
    print(f"Pretrained: {config.model.pretrained}")
    print(f"Image size: {config.data.image_size}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Folds: {config.training.num_folds}")
    print(f"Optimizer: {config.optimizer.optimizer_type.upper()}")
    print(f"Learning rate: {config.optimizer.learning_rate}")
    print(f"Scheduler: {config.optimizer.scheduler_type}")
    print(f"Loss: {config.loss.loss_type}")
    if config.loss.loss_type == 'focal':
        print(f"  Focal gamma: {config.loss.focal_gamma}")
        print(f"  Focal alpha: {config.loss.focal_alpha}")
    print(f"Use class weights: {config.loss.use_class_weights}")
    print(f"Use AMP: {config.training.use_amp}")
    print(f"Early stopping: {config.training.early_stopping}")
    if config.training.early_stopping:
        print(f"  Patience: {config.training.early_stopping_patience}")
        print(f"  Metric: {config.training.early_stopping_metric}")
    print(f"Device: {config.training.device}")
    print(f"Seed: {config.training.seed}")
    print(f"MLflow: {config.logging.use_mlflow}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()

