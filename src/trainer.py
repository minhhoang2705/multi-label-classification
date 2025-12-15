"""
Training pipeline with k-fold cross-validation and MLflow tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
from pathlib import Path
from typing import Optional, Dict
import mlflow
import mlflow.pytorch

from .config import Config
from .models import create_model, save_checkpoint, load_checkpoint
from .losses import create_loss_function, MixupLoss
from .metrics import MetricsCalculator, AverageMeter, MetricTracker, compute_metrics_from_logits
from .utils import (
    set_seed,
    get_device,
    EarlyStopping,
    GradualWarmupScheduler,
    create_optimizer,
    create_scheduler,
    get_lr,
    format_time,
    mixup_data,
    cutmix_data
)
from .dataset import get_dataloaders, prepare_data_for_training


class Trainer:
    """Main trainer class."""
    
    def __init__(self, config: Config):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Set seed
        set_seed(config.training.seed, config.training.deterministic)
        
        # Device
        self.device = get_device(config.training.device)
        
        # Prepare data
        print("\n" + "="*80)
        print("PREPARING DATA")
        print("="*80)
        
        if config.training.num_folds > 1:
            self.folds, self.label_encoder, self.class_weights, self.num_classes = \
                prepare_data_for_training(config.data, config.augmentation, use_kfold=True)
            self.use_kfold = True
        else:
            self.dataloaders, self.label_encoder, self.class_weights, self.num_classes = \
                prepare_data_for_training(config.data, config.augmentation, use_kfold=False)
            self.use_kfold = False
        
        # Update num_classes in config
        self.config.model.num_classes = self.num_classes
        
        # MLflow setup
        if config.logging.use_mlflow:
            mlflow.set_tracking_uri(config.logging.mlflow_tracking_uri)
            mlflow.set_experiment(config.logging.mlflow_experiment_name)
    
    def train_fold(self, fold_idx: int, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train a single fold.

        Args:
            fold_idx: Fold index
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Dictionary of best metrics
        """
        print(f"\n{'='*80}")
        print(f"TRAINING FOLD {fold_idx + 1}/{self.config.training.num_folds}")
        print(f"{'='*80}\n")

        # Clear GPU cache before creating model to avoid OOM errors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ Cleared GPU cache")

        # Create model
        model = create_model(self.config.model).to(self.device)
        
        # Create loss function
        loss_fn = create_loss_function(
            self.config.loss,
            self.class_weights,
            self.device
        )
        
        # Create optimizer
        optimizer = create_optimizer(
            model,
            self.config.optimizer.optimizer_type,
            self.config.optimizer.learning_rate,
            self.config.optimizer.weight_decay,
            self.config.optimizer.momentum
        )
        
        # Create scheduler
        scheduler = create_scheduler(
            optimizer,
            self.config.optimizer.scheduler_type,
            self.config.training.num_epochs,
            min_lr=self.config.optimizer.scheduler_min_lr,
            step_size=self.config.optimizer.step_size,
            gamma=self.config.optimizer.step_gamma,
            patience=self.config.optimizer.plateau_patience,
            factor=self.config.optimizer.plateau_factor
        )
        
        # Warmup scheduler
        if self.config.optimizer.scheduler_warmup_epochs > 0:
            scheduler = GradualWarmupScheduler(
                optimizer,
                self.config.optimizer.scheduler_warmup_epochs,
                self.config.optimizer.learning_rate,
                scheduler
            )
        
        # AMP scaler
        scaler = GradScaler() if self.config.training.use_amp else None

        # Resume from checkpoint if specified
        start_epoch = 0
        if self.config.training.resume_checkpoint is not None:
            print(f"\nResuming from checkpoint: {self.config.training.resume_checkpoint}")
            checkpoint = load_checkpoint(
                model,
                self.config.training.resume_checkpoint,
                self.device,
                strict=True
            )

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Loaded optimizer state")

            # Load scheduler state
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Loaded scheduler state")

            # Load AMP scaler state
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("✓ Loaded AMP scaler state")

            # Resume from next epoch
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"✓ Resuming from epoch {start_epoch + 1}")

            # Load best metrics if available
            if 'metrics' in checkpoint:
                print(f"✓ Previous metrics: {checkpoint['metrics']}")

        # Early stopping
        early_stopping = None
        if self.config.training.early_stopping:
            early_stopping = EarlyStopping(
                patience=self.config.training.early_stopping_patience,
                mode=self.config.training.early_stopping_mode,
                verbose=True
            )

        # Metric tracker
        metric_tracker = MetricTracker()

        # Best metrics
        best_metrics = {
            'epoch': 0,
            self.config.training.early_stopping_metric: 0.0
        }

        # Training loop
        for epoch in range(start_epoch, self.config.training.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            print("-" * 80)
            
            # Unfreeze backbone if needed
            if epoch == self.config.model.freeze_epochs and self.config.model.freeze_backbone:
                model.unfreeze_backbone()
            
            # Train
            train_metrics = self.train_epoch(
                model, train_loader, loss_fn, optimizer, scaler, epoch
            )
            
            # Validate
            val_metrics = self.validate_epoch(
                model, val_loader, loss_fn, epoch
            )
            
            # Update scheduler
            if scheduler is not None:
                if self.config.optimizer.scheduler_type == 'plateau':
                    scheduler.step(val_metrics[self.config.training.early_stopping_metric])
                else:
                    scheduler.step()
            
            # Track metrics
            all_metrics = {**train_metrics, **val_metrics}
            metric_tracker.update(all_metrics, epoch)
            
            # Print metrics
            print(f"\nTrain Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Train Acc: {train_metrics['train_accuracy']:.4f} | "
                  f"Val Acc: {val_metrics['val_accuracy']:.4f}")
            print(f"Val Macro F1: {val_metrics['val_macro_f1']:.4f} | "
                  f"Val Balanced Acc: {val_metrics['val_balanced_accuracy']:.4f}")
            print(f"LR: {get_lr(optimizer):.6f}")
            
            # Check if best model
            metric_value = val_metrics[f"val_{self.config.training.early_stopping_metric}"]
            is_best = False
            
            if self.config.training.early_stopping_mode == 'max':
                if metric_value > best_metrics[self.config.training.early_stopping_metric]:
                    is_best = True
                    best_metrics = {
                        'epoch': epoch,
                        self.config.training.early_stopping_metric: metric_value,
                        **val_metrics
                    }
            else:
                if metric_value < best_metrics[self.config.training.early_stopping_metric]:
                    is_best = True
                    best_metrics = {
                        'epoch': epoch,
                        self.config.training.early_stopping_metric: metric_value,
                        **val_metrics
                    }
            
            # Save checkpoint
            checkpoint_dir = Path(self.config.logging.checkpoint_dir) / f"fold_{fold_idx}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            if is_best or not self.config.logging.save_best_only:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                save_checkpoint(model, optimizer, epoch, all_metrics, str(checkpoint_path), scheduler, scaler)

            if is_best:
                best_checkpoint_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(model, optimizer, epoch, all_metrics, str(best_checkpoint_path), scheduler, scaler)
                print(f"✓ Saved best model (epoch {epoch + 1})")
            
            # Early stopping
            if early_stopping is not None:
                if early_stopping(metric_value):
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        return best_metrics
    
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[GradScaler],
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            model: PyTorch model
            dataloader: Training data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scaler: AMP scaler
            epoch: Current epoch
            
        Returns:
            Dictionary of metrics
        """
        model.train()
        
        loss_meter = AverageMeter('loss')
        acc_meter = AverageMeter('accuracy')
        
        pbar = tqdm(dataloader, desc='Training')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            batch_size = images.size(0)
            
            # Mixup / CutMix
            use_mixup = self.config.augmentation.use_mixup and torch.rand(1).item() < 0.5
            use_cutmix = self.config.augmentation.use_cutmix and torch.rand(1).item() < 0.5
            
            if use_mixup:
                images, targets_a, targets_b, lam = mixup_data(
                    images, targets, self.config.augmentation.mixup_alpha
                )
            elif use_cutmix:
                images, targets_a, targets_b, lam = cutmix_data(
                    images, targets, self.config.augmentation.cutmix_alpha
                )
            
            # Forward pass
            if self.config.training.use_amp:
                with autocast():
                    outputs = model(images)
                    
                    if use_mixup or use_cutmix:
                        loss = lam * loss_fn(outputs, targets_a) + \
                               (1 - lam) * loss_fn(outputs, targets_b)
                    else:
                        loss = loss_fn(outputs, targets)
            else:
                outputs = model(images)
                
                if use_mixup or use_cutmix:
                    loss = lam * loss_fn(outputs, targets_a) + \
                           (1 - lam) * loss_fn(outputs, targets_b)
                else:
                    loss = loss_fn(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            
            if self.config.training.use_amp:
                scaler.scale(loss).backward()
                
                if self.config.optimizer.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.optimizer.grad_clip
                    )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                if self.config.optimizer.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.optimizer.grad_clip
                    )
                
                optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                _, preds = outputs.max(dim=1)
                if use_mixup or use_cutmix:
                    # Use original targets for accuracy
                    correct = (preds == targets).float().sum()
                else:
                    correct = (preds == targets).float().sum()
                accuracy = correct / batch_size
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
        
        return {
            'train_loss': loss_meter.avg,
            'train_accuracy': acc_meter.avg
        }
    
    def validate_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            model: PyTorch model
            dataloader: Validation data loader
            loss_fn: Loss function
            epoch: Current epoch
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        
        loss_meter = AverageMeter('loss')
        metrics_calc = MetricsCalculator(
            self.num_classes,
            self.label_encoder.classes_.tolist()
        )
        
        pbar = tqdm(dataloader, desc='Validation')
        
        with torch.no_grad():
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                batch_size = images.size(0)
                
                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(dim=1)
                
                # Update meters
                loss_meter.update(loss.item(), batch_size)
                metrics_calc.update(preds, targets, probs)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        # Compute metrics
        metrics = metrics_calc.compute()
        
        # Format metrics with 'val_' prefix
        val_metrics = {
            'val_loss': loss_meter.avg,
            'val_accuracy': metrics['accuracy'],
            'val_balanced_accuracy': metrics['balanced_accuracy'],
            'val_precision_macro': metrics['precision_macro'],
            'val_recall_macro': metrics['recall_macro'],
            'val_macro_f1': metrics['f1_macro'],
            'val_precision_weighted': metrics['precision_weighted'],
            'val_recall_weighted': metrics['recall_weighted'],
            'val_f1_weighted': metrics['f1_weighted']
        }
        
        # Add top-k accuracies if available
        for k in [3, 5]:
            key = f'top_{k}_accuracy'
            if key in metrics:
                val_metrics[f'val_{key}'] = metrics[key]
        
        return val_metrics
    
    def train(self):
        """Main training function."""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        start_time = time.time()
        
        if self.use_kfold:
            # K-fold cross-validation
            fold_results = []
            
            for fold_idx in range(self.config.training.num_folds):
                # Skip if specific fold is specified
                if self.config.training.fold_to_train is not None:
                    if fold_idx != self.config.training.fold_to_train:
                        continue
                
                # Start MLflow run
                if self.config.logging.use_mlflow:
                    with mlflow.start_run(run_name=f"fold_{fold_idx}"):
                        # Log config
                        mlflow.log_params(self.config.to_dict())
                        
                        # Get fold data
                        fold_data = self.folds[fold_idx]
                        
                        # Create dataloaders
                        dataloaders = get_dataloaders(
                            self.config.data,
                            self.config.augmentation,
                            fold_data,
                            self.label_encoder
                        )
                        
                        # Train fold
                        best_metrics = self.train_fold(
                            fold_idx,
                            dataloaders['train'],
                            dataloaders['val']
                        )
                        
                        # Log best metrics
                        mlflow.log_metrics(best_metrics)
                        
                        fold_results.append(best_metrics)
                else:
                    # Get fold data
                    fold_data = self.folds[fold_idx]
                    
                    # Create dataloaders
                    dataloaders = get_dataloaders(
                        self.config.data,
                        self.config.augmentation,
                        fold_data,
                        self.label_encoder
                    )
                    
                    # Train fold
                    best_metrics = self.train_fold(
                        fold_idx,
                        dataloaders['train'],
                        dataloaders['val']
                    )
                    
                    fold_results.append(best_metrics)
            
            # Print summary
            print("\n" + "="*80)
            print("K-FOLD CROSS-VALIDATION SUMMARY")
            print("="*80)
            
            metric_name = self.config.training.early_stopping_metric
            fold_scores = [result[metric_name] for result in fold_results]
            
            print(f"\n{metric_name}:")
            for i, score in enumerate(fold_scores):
                print(f"  Fold {i + 1}: {score:.4f}")
            print(f"  Mean: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        
        else:
            # Single train/val/test split
            if self.config.logging.use_mlflow:
                with mlflow.start_run():
                    # Log config
                    mlflow.log_params(self.config.to_dict())
                    
                    # Train
                    best_metrics = self.train_fold(
                        0,
                        self.dataloaders['train'],
                        self.dataloaders['val']
                    )
                    
                    # Log best metrics
                    mlflow.log_metrics(best_metrics)
            else:
                # Train
                best_metrics = self.train_fold(
                    0,
                    self.dataloaders['train'],
                    self.dataloaders['val']
                )
        
        # Print total time
        total_time = time.time() - start_time
        print(f"\nTotal training time: {format_time(total_time)}")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)


import numpy as np

