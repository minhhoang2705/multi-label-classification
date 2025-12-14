"""
Evaluation metrics for imbalanced classification.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    top_k_accuracy_score
)
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Calculate various metrics for classification."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Storage for predictions and targets
        self.reset()
    
    def reset(self):
        """Reset stored predictions and targets."""
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Update with new predictions and targets.
        
        Args:
            predictions: Predicted class indices [B]
            targets: Ground truth labels [B]
            probabilities: Optional class probabilities [B, C]
        """
        self.all_preds.extend(predictions.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        
        if probabilities is not None:
            self.all_probs.extend(probabilities.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metrics
        """
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, preds)
        
        # Balanced accuracy (better for imbalanced datasets)
        metrics['balanced_accuracy'] = balanced_accuracy_score(targets, preds)
        
        # Precision, Recall, F1 (macro and weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, preds, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            targets, preds, average='weighted', zero_division=0
        )
        
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        metrics['precision_weighted'] = precision_weighted
        metrics['recall_weighted'] = recall_weighted
        metrics['f1_weighted'] = f1_weighted
        
        # Top-k accuracy
        if len(self.all_probs) > 0:
            probs = np.array(self.all_probs)
            for k in [1, 3, 5]:
                if k <= self.num_classes:
                    metrics[f'top_{k}_accuracy'] = top_k_accuracy_score(
                        targets, probs, k=k, labels=np.arange(self.num_classes)
                    )
        
        # Per-class metrics
        per_class_metrics = self.compute_per_class_metrics()
        metrics['per_class'] = per_class_metrics
        
        return metrics
    
    def compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics.
        
        Returns:
            Dictionary mapping class names to their metrics
        """
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )
        
        per_class = {}
        for i, class_name in enumerate(self.class_names):
            per_class[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        return per_class
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        return confusion_matrix(targets, preds, labels=np.arange(self.num_classes))
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = False,
        figsize: tuple = (12, 10)
    ):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Optional path to save figure
            normalize: Whether to normalize confusion matrix
            figsize: Figure size
        """
        cm = self.get_confusion_matrix()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        
        # For many classes, don't show all labels
        if self.num_classes > 20:
            sns.heatmap(cm, cmap='Blues', fmt=fmt, cbar=True)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        else:
            sns.heatmap(
                cm,
                annot=True,
                fmt=fmt,
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                cbar=True
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def print_classification_report(self):
        """Print detailed classification report."""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(
            targets,
            preds,
            target_names=self.class_names,
            zero_division=0
        ))
    
    def get_worst_performing_classes(self, n: int = 10, metric: str = 'f1') -> List[tuple]:
        """
        Get worst performing classes.
        
        Args:
            n: Number of classes to return
            metric: Metric to use ('f1', 'precision', 'recall')
            
        Returns:
            List of (class_name, metric_value) tuples
        """
        per_class = self.compute_per_class_metrics()
        
        class_scores = [(name, metrics[metric]) for name, metrics in per_class.items()]
        class_scores.sort(key=lambda x: x[1])
        
        return class_scores[:n]
    
    def get_best_performing_classes(self, n: int = 10, metric: str = 'f1') -> List[tuple]:
        """
        Get best performing classes.
        
        Args:
            n: Number of classes to return
            metric: Metric to use ('f1', 'precision', 'recall')
            
        Returns:
            List of (class_name, metric_value) tuples
        """
        per_class = self.compute_per_class_metrics()
        
        class_scores = [(name, metrics[metric]) for name, metrics in per_class.items()]
        class_scores.sort(key=lambda x: x[1], reverse=True)
        
        return class_scores[:n]


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ''):
        """
        Initialize average meter.
        
        Args:
            name: Name of the meter
        """
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Args:
            val: Value to add
            n: Number of items
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class MetricTracker:
    """Track metrics across epochs."""
    
    def __init__(self):
        """Initialize metric tracker."""
        self.history = {}
    
    def update(self, metrics: Dict[str, float], epoch: int):
        """
        Update metrics for an epoch.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Epoch number
        """
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append((epoch, value))
    
    def get_best(self, metric_name: str, mode: str = 'max') -> tuple:
        """
        Get best value for a metric.
        
        Args:
            metric_name: Name of metric
            mode: 'max' or 'min'
            
        Returns:
            Tuple of (epoch, value)
        """
        if metric_name not in self.history:
            return None, None
        
        history = self.history[metric_name]
        if mode == 'max':
            best_epoch, best_value = max(history, key=lambda x: x[1])
        else:
            best_epoch, best_value = min(history, key=lambda x: x[1])
        
        return best_epoch, best_value
    
    def plot_metrics(
        self,
        metrics: List[str],
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ):
        """
        Plot metric history.
        
        Args:
            metrics: List of metric names to plot
            save_path: Optional path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric_name in zip(axes, metrics):
            if metric_name in self.history:
                epochs, values = zip(*self.history[metric_name])
                ax.plot(epochs, values, marker='o', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} over epochs')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to: {save_path}")
        
        plt.close()


def compute_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple = (1, 3, 5)
) -> Dict[str, float]:
    """
    Compute metrics from logits.
    
    Args:
        logits: Model output logits [B, C]
        targets: Ground truth labels [B]
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        num_classes = logits.size(1)
        
        # Predictions
        _, preds = logits.max(dim=1)
        
        # Accuracy
        correct = (preds == targets).float().sum()
        accuracy = correct / batch_size
        
        metrics = {'accuracy': accuracy.item()}
        
        # Top-k accuracy
        for k in topk:
            if k <= num_classes:
                _, topk_preds = logits.topk(k, dim=1, largest=True, sorted=True)
                correct_k = topk_preds.eq(targets.view(-1, 1).expand_as(topk_preds)).float().sum()
                metrics[f'top_{k}_accuracy'] = (correct_k / batch_size).item()
        
        return metrics

