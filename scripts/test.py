"""
Comprehensive model testing and evaluation script.

Tests model performance on test/validation set with:
- Accuracy metrics (overall, balanced, top-k)
- Per-class performance analysis
- Confusion matrix visualization
- Inference speed benchmarking
- Best/worst performing classes
"""

import sys
import os
from pathlib import Path
import time
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set cache directories
os.environ['HF_HOME'] = str(Path(__file__).parent.parent / '.cache' / 'huggingface')
os.environ['TORCH_HOME'] = str(Path(__file__).parent.parent / '.cache' / 'torch')

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from src.config import Config, get_default_config
from src.models import create_model, load_checkpoint
from src.dataset import prepare_data_for_training, get_dataloaders
from src.metrics import MetricsCalculator
from src.utils import get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Cat Breeds Classification Model')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint to test')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='Model architecture name')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/images',
                        help='Path to images directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')

    # Test settings
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--fold', type=int, default=0,
                        help='Which fold to evaluate (if using k-fold)')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds (must match training)')

    # Output settings
    parser.add_argument('--output_dir', type=str, default='outputs/test_results',
                        help='Directory to save test results')
    parser.add_argument('--save_confusion_matrix', action='store_true', default=True,
                        help='Save confusion matrix plot')
    parser.add_argument('--save_per_class_metrics', action='store_true', default=True,
                        help='Save per-class metrics to JSON')
    parser.add_argument('--save_roc_curves', action='store_true', default=True,
                        help='Save ROC curves plot')
    parser.add_argument('--save_pr_curves', action='store_true', default=True,
                        help='Save Precision-Recall curves plot')
    parser.add_argument('--num_inference_samples', type=int, default=1000,
                        help='Number of samples for inference speed test')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to use for testing')

    return parser.parse_args()


def benchmark_inference_speed(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 1000
) -> dict:
    """
    Benchmark model inference speed.

    Args:
        model: PyTorch model
        dataloader: Test data loader
        device: Device to run inference on
        num_samples: Number of samples to benchmark

    Returns:
        Dictionary with speed metrics
    """
    model.eval()

    times = []
    total_samples = 0

    print(f"\n{'='*80}")
    print(f"BENCHMARKING INFERENCE SPEED ({num_samples} samples)")
    print(f"{'='*80}\n")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Benchmarking')
        for images, _ in pbar:
            if total_samples >= num_samples:
                break

            images = images.to(device)
            batch_size = images.size(0)

            # Warmup on first batch
            if total_samples == 0:
                _ = model(images)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            # Time inference
            start_time = time.perf_counter()
            _ = model(images)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Record time per sample
            batch_time = end_time - start_time
            times.append(batch_time / batch_size)

            total_samples += batch_size

            # Update progress
            pbar.set_postfix({
                'samples': total_samples,
                'avg_ms': f'{np.mean(times) * 1000:.2f}'
            })

    times = np.array(times)

    metrics = {
        'samples_tested': total_samples,
        'avg_time_per_sample_ms': float(np.mean(times) * 1000),
        'std_time_per_sample_ms': float(np.std(times) * 1000),
        'min_time_per_sample_ms': float(np.min(times) * 1000),
        'max_time_per_sample_ms': float(np.max(times) * 1000),
        'median_time_per_sample_ms': float(np.median(times) * 1000),
        'throughput_samples_per_sec': float(1.0 / np.mean(times)),
        'device': str(device)
    }

    print(f"\nInference Speed Results:")
    print(f"  Average: {metrics['avg_time_per_sample_ms']:.2f} ms/sample")
    print(f"  Std Dev: {metrics['std_time_per_sample_ms']:.2f} ms")
    print(f"  Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/sec")

    return metrics


def compute_roc_auc(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int
) -> dict:
    """
    Compute ROC curves and AUC for multi-class classification.

    Args:
        y_true: True labels [N]
        y_probs: Predicted probabilities [N, C]
        num_classes: Number of classes

    Returns:
        Dictionary with ROC data and AUC scores
    """
    # Binarize labels for one-vs-rest ROC
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }


def compute_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int
) -> dict:
    """
    Compute Precision-Recall curves and Average Precision.

    Args:
        y_true: True labels [N]
        y_probs: Predicted probabilities [N, C]
        num_classes: Number of classes

    Returns:
        Dictionary with PR data and AP scores
    """
    # Binarize labels for one-vs-rest PR
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    # Compute PR curve and AP for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_probs[:, i]
        )
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])

    # Compute micro-average PR curve and AP
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_probs.ravel()
    )
    avg_precision["micro"] = average_precision_score(
        y_true_bin, y_probs, average="micro"
    )

    # Compute macro-average AP
    avg_precision["macro"] = average_precision_score(
        y_true_bin, y_probs, average="macro"
    )

    return {
        'precision': precision,
        'recall': recall,
        'average_precision': avg_precision
    }


def plot_roc_curves(
    roc_data: dict,
    class_names: list,
    save_path: str,
    max_classes_to_plot: int = 10
):
    """
    Plot ROC curves for multi-class classification.

    Args:
        roc_data: ROC data from compute_roc_auc()
        class_names: List of class names
        save_path: Path to save plot
        max_classes_to_plot: Maximum individual class curves to plot
    """
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    roc_auc = roc_data['auc']
    num_classes = len(class_names)

    plt.figure(figsize=(12, 10))

    # Plot micro-average ROC curve
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f'micro-average (AUC = {roc_auc["micro"]:.3f})',
        color='deeppink', linestyle=':', linewidth=4
    )

    # Plot macro-average ROC curve
    plt.plot(
        fpr["macro"], tpr["macro"],
        label=f'macro-average (AUC = {roc_auc["macro"]:.3f})',
        color='navy', linestyle=':', linewidth=4
    )

    # Plot individual class ROC curves (top classes by AUC)
    if num_classes <= max_classes_to_plot:
        classes_to_plot = range(num_classes)
    else:
        # Select top classes by AUC
        class_aucs = [(i, roc_auc[i]) for i in range(num_classes)]
        class_aucs.sort(key=lambda x: x[1], reverse=True)
        classes_to_plot = [i for i, _ in class_aucs[:max_classes_to_plot]]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes_to_plot)))

    for i, color in zip(classes_to_plot, colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2, alpha=0.7,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved ROC curves to: {save_path}")


def plot_pr_curves(
    pr_data: dict,
    class_names: list,
    save_path: str,
    max_classes_to_plot: int = 10
):
    """
    Plot Precision-Recall curves for multi-class classification.

    Args:
        pr_data: PR data from compute_pr_curves()
        class_names: List of class names
        save_path: Path to save plot
        max_classes_to_plot: Maximum individual class curves to plot
    """
    precision = pr_data['precision']
    recall = pr_data['recall']
    avg_precision = pr_data['average_precision']
    num_classes = len(class_names)

    plt.figure(figsize=(12, 10))

    # Plot micro-average PR curve
    plt.plot(
        recall["micro"], precision["micro"],
        label=f'micro-average (AP = {avg_precision["micro"]:.3f})',
        color='deeppink', linestyle=':', linewidth=4
    )

    # Plot macro-average (just show AP score, no curve)
    plt.plot(
        [], [],  # Empty plot for legend
        label=f'macro-average (AP = {avg_precision["macro"]:.3f})',
        color='navy', linestyle=':', linewidth=4
    )

    # Plot individual class PR curves (top classes by AP)
    if num_classes <= max_classes_to_plot:
        classes_to_plot = range(num_classes)
    else:
        # Select top classes by AP
        class_aps = [(i, avg_precision[i]) for i in range(num_classes)]
        class_aps.sort(key=lambda x: x[1], reverse=True)
        classes_to_plot = [i for i, _ in class_aps[:max_classes_to_plot]]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes_to_plot)))

    for i, color in zip(classes_to_plot, colors):
        plt.plot(
            recall[i], precision[i], color=color, lw=2, alpha=0.7,
            label=f'{class_names[i]} (AP = {avg_precision[i]:.3f})'
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved PR curves to: {save_path}")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: list
) -> dict:
    """
    Evaluate model on test/validation set.

    Args:
        model: PyTorch model
        dataloader: Test data loader
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: List of class names

    Returns:
        Tuple of (metrics dict, metrics_calculator, all_targets, all_probs)
    """
    model.eval()

    metrics_calc = MetricsCalculator(num_classes, class_names)

    # Store all predictions and targets for ROC/PR curves
    all_targets = []
    all_probs = []

    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL")
    print(f"{'='*80}\n")

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing')

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(dim=1)

            # Update metrics
            metrics_calc.update(preds, targets, probs)

            # Store for ROC/PR curves
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Compute all metrics
    metrics = metrics_calc.compute()

    # Concatenate all targets and probabilities
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    return metrics, metrics_calc, all_targets, all_probs


def print_results(
    metrics: dict,
    speed_metrics: dict = None,
    roc_data: dict = None,
    pr_data: dict = None
):
    """Print test results in formatted table."""
    print(f"\n{'='*80}")
    print("TEST RESULTS")
    print(f"{'='*80}\n")

    # Overall metrics
    print("Overall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
    print()

    # Precision, Recall, F1
    print("Macro Metrics (equal weight per class):")
    print(f"  Precision:          {metrics['precision_macro']:.4f}")
    print(f"  Recall:             {metrics['recall_macro']:.4f}")
    print(f"  F1 Score:           {metrics['f1_macro']:.4f}")
    print()

    print("Weighted Metrics (weighted by class size):")
    print(f"  Precision:          {metrics['precision_weighted']:.4f}")
    print(f"  Recall:             {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score:           {metrics['f1_weighted']:.4f}")
    print()

    # Top-k accuracy
    if 'top_3_accuracy' in metrics:
        print("Top-K Accuracy:")
        print(f"  Top-1:              {metrics['top_1_accuracy']:.4f} ({metrics['top_1_accuracy']*100:.2f}%)")
        print(f"  Top-3:              {metrics['top_3_accuracy']:.4f} ({metrics['top_3_accuracy']*100:.2f}%)")
        print(f"  Top-5:              {metrics['top_5_accuracy']:.4f} ({metrics['top_5_accuracy']*100:.2f}%)")
        print()

    # ROC AUC
    if roc_data:
        roc_auc = roc_data['auc']
        print("ROC AUC Scores:")
        print(f"  Micro-average:      {roc_auc['micro']:.4f}")
        print(f"  Macro-average:      {roc_auc['macro']:.4f}")
        print()

    # Average Precision
    if pr_data:
        avg_precision = pr_data['average_precision']
        print("Average Precision (AP) Scores:")
        print(f"  Micro-average:      {avg_precision['micro']:.4f}")
        print(f"  Macro-average:      {avg_precision['macro']:.4f}")
        print()

    # Inference speed
    if speed_metrics:
        print("Inference Speed:")
        print(f"  Average:            {speed_metrics['avg_time_per_sample_ms']:.2f} ms/sample")
        print(f"  Throughput:         {speed_metrics['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  Device:             {speed_metrics['device']}")
        print()


def save_results(
    metrics: dict,
    metrics_calc: MetricsCalculator,
    speed_metrics: dict,
    args: argparse.Namespace,
    output_dir: Path,
    all_targets: np.ndarray = None,
    all_probs: np.ndarray = None,
    class_names: list = None,
    num_classes: int = None
):
    """Save all test results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}\n")

    # Compute ROC curves if requested
    roc_data = None
    if args.save_roc_curves and all_targets is not None and all_probs is not None:
        print("Computing ROC curves...")
        roc_data = compute_roc_auc(all_targets, all_probs, num_classes)

    # Compute PR curves if requested
    pr_data = None
    if args.save_pr_curves and all_targets is not None and all_probs is not None:
        print("Computing PR curves...")
        pr_data = compute_pr_curves(all_targets, all_probs, num_classes)

    # Save overall metrics
    results = {
        'checkpoint': str(args.checkpoint),
        'model_name': args.model_name,
        'split': args.split,
        'fold': args.fold,
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'balanced_accuracy': float(metrics['balanced_accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'precision_weighted': float(metrics['precision_weighted']),
            'recall_weighted': float(metrics['recall_weighted']),
            'f1_weighted': float(metrics['f1_weighted']),
        },
        'speed_metrics': speed_metrics
    }

    # Add top-k if available
    for k in [1, 3, 5]:
        key = f'top_{k}_accuracy'
        if key in metrics:
            results['metrics'][key] = float(metrics[key])

    # Add ROC AUC scores
    if roc_data:
        results['metrics']['roc_auc_micro'] = float(roc_data['auc']['micro'])
        results['metrics']['roc_auc_macro'] = float(roc_data['auc']['macro'])

    # Add Average Precision scores
    if pr_data:
        results['metrics']['average_precision_micro'] = float(pr_data['average_precision']['micro'])
        results['metrics']['average_precision_macro'] = float(pr_data['average_precision']['macro'])

    metrics_path = output_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved overall metrics to: {metrics_path}")

    # Save per-class metrics
    if args.save_per_class_metrics:
        per_class_path = output_dir / 'per_class_metrics.json'
        with open(per_class_path, 'w') as f:
            json.dump(metrics['per_class'], f, indent=2)
        print(f"✓ Saved per-class metrics to: {per_class_path}")

    # Save confusion matrix
    if args.save_confusion_matrix:
        cm_path = output_dir / 'confusion_matrix.png'
        metrics_calc.plot_confusion_matrix(
            save_path=str(cm_path),
            normalize=True,
            figsize=(16, 14)
        )
        print(f"✓ Saved confusion matrix to: {cm_path}")

    # Save ROC curves
    if roc_data and class_names:
        roc_path = output_dir / 'roc_curves.png'
        plot_roc_curves(roc_data, class_names, str(roc_path))

    # Save PR curves
    if pr_data and class_names:
        pr_path = output_dir / 'pr_curves.png'
        plot_pr_curves(pr_data, class_names, str(pr_path))

    # Print and save worst performing classes
    worst_classes = metrics_calc.get_worst_performing_classes(n=10, metric='f1')
    print("\nWorst Performing Classes (by F1):")
    for i, (class_name, f1_score) in enumerate(worst_classes, 1):
        print(f"  {i}. {class_name}: {f1_score:.4f}")

    # Print and save best performing classes
    best_classes = metrics_calc.get_best_performing_classes(n=10, metric='f1')
    print("\nBest Performing Classes (by F1):")
    for i, (class_name, f1_score) in enumerate(best_classes, 1):
        print(f"  {i}. {class_name}: {f1_score:.4f}")

    # Save to file
    class_performance = {
        'worst_performing': [
            {'rank': i+1, 'class': name, 'f1_score': float(score)}
            for i, (name, score) in enumerate(worst_classes)
        ],
        'best_performing': [
            {'rank': i+1, 'class': name, 'f1_score': float(score)}
            for i, (name, score) in enumerate(best_classes)
        ]
    }

    performance_path = output_dir / 'class_performance.json'
    with open(performance_path, 'w') as f:
        json.dump(class_performance, f, indent=2)
    print(f"\n✓ Saved class performance analysis to: {performance_path}")


def main():
    """Main testing function."""
    args = parse_args()

    print(f"\n{'='*80}")
    print("MODEL TESTING")
    print(f"{'='*80}\n")

    # Configuration
    config = get_default_config()
    config.data.data_dir = args.data_dir
    config.data.image_size = args.image_size
    config.data.num_workers = args.num_workers
    config.data.batch_size = args.batch_size  # Set batch_size in DataConfig
    config.model.name = args.model_name
    config.training.batch_size = args.batch_size
    config.training.num_folds = args.num_folds
    config.training.fold_to_train = args.fold
    config.training.device = args.device

    print(f"Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Model: {args.model_name}")
    print(f"  Split: {args.split}")
    print(f"  Fold: {args.fold}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")

    # Get device
    device = get_device(args.device)
    print(f"\nUsing device: {device}")

    # Prepare data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}\n")

    # Use k-fold to get correct split
    folds, label_encoder, class_weights, num_classes = prepare_data_for_training(
        config.data,
        config.augmentation,
        use_kfold=True
    )

    # Update num_classes in config
    config.model.num_classes = num_classes

    # Get dataloaders for specified fold
    fold_data = folds[args.fold]
    dataloaders = get_dataloaders(
        config.data,
        config.augmentation,
        fold_data,
        label_encoder
    )

    # Select test dataloader
    test_loader = dataloaders[args.split]
    class_names = label_encoder.classes_.tolist()

    print(f"Test set size: {len(test_loader.dataset)} images")
    print(f"Number of classes: {num_classes}")
    print(f"Number of batches: {len(test_loader)}")

    # Create model
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}\n")

    model = create_model(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        device,
        strict=True
    )

    model = model.to(device)
    model.eval()

    if 'epoch' in checkpoint:
        print(f"Loaded checkpoint from epoch: {checkpoint['epoch'] + 1}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    # Benchmark inference speed
    speed_metrics = benchmark_inference_speed(
        model,
        test_loader,
        device,
        num_samples=args.num_inference_samples
    )

    # Evaluate model
    metrics, metrics_calc, all_targets, all_probs = evaluate_model(
        model,
        test_loader,
        device,
        num_classes,
        class_names
    )

    # Compute ROC and PR curves for display (if requested)
    roc_data = None
    pr_data = None

    if args.save_roc_curves:
        print("\nComputing ROC curves for display...")
        roc_data = compute_roc_auc(all_targets, all_probs, num_classes)

    if args.save_pr_curves:
        print("Computing PR curves for display...")
        pr_data = compute_pr_curves(all_targets, all_probs, num_classes)

    # Print results
    print_results(metrics, speed_metrics, roc_data, pr_data)

    # Save results
    output_dir = Path(args.output_dir) / f"fold_{args.fold}" / args.split
    save_results(
        metrics, metrics_calc, speed_metrics, args, output_dir,
        all_targets, all_probs, class_names, num_classes
    )

    print(f"\n{'='*80}")
    print("TESTING COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
