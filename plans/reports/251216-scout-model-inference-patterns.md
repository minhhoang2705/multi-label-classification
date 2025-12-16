# Model Loading & Inference Patterns - Scout Report

## Executive Summary
Found comprehensive model loading, inference, and metrics patterns in multi-class classification project using PyTorch/timm. Key files: `src/models.py` (model loading), `scripts/test.py` (inference), `src/trainer.py` (training/checkpoint flow), `src/metrics.py` (evaluation).

---

## 1. Model Loading Patterns

### Primary Factory: `create_model()` + Checkpoint Loading
**File:** `/home/minh-ubs-k8s/multi-label-classification/src/models.py`

#### Model Creation:
```python
def create_model(config: ModelConfig) -> nn.Module:
    """Create model from configuration."""
    model = TransferLearningModel(config)
    print(f"\nModel: {config.name}")
    print(f"Pretrained: {config.pretrained}")
    print(f"Num classes: {config.num_classes}")
    print(f"Total parameters: {model.get_num_total_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    return model
```

#### Model Class - TransferLearningModel:
```python
class TransferLearningModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Create backbone using timm (PyTorch Image Models)
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
```

#### Checkpoint Loading:
```python
def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cuda',
    strict: bool = True
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    print(f"Loaded checkpoint from: {checkpoint_path}")
    return checkpoint
```

**Key Details:**
- Uses TIMM library for pretrained backbones (ResNet, EfficientNet, ViT, etc.)
- Custom classifier head with dropout for regularization
- Supports backbone freezing for fine-tuning
- Checkpoint includes: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `epoch`, `metrics`

---

## 2. Inference Patterns

### Test/Evaluation Function
**File:** `/home/minh-ubs-k8s/multi-label-classification/scripts/test.py`

#### Core Inference Loop:
```python
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: list
) -> dict:
    """Evaluate model on test/validation set."""
    model.eval()
    
    metrics_calc = MetricsCalculator(num_classes, class_names)
    all_targets = []
    all_probs = []
    
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
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    
    return metrics, metrics_calc, all_targets, all_probs
```

#### Inference Speed Benchmarking:
```python
def benchmark_inference_speed(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 1000
) -> dict:
    """Benchmark model inference speed."""
    model.eval()
    times = []
    total_samples = 0
    
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
    
    return metrics
```

**Key Features:**
- Uses `model.eval()` for inference mode
- `torch.no_grad()` context to disable gradient computation
- CUDA synchronization for accurate timing
- Warmup batch before benchmarking
- Captures both predictions and probabilities

---

## 3. Trainer Inference Patterns

**File:** `/home/minh-ubs-k8s/multi-label-classification/src/trainer.py`

#### Validation Loop (Training Time):
```python
def validate_epoch(
    self,
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    epoch: int
) -> Dict[str, float]:
    """Validate for one epoch."""
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
```

---

## 4. Test Metrics JSON Structure

**Location:** `/home/minh-ubs-k8s/multi-label-classification/outputs/test_results/fold_{fold_idx}/{split}/test_metrics.json`

**Example Output:**
```json
{
  "checkpoint": "outputs/checkpoints/fold_0/best_model.pt",
  "model_name": "resnet50",
  "split": "val",
  "fold": 0,
  "metrics": {
    "accuracy": 0.5496406287023142,
    "balanced_accuracy": 0.23102339124059035,
    "precision_macro": 0.3271334422638197,
    "recall_macro": 0.23102339124059035,
    "f1_macro": 0.2579920324445541,
    "precision_weighted": 0.5154120670400905,
    "recall_weighted": 0.5496406287023142,
    "f1_weighted": 0.5219218436581097,
    "top_1_accuracy": 0.5496406287023142,
    "top_3_accuracy": 0.7949609035621199,
    "top_5_accuracy": 0.8673485506674038,
    "roc_auc_micro": 0.9668315332570956,
    "roc_auc_macro": "NaN",
    "average_precision_micro": 0.5307894423487324,
    "average_precision_macro": 0.2255990834651285
  },
  "speed_metrics": {
    "samples_tested": 1024,
    "avg_time_per_sample_ms": 0.8757092955420376,
    "std_time_per_sample_ms": 0.0008106277182442709,
    "min_time_per_sample_ms": 0.8748576328798663,
    "max_time_per_sample_ms": 0.8776212816883344,
    "median_time_per_sample_ms": 0.8755318776820786,
    "throughput_samples_per_sec": 1141.931466401793,
    "device": "cuda"
  }
}
```

**Metric Categories:**
1. **Overall Metrics:** accuracy, balanced_accuracy
2. **Macro Metrics:** precision_macro, recall_macro, f1_macro (equal weight per class)
3. **Weighted Metrics:** precision_weighted, recall_weighted, f1_weighted (weight by class frequency)
4. **Top-K Accuracy:** top_1/3/5_accuracy
5. **AUC Scores:** roc_auc_micro, roc_auc_macro
6. **Average Precision:** average_precision_micro, average_precision_macro
7. **Speed Metrics:** avg_time_per_sample_ms, throughput_samples_per_sec, device

---

## 5. Checkpoint File Structure

**Location:** `/home/minh-ubs-k8s/multi-label-classification/outputs/checkpoints/fold_{fold_idx}/`

**Files Saved:**
- `best_model.pt` - Best checkpoint (selected by early_stopping_metric)
- `checkpoint_epoch_{epoch}.pt` - All epoch checkpoints (if not save_best_only)

**Checkpoint Contents (from `save_checkpoint()`):**
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),  # Optional
    'scaler_state_dict': scaler.state_dict(),       # Optional (AMP)
    'metrics': metrics                               # Dict of metrics
}
```

**Resume Training Pattern:**
```python
# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint_path, device, strict=True)

# Load optimizer state
if 'optimizer_state_dict' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Load scheduler state
if scheduler is not None and 'scheduler_state_dict' in checkpoint:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Load AMP scaler state
if scaler is not None and 'scaler_state_dict' in checkpoint:
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

# Resume from next epoch
start_epoch = checkpoint.get('epoch', 0) + 1
```

---

## 6. Key Files Summary

| File | Purpose | Key Functions |
|------|---------|----------------|
| `/home/minh-ubs-k8s/multi-label-classification/src/models.py` | Model architecture & loading | `create_model()`, `load_checkpoint()`, `save_checkpoint()`, `TransferLearningModel` |
| `/home/minh-ubs-k8s/multi-label-classification/scripts/test.py` | Inference & evaluation | `evaluate_model()`, `benchmark_inference_speed()`, `compute_roc_auc()`, `compute_pr_curves()` |
| `/home/minh-ubs-k8s/multi-label-classification/src/trainer.py` | Training pipeline | `train_fold()`, `validate_epoch()`, checkpoint resume logic |
| `/home/minh-ubs-k8s/multi-label-classification/src/metrics.py` | Metrics calculation | `MetricsCalculator`, `MetricTracker` |
| `/home/minh-ubs-k8s/multi-label-classification/src/utils.py` | Utilities | `get_device()`, `set_seed()`, optimizer/scheduler creation |

---

## 7. Supported Model Architectures (via TIMM)

From `get_available_models()`:
- ResNet family: resnet18/34/50/101/152, resnext50_32x4d, wide_resnet50_2
- EfficientNet: efficientnet_b0-b7, efficientnetv2_s/m/l
- Vision Transformer: vit_tiny/small/base/large_patch16_224
- ConvNeXt: convnext_tiny/small/base/large
- MobileNet: mobilenetv3_small/large_100
- DenseNet: densenet121/169/201
- RegNet: regnetx/regnety variants
- Swin Transformer: swin_tiny/small/base_patch4_window7_224

---

## 8. Inference Configuration Parameters

**From `scripts/test.py` CLI Args:**
- `--checkpoint`: Path to model checkpoint (required)
- `--model_name`: Model architecture (default: 'resnet50')
- `--batch_size`: Batch size for testing (default: 64)
- `--image_size`: Input image size (default: 224)
- `--split`: 'val' or 'test' (default: 'val')
- `--fold`: Which fold to evaluate (default: 0)
- `--num_folds`: Total folds used in training (default: 5)
- `--device`: 'cuda', 'cpu', 'mps' (default: 'cuda')
- `--num_inference_samples`: Samples for speed benchmark (default: 1000)

---

## Unresolved Questions
- None identified. All patterns clearly documented.
