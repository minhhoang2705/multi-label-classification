# Training Pipeline Guide

This guide explains how to use the ML training pipeline for the Cat Breeds Classification project.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training with Default Settings

```bash
python scripts/train.py
```

### 3. Fast Development Mode (2 epochs, 2 folds)

```bash
python scripts/train.py --fast_dev
```

## Training Modes

### Single Train/Val/Test Split

```bash
python scripts/train.py --num_folds 1
```

### K-Fold Cross-Validation (Default: 5 folds)

```bash
python scripts/train.py --num_folds 5
```

### Train Specific Fold

```bash
python scripts/train.py --num_folds 5 --fold 0
```

## Model Selection

### EfficientNet (Recommended)

```bash
# EfficientNet-B0 (fastest)
python scripts/train.py --model_name efficientnet_b0

# EfficientNet-B3 (balanced)
python scripts/train.py --model_name efficientnet_b3

# EfficientNet-B7 (best accuracy, slower)
python scripts/train.py --model_name efficientnet_b7
```

### ResNet

```bash
# ResNet-50
python scripts/train.py --model_name resnet50

# ResNet-101
python scripts/train.py --model_name resnet101
```

### Vision Transformer

```bash
# ViT Base
python scripts/train.py --model_name vit_base_patch16_224

# ViT Small (faster)
python scripts/train.py --model_name vit_small_patch16_224
```

### ConvNeXt

```bash
# ConvNeXt Tiny
python scripts/train.py --model_name convnext_tiny

# ConvNeXt Base
python scripts/train.py --model_name convnext_base
```

## Loss Functions

### Focal Loss (Default - Best for Imbalanced Data)

```bash
python scripts/train.py --loss_type focal --focal_gamma 2.0 --focal_alpha 0.25
```

### Weighted Cross-Entropy

```bash
python scripts/train.py --loss_type weighted_ce --use_class_weights
```

### Standard Cross-Entropy

```bash
python scripts/train.py --loss_type ce
```

## Data Augmentation

### Basic Augmentation (Default)

Includes: resize, horizontal flip, rotation, color jitter, random erasing

### AutoAugment

```bash
python scripts/train.py --use_autoaugment
```

### RandAugment

```bash
python scripts/train.py --use_randaugment
```

### Mixup

```bash
python scripts/train.py --use_mixup --mixup_alpha 0.2
```

### CutMix

```bash
python scripts/train.py --use_cutmix --cutmix_alpha 1.0
```

### Combined Advanced Augmentation

```bash
python scripts/train.py \
    --use_randaugment \
    --use_mixup --mixup_alpha 0.2 \
    --use_cutmix --cutmix_alpha 1.0
```

## Optimizer and Scheduler

### AdamW (Default)

```bash
python scripts/train.py \
    --optimizer adamw \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --scheduler cosine \
    --warmup_epochs 5
```

### SGD with Momentum

```bash
python scripts/train.py \
    --optimizer sgd \
    --lr 1e-2 \
    --weight_decay 1e-4 \
    --scheduler step
```

### Learning Rate Schedulers

```bash
# Cosine Annealing (Default)
python scripts/train.py --scheduler cosine

# Step Decay
python scripts/train.py --scheduler step

# Reduce on Plateau
python scripts/train.py --scheduler plateau

# No Scheduler
python scripts/train.py --scheduler none
```

## Training Configuration

### Batch Size and Epochs

```bash
python scripts/train.py \
    --batch_size 64 \
    --num_epochs 100
```

### Image Size

```bash
# 224x224 (Default)
python scripts/train.py --image_size 224

# 384x384 (Better accuracy, slower)
python scripts/train.py --image_size 384
```

### Mixed Precision Training

```bash
# Enable (Default)
python scripts/train.py --use_amp

# Disable
python scripts/train.py --use_amp false
```

### Transfer Learning

```bash
# Freeze backbone for first 5 epochs
python scripts/train.py --freeze_backbone --freeze_epochs 5

# Train entire model from start
python scripts/train.py
```

## Early Stopping

```bash
# Enable with macro F1 metric (Default)
python scripts/train.py \
    --early_stopping \
    --early_stopping_patience 10 \
    --early_stopping_metric macro_f1

# Use balanced accuracy
python scripts/train.py \
    --early_stopping \
    --early_stopping_metric balanced_acc
```

## Experiment Tracking

### MLflow (Default)

```bash
# Enable MLflow
python scripts/train.py --use_mlflow --experiment_name my_experiment

# View results
mlflow ui --backend-store-uri mlruns
# Open browser to http://localhost:5000
```

### Disable MLflow

```bash
python scripts/train.py --use_mlflow false
```

## Example Configurations

### Quick Experiment (Fast Development)

```bash
python scripts/train.py \
    --fast_dev \
    --model_name efficientnet_b0 \
    --batch_size 32
```

### Production Training (Best Accuracy)

```bash
python scripts/train.py \
    --model_name efficientnet_b3 \
    --batch_size 64 \
    --num_epochs 100 \
    --num_folds 5 \
    --loss_type focal \
    --focal_gamma 2.0 \
    --use_randaugment \
    --use_mixup \
    --optimizer adamw \
    --lr 1e-4 \
    --scheduler cosine \
    --warmup_epochs 5 \
    --early_stopping \
    --early_stopping_patience 15 \
    --use_amp \
    --use_mlflow
```

### Lightweight Model (Fast Inference)

```bash
python scripts/train.py \
    --model_name mobilenetv3_large_100 \
    --batch_size 128 \
    --num_epochs 50 \
    --loss_type focal \
    --use_mixup
```

### Vision Transformer (State-of-the-art)

```bash
python scripts/train.py \
    --model_name vit_base_patch16_224 \
    --image_size 224 \
    --batch_size 32 \
    --num_epochs 100 \
    --loss_type focal \
    --optimizer adamw \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --use_mixup \
    --use_cutmix
```

## Monitoring Training

### During Training

Training progress is displayed with:
- Progress bars for each epoch
- Real-time loss and accuracy
- Validation metrics after each epoch
- Learning rate updates
- Early stopping counter

### After Training

Check the following locations:
- **Checkpoints**: `outputs/checkpoints/fold_X/`
- **MLflow UI**: Run `mlflow ui` and open http://localhost:5000
- **Best model**: `outputs/checkpoints/fold_X/best_model.pt`

## Metrics for Imbalanced Data

The pipeline tracks the following metrics optimized for imbalanced datasets:

1. **Balanced Accuracy**: Accounts for class imbalance
2. **Macro F1-Score**: Treats all classes equally
3. **Weighted F1-Score**: Weighted by class support
4. **Per-class Precision/Recall/F1**: Individual class performance
5. **Top-k Accuracy**: Top-3 and Top-5 predictions
6. **Confusion Matrix**: Saved as artifact

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python scripts/train.py --batch_size 16

# Reduce image size
python scripts/train.py --image_size 224

# Disable AMP (uses more memory but slower)
python scripts/train.py --use_amp false
```

### Training Too Slow

```bash
# Use smaller model
python scripts/train.py --model_name efficientnet_b0

# Reduce image size
python scripts/train.py --image_size 224

# Increase batch size (if memory allows)
python scripts/train.py --batch_size 64

# Reduce number of workers
python scripts/train.py --num_workers 2
```

### Poor Performance on Minority Classes

```bash
# Use focal loss with higher gamma
python scripts/train.py --loss_type focal --focal_gamma 3.0

# Add more augmentation
python scripts/train.py --use_randaugment --use_mixup --use_cutmix

# Increase class weight importance
python scripts/train.py --loss_type weighted_ce --use_class_weights
```

## Advanced Usage

### Custom Configuration

You can modify `src/config.py` to create custom configurations:

```python
from src.config import Config

# Create custom config
config = Config()
config.model.name = "efficientnet_b5"
config.training.num_epochs = 200
# ... modify other settings

# Use in trainer
from src.trainer import Trainer
trainer = Trainer(config)
trainer.train()
```

### Resume Training

```bash
# Load checkpoint and continue training
# (Implement in future version)
```

### Ensemble Models

Train multiple models and ensemble their predictions for better performance:

```bash
# Train different models
python scripts/train.py --model_name efficientnet_b3 --experiment_name ensemble_1
python scripts/train.py --model_name resnet50 --experiment_name ensemble_2
python scripts/train.py --model_name vit_base_patch16_224 --experiment_name ensemble_3
```

## Best Practices

1. **Start with fast_dev mode** to verify everything works
2. **Use focal loss** for imbalanced datasets
3. **Enable data augmentation** (mixup/cutmix) for better generalization
4. **Use k-fold CV** (5 folds) for robust evaluation
5. **Monitor macro F1** instead of accuracy for imbalanced data
6. **Use mixed precision** (AMP) for faster training
7. **Track experiments** with MLflow
8. **Save best model** based on validation macro F1
9. **Use early stopping** to prevent overfitting
10. **Start with pretrained models** for transfer learning

## Output Structure

```
outputs/
├── checkpoints/
│   ├── fold_0/
│   │   ├── best_model.pt
│   │   ├── checkpoint_epoch_0.pt
│   │   ├── checkpoint_epoch_1.pt
│   │   └── ...
│   ├── fold_1/
│   └── ...
mlruns/
├── 0/
│   ├── meta.yaml
│   └── ...
└── ...
```

## Next Steps

After training:
1. Evaluate on test set
2. Analyze per-class performance
3. Identify problematic classes
4. Fine-tune with adjusted augmentation
5. Create ensemble of best models
6. Deploy best model for inference

## Support

For issues or questions:
1. Check the configuration in `src/config.py`
2. Review the training logs
3. Check MLflow UI for experiment tracking
4. Examine per-class metrics for imbalanced classes

