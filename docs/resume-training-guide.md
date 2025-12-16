# Resume Training Guide

## Quick Start

### Option 1: Automatic Resume (Recommended)

Resume from the latest checkpoint automatically:

```bash
./resume_training.sh
```

With modified parameters:
```bash
./resume_training.sh --batch_size 64 --lr 0.0002
```

### Option 2: Manual Resume

Specify exact checkpoint:
```bash
python3 scripts/train.py --resume outputs/checkpoints/fold_0/checkpoint_epoch_28.pt
```

With modified parameters:
```bash
python3 scripts/train.py \
  --resume outputs/checkpoints/fold_0/checkpoint_epoch_28.pt \
  --batch_size 64 \
  --lr 0.0002
```

## Finding Checkpoints

### List All Checkpoints

```bash
find outputs/checkpoints -name "*.pt" -type f
```

### Find Latest Checkpoint

```bash
find outputs/checkpoints/fold_0 -name "checkpoint_epoch_*.pt" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-
```

### Check Checkpoint Details

```bash
python3 -c "
import torch
cp = torch.load('outputs/checkpoints/fold_0/checkpoint_epoch_28.pt', map_location='cpu')
print(f\"Epoch: {cp['epoch']}\")
print(f\"Metrics: {cp['metrics']}\")
"
```

## What Gets Resumed

When you resume training, the following are restored:

✅ **Model weights** - Exact state from checkpoint epoch
✅ **Optimizer state** - Momentum, learning rate schedule
✅ **Scheduler state** - Learning rate warmup/decay progress
✅ **AMP scaler state** - Mixed precision training state
✅ **Epoch counter** - Continues from next epoch

## Important Notes

### ⚠️ Changing Hyperparameters Mid-Training

**Safe to change:**
- ✅ `--num_epochs` (extend training)
- ✅ `--early_stopping_patience`
- ✅ `--checkpoint_dir` (save to different location)

**Can cause issues:**
- ⚠️ `--batch_size` - Optimizer state won't match new batch stats
- ⚠️ `--lr` - Will override scheduler's current LR
- ⚠️ `--model_name` - Different architecture, won't load
- ⚠️ `--num_classes` - Model output size mismatch

**Recommendation:**
- If changing `batch_size` or `lr`, better to start fresh
- Use resume only for extending training or recovering from crashes

### Resume After Crash/Interruption

If training was interrupted (Ctrl+C, power loss, etc.):

```bash
# Find last saved checkpoint
ls -lht outputs/checkpoints/fold_0/ | head -5

# Resume from it
python3 scripts/train.py --resume outputs/checkpoints/fold_0/checkpoint_epoch_XX.pt
```

### Resume Best Model

To continue training from the best performing model:

```bash
python3 scripts/train.py --resume outputs/checkpoints/fold_0/best_model.pt
```

## Examples

### Example 1: Extend Training

Started with 50 epochs, want to train for 20 more:

```bash
# Original training stopped at epoch 49
python3 scripts/train.py \
  --resume outputs/checkpoints/fold_0/checkpoint_epoch_49.pt \
  --num_epochs 70  # 50 + 20 more
```

### Example 2: Resume After Interruption

Training crashed at epoch 28:

```bash
./resume_training.sh
# Automatically finds checkpoint_epoch_28.pt and resumes from epoch 29
```

### Example 3: Resume with Different Early Stopping

Want to be more patient with early stopping:

```bash
python3 scripts/train.py \
  --resume outputs/checkpoints/fold_0/best_model.pt \
  --early_stopping_patience 20  # Was 10 before
```

### Example 4: Resume Specific Fold

For k-fold cross-validation:

```bash
# Resume fold 2
python3 scripts/train.py \
  --resume outputs/checkpoints/fold_2/checkpoint_epoch_15.pt \
  --fold 2
```

## Troubleshooting

### Error: "Checkpoint not found"

```bash
# Check if file exists
ls -la outputs/checkpoints/fold_0/checkpoint_epoch_28.pt

# Use absolute path if needed
python3 scripts/train.py --resume /full/path/to/checkpoint.pt
```

### Error: "Key mismatch" or "Model architecture mismatch"

This happens when:
- Trying to load checkpoint from different model architecture
- Model code changed after checkpoint was saved

**Solution:**
- Use same `--model_name` as when checkpoint was created
- Or start fresh training with new architecture

### Optimizer State Doesn't Match

If you changed batch size and see warnings:

```
UserWarning: Optimizer state dict has different learning rate
```

**Solution:**
- This is expected when changing hyperparameters
- Consider starting fresh if results are poor
- Or ignore if just extending training

### Training Starts from Epoch 0

Check that:
1. You used `--resume` flag correctly
2. Checkpoint path is correct
3. Checkpoint file isn't corrupted

```bash
# Verify checkpoint loads
python3 -c "import torch; print(torch.load('path/to/checkpoint.pt', map_location='cpu').keys())"
```

## Best Practices

### 1. Always Save Regular Checkpoints

Default config saves checkpoints every epoch:
- `checkpoint_epoch_N.pt` - Every epoch (if `save_best_only=False`)
- `best_model.pt` - Best validation metric

### 2. Use Version Control for Checkpoints

For important experiments:
```bash
# Copy best checkpoint with descriptive name
cp outputs/checkpoints/fold_0/best_model.pt \
   saved_models/efficientnet_b0_epoch50_f1_0.85.pt
```

### 3. Monitor Disk Space

Checkpoints are large (~270MB each):
```bash
# Check checkpoint directory size
du -sh outputs/checkpoints/

# Remove old checkpoints (keep only best and latest 5)
find outputs/checkpoints/fold_0 -name "checkpoint_epoch_*.pt" | sort -V | head -n -5 | xargs rm
```

### 4. Test Resume Before Long Training

Before starting 100 epoch training:
```bash
# Train 2 epochs
python3 scripts/train.py --num_epochs 2

# Resume for 2 more
python3 scripts/train.py --resume outputs/checkpoints/fold_0/checkpoint_epoch_1.pt --num_epochs 4

# Verify it continues correctly
```

## Integration with MLflow

Resume training is tracked in MLflow:
- New run created (not continuation of old run)
- Can compare resumed vs original in MLflow UI
- Use run names to identify resumed runs:

```bash
# Add descriptive run name in code
mlflow.set_tag("resumed_from", "epoch_28")
```

## Advanced Usage

### Resume with Custom Learning Rate Schedule

```python
# In custom training script
checkpoint = torch.load('checkpoint.pt')
start_epoch = checkpoint['epoch'] + 1

# Adjust LR for remaining epochs
for epoch in range(start_epoch, total_epochs):
    # Custom LR schedule
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

### Resume Across Different Machines

```bash
# On machine A (training)
python3 scripts/train.py --num_epochs 50
# Interrupted at epoch 25

# Copy checkpoint to machine B
scp outputs/checkpoints/fold_0/checkpoint_epoch_25.pt user@machine_b:/path/

# On machine B (resume)
python3 scripts/train.py --resume checkpoint_epoch_25.pt --num_epochs 50
```

## Summary

**Resume training in 3 steps:**

1. **Find checkpoint:**
   ```bash
   ls -lht outputs/checkpoints/fold_0/
   ```

2. **Resume:**
   ```bash
   ./resume_training.sh
   ```

3. **Verify:**
   - Check epoch number in logs
   - Monitor metrics continue from previous values
   - Verify in MLflow UI

🎯 **Tip:** Use `./resume_training.sh` for simplest workflow - it finds latest checkpoint automatically!
