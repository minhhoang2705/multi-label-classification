# Bug Reports & Lessons Learned

## Critical: Focal Loss Alpha Weighting Bug (2025-12-15)

### Symptoms
- Training accuracy stuck at 0.00 after multiple epochs
- Training loss increasing/diverging instead of decreasing
- Validation accuracy also 0.00
- No NaN values, but model not learning

### Root Cause

**Incorrect alpha weighting implementation for multi-class classification**

Location: `src/losses.py:70` (FocalLoss class)

```python
# WRONG - treats multi-class labels as binary values
alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
focal_loss = alpha_t * focal_loss
```

**Why this fails:**

1. Focal loss alpha parameter designed for binary classification (targets ∈ {0, 1})
2. Our dataset has 67 classes with labels 0-66
3. When target=10 and alpha=0.25:
   ```
   alpha_t = 0.25 * 10 + 0.75 * (1 - 10)
          = 2.5 + 0.75 * (-9)
          = 2.5 - 6.75
          = -4.25  ❌ NEGATIVE WEIGHT!
   ```
4. Negative weights flip gradient direction → optimizer moves in wrong direction
5. Loss diverges, model can't learn

### The Fix

**Option 1: Disable alpha weighting (recommended for multi-class)**
```python
# Removed incorrect alpha calculation
# focal_loss = (1 - p) ** self.gamma * ce_loss  # Just use gamma parameter
```

**Option 2: Use per-class alpha tensor (advanced)**
```python
# Would require alpha as tensor of shape [num_classes]
# alpha_t = alpha[targets]
# focal_loss = alpha_t * focal_loss
```

**Option 3: Switch to weighted cross-entropy**
```bash
python3 scripts/train.py --loss_type weighted_ce
```

### Prevention

**Code review checklist:**
- [ ] Is loss function tested with actual dataset labels?
- [ ] Does implementation handle multi-class vs binary correctly?
- [ ] Are scalar parameters broadcasting correctly with tensor shapes?
- [ ] Does the loss decrease in a minimal training test?

**Warning signs:**
- Loss diverging from epoch 1
- Accuracy stuck at 0.00 or random guessing (1/num_classes)
- Gradients with unusual magnitudes (check with `torch.nn.utils.clip_grad_norm_`)

### Debugging Process

1. Check loss behavior: decreasing vs flat vs diverging
2. Verify label encoding: print actual label values
3. Test loss function in isolation:
   ```python
   import torch
   loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
   dummy_logits = torch.randn(4, 67)  # batch=4, classes=67
   dummy_targets = torch.tensor([0, 10, 33, 66])
   loss = loss_fn(dummy_logits, dummy_targets)
   print(loss)  # Should be positive and reasonable
   ```
4. Check for NaN/Inf: `torch.isnan(loss).any()`, `torch.isinf(loss).any()`
5. Verify gradients: `model.parameters()` should have reasonable values

### References

- Original Focal Loss paper: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- Note: Paper focuses on binary object detection; multi-class adaptation requires care
- Alpha parameter primarily for balancing positive/negative in binary case
- For multi-class imbalance, use class weights instead

### Lessons Learned

1. **Test loss functions with realistic data** before full training
2. **Binary vs multi-class** implementations differ significantly
3. **Don't blindly copy formulas** - understand assumptions
4. **Monitor metrics from epoch 1** - divergence indicates fundamental issues
5. **Start simple** - use standard CrossEntropy before exotic losses
