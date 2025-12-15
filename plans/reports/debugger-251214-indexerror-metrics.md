# Debug Report: IndexError in Metrics Calculation

**Date**: 2025-12-14
**Issue**: IndexError during validation metrics computation
**Severity**: Critical - blocks training

## Error Summary

```
IndexError: index 66 is out of bounds for axis 0 with size 66
```

Location: `src/metrics.py:129` in `compute_per_class_metrics()`

## Root Cause Analysis

### 1. **Model Predicting Out-of-Range Class Index**

**Primary Issue**: Model outputs class index 66, but dataset has only 66 classes (indices 0-65)

Dataset has **67 breeds** (confirmed via README.md and directory count), but:
- Valid class indices: 0-65 (66 classes total)
- Model prediction: 66 (invalid index)
- This occurs when model outputs argmax = 66

### 2. **sklearn Behavior with Unknown Classes**

When `precision_recall_fscore_support` receives predictions containing classes not in targets:
- Returns metrics array of size = max(unique(targets), unique(preds)) + 1
- If preds contain class 66, targets contain 0-65 → output arrays have 67 elements
- Code expects array size = num_classes = 66

**Warning observed**: `y_pred contains classes not in y_true`

### 3. **Iteration Mismatch**

Code at line 127-133:
```python
for i, class_name in enumerate(self.class_names):  # iterates 0-65 (66 items)
    per_class[class_name] = {
        'precision': float(precision[i]),  # precision has 67 elements
        ...
    }
```

Works fine for i=0-65, but if precision array has 67 elements due to class 66 prediction:
- Loop tries to access indices 0-65
- If model predicted class 66, sklearn adds extra element at index 66
- When i=65, trying to access precision[65] works
- But the REAL issue: **model shouldn't predict class 66**

### 4. **Actual Bug Location**

Model architecture issue:
- num_classes initialized as 66 (from label_encoder with 66 classes)
- Final linear layer: `nn.Linear(features, 66)` → outputs logits for classes 0-65
- `torch.argmax()` should return max index 65
- **But model IS outputting index 66** → indicates model has 67 output neurons, not 66

## Evidence Chain

1. README states 67 breeds
2. Directory count: 67 breed folders
3. LabelEncoder should create 66 unique classes (0-66 range = 67 classes)
4. Code sets `num_classes = len(label_encoder.classes_)` → should be 67, not 66
5. Warning: "y_pred contains classes not in y_true" confirms prediction mismatch

## Root Cause

**Inconsistent class counting logic**:

In `dataset.py:306-307`:
```python
labels = label_encoder.transform(breed_names)
num_classes = len(label_encoder.classes_)  # This should be 67
```

But somewhere in the pipeline:
- If label_encoder has 67 classes → num_classes should be 67
- Model gets num_classes = 66 (off-by-one error)
- Model outputs 66 logits (indices 0-65)
- But somehow predicts index 66

**Most likely**: LabelEncoder correctly identifies 67 classes, but:
1. Model receives wrong num_classes (66 instead of 67)
2. OR validation data has class 66 but model only trained on 0-65

## Verification Needed

Check actual values during runtime:
1. `len(label_encoder.classes_)` - should be 67
2. `self.num_classes` in Trainer - check if 66 or 67
3. Model final layer output size - should match num_classes
4. Unique values in validation targets

## Proposed Fixes

### Fix 1: Add Labels Parameter to sklearn Metrics (Immediate)

In `metrics.py:122-124`, specify explicit labels:
```python
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds,
    labels=np.arange(self.num_classes),  # ADD THIS
    average=None,
    zero_division=0
)
```

This ensures output arrays always have size = num_classes, ignoring unexpected predictions.

### Fix 2: Verify num_classes Throughout Pipeline

1. Print/log `len(label_encoder.classes_)` after creation
2. Print/log `self.num_classes` in Trainer.__init__
3. Print/log model output shape in forward pass
4. Add assertion: `assert model.classifier[-1].out_features == self.num_classes`

### Fix 3: Add Prediction Clipping (Defensive)

In `trainer.py:392`, add bounds checking:
```python
_, preds = outputs.max(dim=1)
preds = torch.clamp(preds, 0, self.num_classes - 1)  # ADD THIS
```

### Fix 4: Investigation Script

Run before training:
```python
from src.dataset import load_dataset_metadata

image_paths, breed_names, label_encoder = load_dataset_metadata('data/images')
print(f"Unique breeds: {len(set(breed_names))}")
print(f"Label encoder classes: {len(label_encoder.classes_)}")
print(f"Label range: {min(label_encoder.transform(breed_names))} - {max(label_encoder.transform(breed_names))}")
```

## Recommended Action Plan

1. **Immediate**: Apply Fix 1 (labels parameter) to prevent crash
2. **Diagnostic**: Add logging to verify num_classes value
3. **Investigation**: Check if 66 or 67 classes exist
4. **Root fix**: Ensure num_classes = len(label_encoder.classes_) = 67

## Unresolved Questions

1. Why does model predict class 66 if it only has 66 output neurons?
2. Is label_encoder creating 66 or 67 classes from 67 breed folders?
3. Are all 67 breeds present in validation fold or only 66?
4. Is there an off-by-one error in class counting logic?
