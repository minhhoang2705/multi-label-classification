# Scout Report Summary: num_classes Off-by-One Bug Analysis

**Date:** 2025-12-14
**Status:** INVESTIGATION COMPLETE

---

## Key Findings

### Location Matrix

| Issue | File | Line(s) | Severity | Status |
|-------|------|---------|----------|--------|
| Dataset num_classes calc | `dataset.py` | 306-307 | NONE | ✓ Correct (67 classes) |
| Trainer assignment | `trainer.py` | 58-68 | NONE | ✓ Correct (67 classes) |
| Config hardcoded value | `config.py` | 71 | NONE | ✓ Correct (67 classes) |
| **Per-class metrics bug** | **`metrics.py`** | **122-135** | **CRITICAL** | **✗ IndexError** |
| Confusion matrix | `metrics.py` | 146 | NONE | ✓ Correct (uses labels param) |

---

## The Real Problem: metrics.py Lines 122-135

### What's Happening

The `compute_per_class_metrics()` method assumes all 67 classes will appear in the validation/test data, but scikit-learn's `precision_recall_fscore_support()` only returns metrics for **observed classes**.

### Code Flow

```python
# Line 122: Call without specifying labels parameter
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0
)
# Returns arrays of shape (N_OBSERVED_CLASSES,)
# If val set has 50 unique breeds, this returns arrays of length 50

# Line 127: But iterate through ALL 67 classes
for i, class_name in enumerate(self.class_names):  # 0-66
    per_class[class_name] = {
        'precision': float(precision[i]),  # IndexError when i >= 50!
        ...
    }
```

### Example Failure

Validation set contains only 60 unique breeds:
- `precision` array length = 60 (indices 0-59)
- `self.class_names` has 67 items
- Loop tries to access `precision[60]` through `precision[66]`
- **IndexError: index 60 is out of bounds for axis 0 with size 60**

---

## All Verified Locations

### 1. Dataset num_classes Calculation ✓

**File:** `/home/minh-ubs-k8s/multi-label-classification/src/dataset.py`

Lines 303-310:
```python
image_paths, breed_names, label_encoder = load_dataset_metadata(
    data_config.data_dir)
labels = label_encoder.transform(breed_names)
num_classes = len(label_encoder.classes_)  # <-- CORRECT: Returns 67
class_weights = compute_class_weights(labels, num_classes)
```

**Verification:**
- Actual breed folders: 67
- LabelEncoder.classes_: 67
- Result: `num_classes = 67` ✓

---

### 2. Trainer num_classes Assignment ✓

**File:** `/home/minh-ubs-k8s/multi-label-classification/src/trainer.py`

Lines 58-68:
```python
if config.training.num_folds > 1:
    self.folds, self.label_encoder, self.class_weights, self.num_classes = \
        prepare_data_for_training(config.data, config.augmentation, use_kfold=True)
    self.use_kfold = True
else:
    self.dataloaders, self.label_encoder, self.class_weights, self.num_classes = \
        prepare_data_for_training(config.data, config.augmentation, use_kfold=False)
    self.use_kfold = False

# Update num_classes in config
self.config.model.num_classes = self.num_classes  # <-- CORRECT: Assigns 67
```

**Verification:**
- `prepare_data_for_training()` returns 67
- Trainer stores `self.num_classes = 67`
- Config updated with correct value ✓

---

### 3. Metrics computation - THE BUG ✗

**File:** `/home/minh-ubs-k8s/multi-label-classification/src/metrics.py`

Lines 112-135:
```python
def compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
    preds = np.array(self.all_preds)
    targets = np.array(self.all_targets)
    
    # PROBLEM: Missing labels parameter!
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0
    )
    # Returns only observed classes, not all 67
    
    per_class = {}
    for i, class_name in enumerate(self.class_names):
        # CRASH: self.class_names has 67 items
        # But precision/recall/f1/support have fewer
        per_class[class_name] = {
            'precision': float(precision[i]),  # IndexError!
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return per_class
```

**Issue:** Missing `labels=np.arange(self.num_classes)` parameter

---

### 4. Label Encoder Details ✓

**File:** `/home/minh-ubs-k8s/multi-label-classification/src/dataset.py`

Lines 81-116 (`load_dataset_metadata` function):
```python
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(breed_names)
print(f"Loaded {len(image_paths)} images from {len(label_encoder.classes_)} breeds")
return image_paths, breed_names, label_encoder
```

**Verification:**
- LabelEncoder correctly generates 67 classes from 67 breed folders
- Sorted alphabetically: Abyssinian → York Chocolate
- Result: `label_encoder.classes_` has 67 items ✓

---

## Code Snippets by Location

### dataset.py Lines 306-307
```python
306→    labels = label_encoder.transform(breed_names)
307→    num_classes = len(label_encoder.classes_)
```

### trainer.py Lines 59-68
```python
 59→            self.folds, self.label_encoder, self.class_weights, self.num_classes = \
 60→                prepare_data_for_training(config.data, config.augmentation, use_kfold=True)
 61→            self.use_kfold = True
 62→        else:
 63→            self.dataloaders, self.label_encoder, self.class_weights, self.num_classes = \
 64→                prepare_data_for_training(config.data, config.augmentation, use_kfold=False)
 65→            self.use_kfold = False
 66→        
 67→        # Update num_classes in config
 68→        self.config.model.num_classes = self.num_classes
```

### metrics.py Lines 122-135 (THE BUG)
```python
122→        precision, recall, f1, support = precision_recall_fscore_support(
123→            targets, preds, average=None, zero_division=0
124→        )
125→        
126→        per_class = {}
127→        for i, class_name in enumerate(self.class_names):
128→            per_class[class_name] = {
129→                'precision': float(precision[i]),
130→                'recall': float(recall[i]),
131→                'f1': float(f1[i]),
132→                'support': int(support[i])
133→            }
```

### config.py Line 71
```python
 71→    num_classes: int = 67
```

---

## Fix Recommendations

### CRITICAL: Fix metrics.py compute_per_class_metrics()

**Change lines 122-135 to:**

```python
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0,
    labels=np.arange(self.num_classes)  # ADD THIS LINE
)

per_class = {}
for i, class_name in enumerate(self.class_names):
    per_class[class_name] = {
        'precision': float(precision[i]),
        'recall': float(recall[i]),
        'f1': float(f1[i]),
        'support': int(support[i])
    }
```

This ensures `precision`, `recall`, `f1`, and `support` all have 67 elements with zeros for unobserved classes.

### SECONDARY: Check print_classification_report()

Lines 210-214 should also specify labels parameter:

```python
print(classification_report(
    targets,
    preds,
    target_names=self.class_names,
    zero_division=0,
    labels=np.arange(self.num_classes)  # ADD THIS
))
```

---

## Files Analyzed

1. `/home/minh-ubs-k8s/multi-label-classification/src/dataset.py` - Full file (382 lines)
2. `/home/minh-ubs-k8s/multi-label-classification/src/trainer.py` - Full file (540 lines)
3. `/home/minh-ubs-k8s/multi-label-classification/src/metrics.py` - Full file (408 lines)
4. `/home/minh-ubs-k8s/multi-label-classification/src/config.py` - Partial (ModelConfig class)

## Data Verification

Breed folder count: 67 confirmed
- Directory: `/home/minh-ubs-k8s/multi-label-classification/data/images/`
- Command: `ls -d /home/minh-ubs-k8s/multi-label-classification/data/images/*/ | wc -l`
- Result: 67

---

## Conclusion

**No off-by-one error exists in num_classes calculation.** The real issue is in metrics computation where the code assumes all 67 classes are represented in validation data, causing IndexError when they're not. This is not technically an "off-by-one" but rather a missing parameter issue in scikit-learn function call.

**All num_classes calculations are correct: 67 is the right value.**
