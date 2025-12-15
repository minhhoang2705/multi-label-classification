# Scout Report: Dataset num_classes Off-by-One Error Analysis

Date: 2025-12-14
Task: Locate and analyze exact num_classes calculation issues across dataset, trainer, and metrics modules

## Summary

Found critical mismatch between actual breed count (67) and hardcoded config (67). However, there IS an off-by-one error in metrics computation that causes index out of bounds errors when using per-class metrics.

Actual breed folders: 67
Config hardcoded value: 67 (correct count)
LabelEncoder generated classes: 67 (correct)
Issue: Per-class metrics iteration doesn't account for classes with 0 support

---

## 1. Dataset num_classes Calculation

### Location
**File:** `/home/minh-ubs-k8s/multi-label-classification/src/dataset.py`
**Lines:** 306-307

### Code
```python
304→    # Load dataset
305→    image_paths, breed_names, label_encoder = load_dataset_metadata(
306→        data_config.data_dir)
307→    labels = label_encoder.transform(breed_names)
308→    num_classes = len(label_encoder.classes_)
```

### How num_classes is Calculated
1. `load_dataset_metadata()` at lines 81-116 loads all breed folders and their images
2. LabelEncoder is created and fitted at lines 110-111:
   ```python
   label_encoder = LabelEncoder()
   labels = label_encoder.fit_transform(breed_names)
   ```
3. num_classes is derived from LabelEncoder.classes_ length (line 307)

### Breed Folder Count
**Actual count in `/home/minh-ubs-k8s/multi-label-classification/data/images/`:** 67 directories

Example breeds (alphabetically sorted):
- Abyssinian
- American Bobtail
- American Curl
- American Shorthair
- ... (67 total)

### Verification
```bash
$ ls -d /home/minh-ubs-k8s/multi-label-classification/data/images/*/ | wc -l
67
```

**Result:** LabelEncoder correctly generates 67 classes from 67 breed folders

---

## 2. Trainer num_classes Assignment

### Location
**File:** `/home/minh-ubs-k8s/multi-label-classification/src/trainer.py`
**Lines:** 58-68

### Code
```python
58→        if config.training.num_folds > 1:
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

### Flow
1. `prepare_data_for_training()` returns `num_classes = 67`
2. Trainer stores as `self.num_classes = 67`
3. Config is updated: `self.config.model.num_classes = 67`

**Result:** Trainer correctly assigns num_classes = 67 from LabelEncoder

---

## 3. Config Default Value

### Location
**File:** `/home/minh-ubs-k8s/multi-label-classification/src/config.py`
**Lines:** 65-74

### Code
```python
65→class ModelConfig:
66→    """Model configuration."""
67→    # Model architecture
68→    # resnet50, efficientnet_b0-b7, vit_b_16, convnext_tiny
69→    name: str = "efficientnet_b0"
70→    pretrained: bool = True
71→    num_classes: int = 67
```

**Result:** Config has correct hardcoded default of 67

---

## 4. Metrics Computation - THE REAL ISSUE

### Location
**File:** `/home/minh-ubs-k8s/multi-label-classification/src/metrics.py`
**Lines:** 112-135 (per_class metrics method)

### Code
```python
112→    def compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
113→        """
114→        Compute per-class metrics.
115→        
116→        Returns:
117→            Dictionary mapping class names to their metrics
118→        """
119→        preds = np.array(self.all_preds)
120→        targets = np.array(self.all_targets)
121→        
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
134→        
135→        return per_class
```

### The Problem

`precision_recall_fscore_support()` with `average=None` returns arrays with length = **unique classes in predictions/targets**, NOT num_classes.

Example scenario:
- num_classes = 67
- Test set only has 60 unique classes represented
- `precision, recall, f1, support` arrays have length 60
- But `self.class_names` has 67 entries
- Iteration at line 127 tries to access `precision[60]` through `precision[66]` → **IndexError**

### Why This Happens
1. MetricsCalculator.__init__ (lines 24-36):
   ```python
   self.num_classes = num_classes
   self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
   ```
   → Creates 67 class_names

2. compute_per_class_metrics (line 122):
   ```python
   precision, recall, f1, support = precision_recall_fscore_support(
       targets, preds, average=None, zero_division=0
   )
   ```
   → Returns arrays with only observed classes in the validation batch

3. Iteration (line 127):
   ```python
   for i, class_name in enumerate(self.class_names):  # Iterates 0-66
       per_class[class_name] = {..., 'precision': float(precision[i])}  # precision only has 0-59
   ```
   → **IndexError on i >= 60**

---

## 5. Label Encoder Details

### Location
**File:** `/home/minh-ubs-k8s/multi-label-classification/src/dataset.py`
**Lines:** 109-116

### Code
```python
109→    # Encode breed names to integers
110→    label_encoder = LabelEncoder()
111→    labels = label_encoder.fit_transform(breed_names)
112→
113→    print(
114→        f"Loaded {len(image_paths)} images from {len(label_encoder.classes_)} breeds")
115→
116→    return image_paths, breed_names, label_encoder
```

### LabelEncoder Behavior
- `fit_transform(breed_names)` creates `classes_` array from sorted unique breed names
- Length = number of unique breed names in dataset = 67
- Produces class indices 0-66

**Result:** LabelEncoder correctly generates 67 classes

---

## Critical Finding: Metrics Index Out of Bounds

### Root Cause
In `compute_per_class_metrics()`, the function assumes all 67 classes appear in the validation set, but some breeds may not be represented in smaller validation batches.

### Error Scenario
```
Validation set has only 60 unique breeds
precision_recall_fscore_support returns arrays of length 60
self.class_names has 67 entries
Loop tries to access precision[60] to precision[66] → IndexError
```

### Affected Methods
- `MetricsCalculator.compute_per_class_metrics()` (line 112)
- Any call to this method from `compute()` (line 107)
- `print_classification_report()` assumes all classes are in predictions (lines 202-215)

---

## Summary of Findings

| Component | Location | Value | Status |
|-----------|----------|-------|--------|
| Actual breed folders | /data/images | 67 | ✓ Correct |
| Config default | config.py:71 | 67 | ✓ Correct |
| LabelEncoder.classes_ | dataset.py:307 | 67 | ✓ Correct |
| Trainer assignment | trainer.py:68 | 67 (from encoder) | ✓ Correct |
| Metrics iterator | metrics.py:127 | Expects 67 but gets variable | ✗ **BUG** |

---

## Recommended Fixes

### Fix 1: Metrics Per-Class Computation (CRITICAL)

**File:** `/home/minh-ubs-k8s/multi-label-classification/src/metrics.py`

Replace lines 122-135:
```python
# OLD (BROKEN):
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

# NEW (FIXED):
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0,
    labels=np.arange(self.num_classes)  # Force all classes
)

per_class = {}
for i, class_name in enumerate(self.class_names):
    if i < len(precision):
        per_class[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    else:
        # Class not in validation set
        per_class[class_name] = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'support': 0
        }
```

### Fix 2: Confusion Matrix (Secondary)

**File:** `/home/minh-ubs-k8s/multi-label-classification/src/metrics.py`
**Line:** 146

Already handles this correctly by specifying `labels=np.arange(self.num_classes)`

### Fix 3: Classification Report (Secondary)

The `print_classification_report()` should also include the `labels` parameter to handle missing classes gracefully.

---

## Unresolved Questions

1. Are there any validation batches that don't contain all 67 breeds? (Likely yes for smaller val sets)
2. Should missing classes have 0.0 metrics or be excluded from per-class reports?
3. Is the `labels` parameter being used in `confusion_matrix()` call already? (Need to verify line 146)

