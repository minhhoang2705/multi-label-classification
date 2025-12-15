# Detailed Code Analysis Report: num_classes Bug Investigation

**Date:** 2025-12-14
**Focus:** Exact locations and code analysis of num_classes handling across dataset, trainer, and metrics modules

---

## Executive Summary

Investigated three areas requested by debugger analysis:

1. **Dataset num_classes calculation** (dataset.py:306-307) - ✓ CORRECT
2. **Trainer num_classes assignment** (trainer.py:59-68) - ✓ CORRECT  
3. **Metrics computation** (metrics.py:122-135) - ✗ **CRITICAL BUG FOUND**

**Root Cause:** Missing `labels` parameter in `precision_recall_fscore_support()` call causes IndexError when validation set lacks all 67 breed classes.

---

## 1. Dataset num_classes Calculation (dataset.py)

### Location: Lines 303-310

```python
303→def prepare_data_for_training(
304→    data_config: DataConfig,
305→    aug_config: AugmentationConfig,
306→    use_kfold: bool = False
307→) -> Tuple:
308→    """
309→    Prepare all data for training.
310→    ...
311→    """
312→    # Load dataset
313→    image_paths, breed_names, label_encoder = load_dataset_metadata(
314→        data_config.data_dir)
315→    labels = label_encoder.transform(breed_names)
316→    num_classes = len(label_encoder.classes_)
```

### Code Path: How num_classes is Derived

#### Step 1: load_dataset_metadata() [Lines 81-116]

```python
 81→def load_dataset_metadata(data_dir: str) -> Tuple[List[str], List[str], LabelEncoder]:
 82→    """
 83→    Load dataset metadata from directory structure.
 84→    ...
 85→    """
 86→    data_path = Path(data_dir)
 87→
 88→    image_paths = []
 89→    breed_names = []
 90→
 91→    # Iterate through breed folders
 92→    for breed_folder in sorted(data_path.iterdir()):
 93→        if not breed_folder.is_dir():
 94→            continue
 95→
 96→        breed_name = breed_folder.name
 97→
 98→        # Get all images in this breed folder
 99→        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
100→            for img_path in breed_folder.glob(ext):
101→                image_paths.append(str(img_path))
102→                breed_names.append(breed_name)
103→
104→    # Encode breed names to integers
105→    label_encoder = LabelEncoder()
106→    labels = label_encoder.fit_transform(breed_names)
107→
108→    print(
109→        f"Loaded {len(image_paths)} images from {len(label_encoder.classes_)} breeds")
110→
111→    return image_paths, breed_names, label_encoder
```

**What happens:**
- Line 92: Iterate through 67 breed folders (Abyssinian, American Bobtail, ... York Chocolate)
- Lines 100-102: Append image path and breed name for each image
- Line 106: LabelEncoder.fit_transform() learns unique breed names
- Line 111: Return encoder with .classes_ = array of 67 unique breed names

#### Step 2: LabelEncoder Behavior

```python
# After fit_transform:
label_encoder.classes_ = array(['Abyssinian', 'American Bobtail', ..., 'York Chocolate'])
len(label_encoder.classes_) = 67
```

#### Step 3: num_classes Extraction

```python
316→    num_classes = len(label_encoder.classes_)  # <-- 67
```

### Verification: Breed Folder Count

```bash
$ ls -d /home/minh-ubs-k8s/multi-label-classification/data/images/*/ | wc -l
67
```

Total breed directories: **67** ✓

### Verification: Sample Breeds

```bash
$ ls /home/minh-ubs-k8s/multi-label-classification/data/images/ | sort
Abyssinian
American Bobtail
American Curl
American Shorthair
American Wirehair
...
(67 total)
```

### Conclusion: Dataset ✓ CORRECT

The num_classes calculation is **CORRECT = 67**. LabelEncoder properly learns all 67 breed classes from the directory structure.

---

## 2. Trainer num_classes Assignment (trainer.py)

### Location: Lines 38-73

```python
 38→    def __init__(self, config: Config):
 39→        """
 40→        Initialize trainer.
 41→        
 42→        Args:
 43→            config: Training configuration
 44→        """
 45→        self.config = config
 46→        
 47→        # Set seed
 48→        set_seed(config.training.seed, config.training.deterministic)
 49→        
 50→        # Device
 51→        self.device = get_device(config.training.device)
 52→        
 53→        # Prepare data
 54→        print("\n" + "="*80)
 55→        print("PREPARING DATA")
 56→        print("="*80)
 57→        
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

### Code Path: How num_classes Flows

#### Step 1: Call prepare_data_for_training() [Lines 59-60 or 63-64]

```python
# For k-fold:
self.folds, self.label_encoder, self.class_weights, self.num_classes = \
    prepare_data_for_training(config.data, config.augmentation, use_kfold=True)

# For single split:
self.dataloaders, self.label_encoder, self.class_weights, self.num_classes = \
    prepare_data_for_training(config.data, config.augmentation, use_kfold=False)
```

**Function returns tuple:** `(dataloaders/folds, label_encoder, class_weights, num_classes)`

From dataset.py line 340:
```python
return dataloaders, label_encoder, class_weights, num_classes
```

Where `num_classes = len(label_encoder.classes_) = 67`

#### Step 2: Store in self.num_classes [Lines 59-65]

```python
self.num_classes = 67  # Extracted from return value
```

#### Step 3: Update config [Line 68]

```python
self.config.model.num_classes = self.num_classes  # = 67
```

### Code Flow Verification

| Variable | Value | Source |
|----------|-------|--------|
| label_encoder.classes_ | 67 items | load_dataset_metadata() |
| num_classes (returned) | 67 | len(label_encoder.classes_) |
| self.num_classes | 67 | Assignment from return |
| self.config.model.num_classes | 67 | Updated from self.num_classes |

### Conclusion: Trainer ✓ CORRECT

The trainer correctly:
1. Receives num_classes = 67 from prepare_data_for_training()
2. Stores as self.num_classes = 67
3. Updates config: self.config.model.num_classes = 67

---

## 3. Metrics Computation - THE CRITICAL BUG

### Location: metrics.py Lines 112-135

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

### The Problem: Missing labels Parameter

#### What precision_recall_fscore_support() Returns

**Without labels parameter:**
```python
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0
)
```

Returns arrays of shape `(n_classes_observed,)` where n_classes_observed = number of unique classes in targets/preds.

**Example Scenario:**
- Total classes: 67 breeds
- Validation set contains images from: 60 unique breeds
- `precision` shape: (60,)
- `recall` shape: (60,)
- `f1` shape: (60,)
- `support` shape: (60,)

**With labels parameter:**
```python
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0,
    labels=np.arange(self.num_classes)  # <-- Force 67 classes
)
```

Returns arrays of shape `(67,)` with zeros for unobserved classes.

#### The Index Mismatch

| Line | Variable | Length | Status |
|------|----------|--------|--------|
| 122-124 | precision | 60 | Based on observed classes |
| 122-124 | recall | 60 | Based on observed classes |
| 122-124 | f1 | 60 | Based on observed classes |
| 122-124 | support | 60 | Based on observed classes |
| 24-33 (init) | self.class_names | 67 | Based on num_classes |
| 127 | Loop range | 0-66 | Iterates 67 times |
| 129 | precision[i] | **IndexError** | Tries to access precision[60-66] |

#### Error Trace

```python
# Loop iteration 60
per_class['class_60'] = {
    'precision': float(precision[60])  # ← IndexError: index 60 out of bounds
    ...
}
```

### Why This Bug Happens

1. **MetricsCalculator.__init__()** assumes all num_classes will be observed:
   ```python
   self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
   # Creates 67 class names
   ```

2. **compute_per_class_metrics()** calls sklearn function without forcing all classes:
   ```python
   precision_recall_fscore_support(
       targets, preds, average=None, zero_division=0
       # Missing: labels=np.arange(self.num_classes)
   )
   ```

3. **Result:** Metric arrays have length of observed classes, not num_classes

4. **Loop attempts to access** indices 0-66 of arrays with length < 67

### Code Evidence: MetricsCalculator.__init__

```python
 24→    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
 25→        """
 26→        Initialize metrics calculator.
 27→        
 28→        Args:
 29→            num_classes: Number of classes
 30→            class_names: Optional list of class names
 31→        """
 32→        self.num_classes = num_classes  # = 67
 33→        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
 34→        # Creates 67 class names even if not all observed
```

### Code Evidence: Usage in trainer.py

```python
373→        metrics_calc = MetricsCalculator(
374→            self.num_classes,  # = 67
375→            self.label_encoder.classes_.tolist()  # = 67 breed names
376→        )
```

So self.class_names = 67 breed names (Abyssinian, American Bobtail, ..., York Chocolate)

### When the Bug Manifests

**Scenario 1: Normal Training (Full Batch)**
- Validation set has all 67 breeds represented
- precision_recall_fscore_support returns arrays of length 67
- No error

**Scenario 2: Small Batch (BUG TRIGGERS)**
- Validation set has only 60 breeds represented
- precision_recall_fscore_support returns arrays of length 60
- Loop tries to access indices 60-66
- **IndexError: index out of bounds**

### Reproduction

```python
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# Simulated scenario
targets = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 10 samples, 10 classes
preds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Without labels parameter (current code)
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0
)
print(len(precision))  # Output: 10

# Now try to access as if there were 67 classes
class_names = [f"Class_{i}" for i in range(67)]
for i, class_name in enumerate(class_names):
    print(precision[i])  # IndexError when i >= 10!

# With labels parameter (fixed code)
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0,
    labels=np.arange(67)
)
print(len(precision))  # Output: 67 (with zeros for unobserved classes)
```

### Conclusion: Metrics ✗ CRITICAL BUG

The `compute_per_class_metrics()` method has a **critical bug**: it doesn't specify `labels` parameter in `precision_recall_fscore_support()`, causing IndexError when validation set lacks all 67 breed classes.

---

## Config num_classes Default Value

### Location: config.py Lines 65-74

```python
 65→class ModelConfig:
 66→    """Model configuration."""
 67→    # Model architecture
 68→    # resnet50, efficientnet_b0-b7, vit_b_16, convnext_tiny
 69→    name: str = "efficientnet_b0"
 70→    pretrained: bool = True
 71→    num_classes: int = 67
 72→
 73→    # Transfer learning
 74→    freeze_backbone: bool = False
```

The hardcoded default of 67 is **CORRECT**.

---

## Confusion Matrix Comparison

### Location: metrics.py Line 146

```python
146→        return confusion_matrix(targets, preds, labels=np.arange(self.num_classes))
```

**NOTE:** This implementation CORRECTLY includes `labels` parameter, ensuring all 67 classes are represented in the confusion matrix. This is the right approach that should be replicated in `compute_per_class_metrics()`.

---

## Summary Table: All Findings

| Component | File | Lines | Status | Issue |
|-----------|------|-------|--------|-------|
| Breed folder count | Data | `/data/images/` | 67 | None |
| load_dataset_metadata | dataset.py | 81-116 | ✓ OK | None |
| LabelEncoder.classes_ | dataset.py | 106-111 | ✓ OK | None |
| num_classes = len(label_encoder.classes_) | dataset.py | 307 | ✓ OK | None |
| prepare_data_for_training returns | dataset.py | 340 | ✓ OK | None |
| Trainer stores self.num_classes | trainer.py | 59-65 | ✓ OK | None |
| Trainer updates config | trainer.py | 68 | ✓ OK | None |
| Config hardcoded default | config.py | 71 | ✓ OK | None |
| **precision_recall_fscore_support call** | **metrics.py** | **122-124** | **✗ BUG** | **Missing labels param** |
| Iteration through class_names | metrics.py | 127-133 | ✓ Logic OK | Fails due to above bug |
| Confusion matrix | metrics.py | 146 | ✓ OK | Correctly uses labels |

---

## Recommended Fix

### File: `/home/minh-ubs-k8s/multi-label-classification/src/metrics.py`
### Method: `compute_per_class_metrics()`
### Lines to change: 122-124

**Current (BROKEN):**
```python
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0
)
```

**Fixed (CORRECT):**
```python
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0,
    labels=np.arange(self.num_classes)
)
```

This ensures all 67 classes are represented in the returned arrays, matching the length of `self.class_names`.

