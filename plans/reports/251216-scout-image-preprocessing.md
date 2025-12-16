# Image Preprocessing Patterns - Cat Breeds Classification

## Project Overview
Cat breeds multi-label classification with 67 breeds using ImageNet-based normalization and comprehensive augmentation pipeline.

## Key Files Located
- `/home/minh-ubs-k8s/multi-label-classification/src/augmentations.py` - Transform pipelines
- `/home/minh-ubs-k8s/multi-label-classification/src/dataset.py` - Data loading & stratified splits
- `/home/minh-ubs-k8s/multi-label-classification/src/config.py` - Configuration & constants

---

## Normalization Constants (ImageNet Stats)

**Source:** `src/config.py` - `AugmentationConfig` class (lines 57-61)

```python
normalize_mean: List[float] = field(
    default_factory=lambda: [0.485, 0.456, 0.406])
normalize_std: List[float] = field(
    default_factory=lambda: [0.229, 0.224, 0.225])
```

**Values:**
- `mean = [0.485, 0.456, 0.406]` (RGB channels)
- `std = [0.229, 0.224, 0.225]` (RGB channels)

Standard ImageNet normalization for transfer learning models.

---

## Validation Transforms Pipeline

**File:** `src/augmentations.py` lines 171-189

**Function:** `get_val_transforms(config: AugmentationConfig, image_size: int = 224) -> T.Compose`

### Pipeline Order (Inference/Validation):
1. **Resize** → `(224, 224)` (default, configurable)
2. **ToTensor** → Convert PIL Image to tensor [0,1]
3. **Normalize** → Apply ImageNet statistics
   - mean: [0.485, 0.456, 0.406]
   - std: [0.229, 0.224, 0.225]

### Code:
```python
def get_val_transforms(config: AugmentationConfig, image_size: int = 224) -> T.Compose:
    """
    Get validation/test transforms.

    Args:
        config: Augmentation configuration
        image_size: Target image size

    Returns:
        Composed transforms
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std
        )
    ])
```

---

## Training Transforms Pipeline

**File:** `src/augmentations.py` lines 100-168

**Function:** `get_train_transforms(config: AugmentationConfig, image_size: int = 224) -> T.Compose`

### Pipeline Order (Training):
1. **Resize** → `(224, 224)` (default, configurable)
2. **RandomHorizontalFlip** → p=0.5 (if enabled)
3. **RandomVerticalFlip** → p=0.5 (if enabled, default=False)
4. **RandomRotation** → degrees=15 (if enabled, default=True)
5. **AutoAugment** → ImageNet policy (if enabled, default=False)
6. **RandAugment** → num_ops=2, magnitude=9 (if enabled, default=False)
7. **ColorJitter** → brightness/contrast/saturation/hue (if enabled)
   - brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
8. **ToTensor** → Convert PIL Image to tensor
9. **RandomErasing** → p=0.25, scale=(0.02, 0.33) (if enabled, default=True)
10. **Normalize** → ImageNet statistics

### Configuration (src/config.py AugmentationConfig):
```python
@dataclass
class AugmentationConfig:
    # Basic augmentations
    horizontal_flip: bool = True          # Enabled
    vertical_flip: bool = False           # Disabled
    rotation_degrees: int = 15            # 15 degrees

    # Color augmentations
    color_jitter: bool = True             # Enabled
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1

    # Advanced augmentations
    random_erasing: bool = True           # Enabled
    random_erasing_prob: float = 0.25

    # AutoAugment / RandAugment
    use_autoaugment: bool = False         # Disabled
    use_randaugment: bool = False         # Disabled
    randaugment_n: int = 2
    randaugment_m: int = 9

    # Normalization (ImageNet stats)
    normalize_mean: List[float] = [0.485, 0.456, 0.406]
    normalize_std: List[float] = [0.229, 0.224, 0.225]
```

---

## Data Loading & Image Preprocessing

**File:** `src/dataset.py`

### Image Loading (lines 47-72):
- **Format:** Converts all images to RGB (even grayscale)
- **Error Handling:** Falls back to 224x224 black image on load failure
- **Default Size:** 224x224 (matches model input requirements)

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
    img_path = self.image_paths[idx]
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        image = Image.new('RGB', (224, 224), color='black')
    
    if self.transform:
        image = self.transform(image)
    
    label = self.labels[idx]
    return image, label
```

### Image Size Configuration (src/config.py):
```python
@dataclass
class DataConfig:
    data_dir: str = "data/images"
    csv_path: Optional[str] = "data/clean_metadata.csv"
    image_size: int = 224           # Input size for model
    num_workers: int = 4
    pin_memory: bool = True
```

---

## Test-Time Augmentation (TTA) Pipeline

**File:** `src/augmentations.py` lines 192-238

**Function:** `get_test_time_augmentation_transforms(config, image_size=224, n_augmentations=5)`

### TTA Variants (5 augmentations by default):
1. **Original** - Standard validation transforms
2. **Horizontal Flip** - Deterministic flip (p=1.0)
3. **Rotation -10°** - Fixed left rotation
4. **Rotation +10°** - Fixed right rotation
5. **Center Crop** - Resize to 110% then center crop

Each TTA variant includes full normalization pipeline.

---

## Data Pipeline Integration

**File:** `src/dataset.py` lines 233-283

### Function: `get_dataloaders(...)`

**Creates three dataloaders with conditional transforms:**
- **Train Split:** Uses `get_train_transforms()` (aggressive augmentation)
- **Val/Test Splits:** Uses `get_val_transforms()` (minimal transforms)

**Batch Sizes:**
- Train: `batch_size` (default=32)
- Val/Test: `batch_size * 2` (default=64) - larger batches for inference

```python
def get_dataloaders(
    data_config: DataConfig,
    aug_config: AugmentationConfig,
    splits: Dict[str, Tuple[List[str], np.ndarray]],
    label_encoder: LabelEncoder,
    use_train_augmentation: bool = True
) -> Dict[str, DataLoader]:
    
    for split_name, (paths, labels) in splits.items():
        if split_name == 'train' and use_train_augmentation:
            transform = get_train_transforms(
                aug_config, data_config.image_size)
        else:
            transform = get_val_transforms(aug_config, data_config.image_size)
        
        dataset = CatBreedsDataset(
            image_paths=paths,
            labels=labels.tolist(),
            transform=transform,
            label_encoder=label_encoder
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=data_config.batch_size if split_name == 'train' else data_config.batch_size * 2,
            shuffle=(split_name == 'train'),
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            drop_last=(split_name == 'train')
        )
```

---

## Denormalization Utility

**File:** `src/augmentations.py` lines 241-258

For visualization of normalized tensors:

```python
class Denormalize:
    """Denormalize images for visualization."""

    def __init__(self, mean: list, std: list):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean
```

**Usage Example:**
```python
denorm = Denormalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
original_img = denorm(normalized_tensor)
```

---

## Mixup & CutMix Augmentation

**File:** `src/augmentations.py` lines 15-97

### MixUp Class:
- Default alpha=0.2
- Blends images: `mixed = lambda * img_a + (1-lambda) * img_b`
- Lambda sampled from Beta(alpha, alpha)

### CutMix Class:
- Default alpha=1.0
- Random box cut and replace
- Adjusts lambda based on pixel ratio affected

Configuration flags (src/config.py):
```python
use_mixup: bool = False              # Disabled by default
mixup_alpha: float = 0.2
use_cutmix: bool = False             # Disabled by default
cutmix_alpha: float = 1.0
```

---

## Data Splitting Strategy

**File:** `src/dataset.py` lines 119-169

### Stratified Split:
- Train: 80% (maintains class distribution)
- Val: 10% (maintains class distribution)
- Test: 10% (maintains class distribution)
- Random seed: 42

### K-Fold Cross-Validation:
- Default: 5 folds
- Stratified StratifiedKFold
- Random seed: 42

Both ensure class balance across splits for imbalanced dataset (67 breeds).

---

## Summary

| Component | Value | File |
|-----------|-------|------|
| Input Size | 224×224 | config.py |
| Normalization Mean | [0.485, 0.456, 0.406] | config.py |
| Normalization Std | [0.229, 0.224, 0.225] | config.py |
| Val Pipeline | Resize → ToTensor → Normalize | augmentations.py |
| Train Pipeline | Resize → Flips → Rotation → Jitter → ToTensor → Erasing → Normalize | augmentations.py |
| Default Augmentations | H-flip(0.5), Rotation(15°), ColorJitter, RandomErasing(0.25) | config.py |
| TTA Variants | 5 (Original, H-flip, Rot±10°, CenterCrop) | augmentations.py |
| Batch Sizes | Train: 32, Val/Test: 64 | dataset.py |
| Data Split | Train:Val:Test = 80:10:10 | dataset.py |

---

## Unresolved Questions
None - all preprocessing patterns extracted with exact values.
