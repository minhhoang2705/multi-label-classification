# Class Labels & Breed Mapping Scout Report

## Summary
Codebase uses scikit-learn's `LabelEncoder` for mapping breed names (strings) to integer labels (0-66). Total of 67 cat breed classes configured in `ModelConfig.num_classes`.

## Label Encoding

### File: `/home/minh-ubs-k8s/multi-label-classification/src/config.py`
**Num Classes:** 67 (hardcoded in ModelConfig)
```python
@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = 67  # 67 cat breeds
```

### File: `/home/minh-ubs-k8s/multi-label-classification/src/dataset.py`
**Label Encoding Pattern:**
- LabelEncoder from sklearn.preprocessing creates integer encodings
- Breeds loaded from directory structure in `data/images/[breed_name]/`
- Breed names fitted to create label mapping: `label_encoder.fit_transform(breed_names)`
- Encoded labels range from 0 to 66

**Key Functions:**

1. **load_dataset_metadata()** (line 81-116)
   - Iterates through breed folders in `data/images/`
   - Creates LabelEncoder and fits it with breed names
   - Returns: (image_paths, breed_names, label_encoder)
   ```python
   label_encoder = LabelEncoder()
   labels = label_encoder.fit_transform(breed_names)
   print(f"Loaded {len(image_paths)} images from {len(label_encoder.classes_)} breeds")
   ```

2. **CatBreedsDataset.get_breed_name()** (line 74-78)
   - Converts integer label back to breed name
   ```python
   def get_breed_name(self, label: int) -> str:
       if self.label_encoder:
           return self.label_encoder.inverse_transform([label])[0]
       return str(label)
   ```

3. **DatasetStatistics.compute_statistics()** (line 347-359)
   - Uses `label_encoder.inverse_transform(unique)` to get breed names from labels

## Breed Names (67 Total Classes)

Stored in sorted order from CSV metadata:

```
0: Abyssinian
1: American Bobtail
2: American Curl
3: American Shorthair
4: American Wirehair
5: Applehead Siamese
6: Balinese
7: Bengal
8: Birman
9: Bombay
10: British Shorthair
11: Burmese
12: Burmilla
13: Calico
14: Canadian Hairless
15: Chartreux
16: Chausie
17: Chinchilla
18: Cornish Rex
19: Cymric
20: Devon Rex
21: Dilute Calico
22: Dilute Tortoiseshell
23: Domestic Long Hair
24: Domestic Medium Hair
25: Domestic Short Hair
26: Egyptian Mau
27: Exotic Shorthair
28: Extra-Toes Cat - Hemingway Polydactyl
29: Havana
30: Himalayan
31: Japanese Bobtail
32: Javanese
33: Korat
34: LaPerm
35: Maine Coon
36: Manx
37: Munchkin
38: Nebelung
39: Norwegian Forest Cat
40: Ocicat
41: Oriental Long Hair
42: Oriental Short Hair
43: Oriental Tabby
44: Persian
45: Pixiebob
46: Ragamuffin
47: Ragdoll
48: Russian Blue
49: Scottish Fold
50: Selkirk Rex
51: Siamese
52: Siberian
53: Silver
54: Singapura
55: Snowshoe
56: Somali
57: Sphynx - Hairless Cat
58: Tabby
59: Tiger
60: Tonkinese
61: Torbie
62: Tortoiseshell
63: Turkish Angora
64: Turkish Van
65: Tuxedo
66: York Chocolate
```

## CSV Metadata Structure

**File:** `/home/minh-ubs-k8s/multi-label-classification/data/clean_metadata.csv`

**Columns:**
- `breed`: Breed name (source of labels)
- `file_path`: Path to image file
- `file_name`: Image filename
- `file_size_bytes`: Size in bytes
- `width`: Image width
- `height`: Image height
- `mode`: Color mode (RGB)
- `aspect_ratio`: Width/height ratio
- `file_size_mb`: Size in MB

## How to Get Breed Names From Labels

### In Training/Testing Code:

**Access via label_encoder object:**
```python
# From trainer.py (line 414)
class_names = self.label_encoder.classes_.tolist()

# From scripts/test.py (line 726)
class_names = label_encoder.classes_.tolist()
```

**Convert single label to breed name:**
```python
# From dataset.py (line 77)
breed_name = label_encoder.inverse_transform([label])[0]
```

**Get label from breed name:**
```python
label = label_encoder.transform(['Persian'])[0]  # Returns 44
```

## Data Flow

1. **Data Loading:** `load_dataset_metadata()` reads breed folders → LabelEncoder fits
2. **Training:** Images + integer labels fed to model → outputs predictions for classes 0-66
3. **Inference:** Model output (0-66) → `inverse_transform()` → breed name
4. **Validation:** Metrics computed with both integer labels and breed names via LabelEncoder

## Key Files

| File | Purpose |
|------|---------|
| `/home/minh-ubs-k8s/multi-label-classification/src/config.py` | Defines num_classes=67 |
| `/home/minh-ubs-k8s/multi-label-classification/src/dataset.py` | LabelEncoder creation and usage |
| `/home/minh-ubs-k8s/multi-label-classification/data/clean_metadata.csv` | Breed list source |
| `/home/minh-ubs-k8s/multi-label-classification/src/trainer.py` | Uses label_encoder.classes_.tolist() |
| `/home/minh-ubs-k8s/multi-label-classification/scripts/test.py` | Test script accessing class_names |

## Code Snippets

**LabelEncoder initialization:**
```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(breed_names)
```

**Get class names list:**
```python
class_names = label_encoder.classes_.tolist()  # Returns list of 67 breed names
```

**Reverse mapping (label → name):**
```python
breed_name = label_encoder.inverse_transform([label_int])[0]
```

**Forward mapping (name → label):**
```python
label_int = label_encoder.transform(['Persian'])[0]
```

---

**Date Generated:** 2025-12-16
**Codebase:** /home/minh-ubs-k8s/multi-label-classification

