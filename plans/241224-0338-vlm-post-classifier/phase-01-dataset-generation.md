# Phase 1: VLM Fine-Tuning Dataset Generation

**Parent:** [plan.md](./plan.md) | **Status:** Pending | **Priority:** High

## Overview

Generate VLM fine-tuning dataset from existing 127K cat breed images with breed-specific conversation templates.

## Key Insights

- Existing dataset: 67 breeds, ~127K images, varying class sizes
- VLM requires JSONL format with `image` + `conversations` structure
- LLaVA expects 336x336, CLIP uses 224x224 (flexible)
- Class imbalance must be addressed via stratified sampling

## Requirements

1. Convert existing images to VLM-compatible JSONL format
2. Generate varied conversation templates per breed
3. Stratified sampling for train/val/test splits
4. Handle 7 priority classes with oversampling

## Architecture

```
data/images/{breed}/*.jpg
    ↓
scripts/generate_vlm_dataset.py
    ↓
data/vlm/
├── train.jsonl      (80%)
├── val.jsonl        (10%)
├── test.jsonl       (10%)
└── priority_breeds.jsonl  (7 minority classes, separate)
```

## Related Files

- `data/images/` - Source images
- `data/vlm_sample_fireworks.jsonl` - Reference format
- `scripts/generate_vlm_dataset.py` - Existing stub

## Implementation Steps

### Step 1: Create Conversation Templates

```python
# src/vlm/dataset_generator.py

QUESTION_TEMPLATES = [
    "What cat breed is this?",
    "Can you identify this cat's breed?",
    "What type of cat is shown in this image?",
    "Identify the breed of this cat.",
    "What breed does this cat belong to?",
]

ANSWER_TEMPLATES = [
    "This is a {breed}.",
    "The cat in the image is a {breed}.",
    "This cat belongs to the {breed} breed.",
    "{breed}",
    "I can identify this as a {breed} cat.",
]

PRIORITY_BREEDS = [
    "American Wirehair", "Burmilla", "Canadian Hairless",
    "Chinchilla", "Cymric", "Oriental Long Hair", "York Chocolate"
]
```

### Step 2: Dataset Generator Class

```python
# src/vlm/dataset_generator.py

import json
import random
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split

class VLMDatasetGenerator:
    def __init__(
        self,
        image_dir: str = "data/images",
        output_dir: str = "data/vlm",
        priority_oversample: int = 5  # 5x oversample for priority breeds
    ):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.priority_oversample = priority_oversample
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_conversation(self, breed: str) -> Dict:
        """Generate single Q&A pair for breed."""
        question = random.choice(QUESTION_TEMPLATES)
        answer = random.choice(ANSWER_TEMPLATES).format(breed=breed)
        return {
            "from": "human", "value": f"<image>\n{question}"
        }, {
            "from": "gpt", "value": answer
        }

    def create_sample(self, image_path: Path, breed: str) -> Dict:
        """Create JSONL sample entry."""
        q, a = self.generate_conversation(breed)
        return {
            "id": f"{breed}_{image_path.stem}",
            "image": str(image_path.relative_to(self.image_dir.parent)),
            "conversations": [q, a],
            "metadata": {"breed": breed}
        }

    def generate_dataset(self, max_per_class: int = 2000) -> Dict[str, List]:
        """Generate full dataset with stratified splits."""
        all_samples = []
        breed_counts = {}

        for breed_dir in sorted(self.image_dir.iterdir()):
            if not breed_dir.is_dir():
                continue

            breed = breed_dir.name
            images = list(breed_dir.glob("*.jpg"))[:max_per_class]
            breed_counts[breed] = len(images)

            # Oversample priority breeds
            if breed in PRIORITY_BREEDS:
                images = images * self.priority_oversample

            for img in images:
                all_samples.append(self.create_sample(img, breed))

        # Stratified split
        labels = [s["metadata"]["breed"] for s in all_samples]
        train, temp = train_test_split(all_samples, test_size=0.2, stratify=labels)
        temp_labels = [s["metadata"]["breed"] for s in temp]
        val, test = train_test_split(temp, test_size=0.5, stratify=temp_labels)

        return {"train": train, "val": val, "test": test}

    def save_jsonl(self, samples: List[Dict], filename: str):
        """Save samples to JSONL file."""
        with open(self.output_dir / filename, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

    def run(self):
        """Generate and save all datasets."""
        splits = self.generate_dataset()
        for split_name, samples in splits.items():
            self.save_jsonl(samples, f"{split_name}.jsonl")
            print(f"{split_name}: {len(samples)} samples")
```

### Step 3: CLI Script

```python
# scripts/generate_vlm_dataset.py (update existing)

import argparse
import sys
sys.path.insert(0, ".")
from src.vlm.dataset_generator import VLMDatasetGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="data/images")
    parser.add_argument("--output_dir", default="data/vlm")
    parser.add_argument("--max_per_class", type=int, default=2000)
    parser.add_argument("--priority_oversample", type=int, default=5)
    args = parser.parse_args()

    generator = VLMDatasetGenerator(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        priority_oversample=args.priority_oversample
    )
    generator.run()

if __name__ == "__main__":
    main()
```

### Step 4: Fireworks/LLaVA Format Converter

```python
# src/vlm/format_converter.py

def to_fireworks_format(sample: Dict) -> Dict:
    """Convert to Fireworks AI fine-tuning format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["conversations"][0]["value"].replace("<image>\n", "")},
                    {"type": "image_url", "image_url": {"url": f"file://{sample['image']}"}}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["conversations"][1]["value"]}]
            }
        ]
    }

def to_llava_format(sample: Dict) -> Dict:
    """Convert to LLaVA fine-tuning format."""
    return {
        "id": sample["id"],
        "image": sample["image"],
        "conversations": sample["conversations"]
    }
```

## Todo

- [ ] Create `src/vlm/` module directory
- [ ] Implement `dataset_generator.py`
- [ ] Implement `format_converter.py`
- [ ] Update `scripts/generate_vlm_dataset.py`
- [ ] Generate train/val/test splits
- [ ] Verify JSONL format compatibility
- [ ] Generate priority breeds dataset

## Success Criteria

- [ ] JSONL files generated: train.jsonl, val.jsonl, test.jsonl
- [ ] Train set: ~100K samples (with oversampling)
- [ ] Val/Test: ~12K samples each
- [ ] Priority breeds oversampled 5x
- [ ] Format validated against existing `vlm_sample_fireworks.jsonl`

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Disk space for copied images | Low | Use relative paths, no image copying |
| Imbalanced splits | Medium | Stratified sampling + priority oversampling |
| Format incompatibility | Low | Validate against existing samples |

## Output Files

```
data/vlm/
├── train.jsonl           # ~100K samples
├── val.jsonl             # ~12K samples
├── test.jsonl            # ~12K samples
├── train_fireworks.jsonl # Fireworks format
└── train_llava.jsonl     # LLaVA format
```

---
**Estimated Effort:** 4-6 hours | **Dependencies:** None
