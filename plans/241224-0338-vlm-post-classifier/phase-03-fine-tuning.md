# Phase 3: VLM Fine-Tuning (Optional)

**Parent:** [plan.md](./plan.md) | **Depends:** [Phase 1](./phase-01-dataset-generation.md)
**Status:** Pending | **Priority:** Medium

## Overview

Fine-tune LLaVA-7B on cat breed dataset for improved domain-specific classification. Optional phase - zero-shot CLIP may be sufficient.

## Key Insights

- LLaVA-7B with LoRA: 8GB VRAM, 6-10 hours training
- Expected improvement: +3-5% over zero-shot on minority classes
- Fine-tuning useful if zero-shot underperforms on priority breeds
- LoRA rank 16-32 optimal for classification tasks

## Requirements

1. Generated VLM dataset from Phase 1
2. GPU with 16GB+ VRAM (RTX 3090, A100)
3. LLaVA or Pixtral fine-tuning framework

## When to Use This Phase

Skip if:
- Zero-shot CLIP achieves >80% on priority breeds
- Latency budget tight (<150ms)
- No GPU for training

Proceed if:
- Priority breeds still <60% F1 after Phase 2
- Explainability/reasoning needed
- Accuracy > latency priority

## Architecture

```
Phase 1 Dataset (data/vlm/)
    ↓
LoRA Fine-Tuning (r=32, freeze vision encoder)
    ↓
Checkpoint: outputs/vlm_checkpoints/llava-cat-breeds/
    ↓
Serve via vLLM or llama.cpp
```

## Related Files

- `data/vlm/train.jsonl` - Training data (Phase 1)
- `data/vlm/val.jsonl` - Validation data
- `scripts/train_vlm.py` - Fine-tuning script (new)

## Implementation Steps

### Step 1: Install Fine-Tuning Dependencies

```bash
# LLaVA fine-tuning requirements
pip install bitsandbytes peft accelerate
pip install deepspeed  # Optional for multi-GPU

# Or use vLLM for inference
pip install vllm
```

### Step 2: LoRA Training Configuration

```python
# configs/vlm_lora_config.py

from peft import LoraConfig

LORA_CONFIG = LoraConfig(
    r=32,                       # LoRA rank
    lora_alpha=64,              # Alpha = 2 * r
    target_modules=[
        "q_proj", "v_proj",     # Attention modules
        "k_proj", "o_proj",
        "gate_proj", "up_proj", # MLP modules
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

TRAINING_CONFIG = {
    "output_dir": "outputs/vlm_checkpoints/llava-cat-breeds",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,  # Effective batch = 32
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "fp16": True,
    "dataloader_num_workers": 4,
}
```

### Step 3: Fine-Tuning Script

```python
# scripts/train_vlm.py

import argparse
import json
import torch
from pathlib import Path
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model
from torch.utils.data import Dataset
from PIL import Image

class CatBreedVLMDataset(Dataset):
    """Dataset for VLM fine-tuning."""

    def __init__(self, jsonl_path: str, processor, image_base: str = "data"):
        self.processor = processor
        self.image_base = Path(image_base)
        self.samples = []

        with open(jsonl_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.image_base / sample["image"]
        image = Image.open(image_path).convert("RGB")

        # Build conversation
        conversation = sample["conversations"]
        user_msg = conversation[0]["value"].replace("<image>\n", "")
        assistant_msg = conversation[1]["value"]

        # Process with LLaVA processor
        prompt = f"USER: <image>\n{user_msg}\nASSISTANT: {assistant_msg}"

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True
        )

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--train_data", default="data/vlm/train.jsonl")
    parser.add_argument("--val_data", default="data/vlm/val.jsonl")
    parser.add_argument("--output_dir", default="outputs/vlm_checkpoints/llava-cat-breeds")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Apply LoRA
    from configs.vlm_lora_config import LORA_CONFIG
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # Load datasets
    train_dataset = CatBreedVLMDataset(args.train_data, processor)
    val_dataset = CatBreedVLMDataset(args.val_data, processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        report_to="mlflow",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    trainer.train()

    # Save final model
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
```

### Step 4: Inference with Fine-Tuned Model

```python
# api/services/vlm_service.py (add fine-tuned model support)

async def load_finetuned_model(
    self,
    checkpoint_path: str = "outputs/vlm_checkpoints/llava-cat-breeds",
    device: str = "auto"
) -> None:
    """Load fine-tuned LLaVA model."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel

    self._processor = AutoProcessor.from_pretrained(checkpoint_path)

    # Load base model + LoRA weights
    base_model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    self._model = PeftModel.from_pretrained(base_model, checkpoint_path)
    self._model.eval()
    self._is_loaded = True

def predict_finetuned(self, image, prompt: str) -> str:
    """Run inference with fine-tuned LLaVA."""
    inputs = self._processor(
        text=f"USER: <image>\n{prompt}\nASSISTANT:",
        images=image,
        return_tensors="pt"
    ).to(self._device)

    with torch.no_grad():
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    response = self._processor.decode(output_ids[0], skip_special_tokens=True)
    # Extract breed from response
    return self._parse_breed_response(response)
```

### Step 5: Evaluation Script

```python
# scripts/eval_vlm.py

import argparse
from sklearn.metrics import classification_report
from pathlib import Path
import json

def evaluate_vlm(model, test_jsonl: str, class_names: list):
    """Evaluate VLM on test set."""
    y_true, y_pred = [], []

    with open(test_jsonl) as f:
        for line in f:
            sample = json.loads(line)
            true_breed = sample["metadata"]["breed"]

            # Run inference
            pred_breed = model.predict(sample["image"])

            y_true.append(true_breed)
            y_pred.append(pred_breed)

    # Report
    print(classification_report(
        y_true, y_pred,
        labels=class_names,
        zero_division=0
    ))

    # Per-priority breed analysis
    priority_breeds = [
        "American Wirehair", "Burmilla", "Canadian Hairless",
        "Chinchilla", "Cymric", "Oriental Long Hair", "York Chocolate"
    ]
    priority_correct = sum(1 for t, p in zip(y_true, y_pred)
                          if t in priority_breeds and t == p)
    priority_total = sum(1 for t in y_true if t in priority_breeds)

    print(f"\nPriority Breeds: {priority_correct}/{priority_total} "
          f"({100*priority_correct/max(1,priority_total):.1f}%)")
```

## Todo

- [ ] Create `configs/vlm_lora_config.py`
- [ ] Create `scripts/train_vlm.py`
- [ ] Create `scripts/eval_vlm.py`
- [ ] Run training (3 epochs, ~6-10 hours)
- [ ] Evaluate on test set
- [ ] Compare vs zero-shot CLIP
- [ ] Integrate into inference service if beneficial

## Success Criteria

- [ ] Training completes without OOM
- [ ] Val loss decreases over epochs
- [ ] Priority breed F1 > 50% (vs 0% baseline)
- [ ] +3-5% overall accuracy vs zero-shot CLIP

## Hardware Requirements

| Config | GPU | VRAM | Time |
|--------|-----|------|------|
| LoRA r=16 | RTX 3090 | 16GB | 8-12h |
| LoRA r=32 | A100 40GB | 24GB | 4-6h |
| QLoRA 4-bit | RTX 4090 | 20GB | 10-14h |

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| OOM during training | High | Reduce batch size, use gradient checkpointing |
| Overfitting | Medium | Early stopping, validation monitoring |
| No improvement | Medium | Skip phase, use zero-shot CLIP |
| Long training time | Low | Use QLoRA, smaller model |

## Checkpoints

```
outputs/vlm_checkpoints/llava-cat-breeds/
├── adapter_config.json
├── adapter_model.safetensors
├── preprocessor_config.json
└── training_args.bin
```

---
**Estimated Effort:** 2-3 days | **Dependencies:** Phase 1
