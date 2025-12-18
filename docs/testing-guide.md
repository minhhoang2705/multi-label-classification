# Model Testing Guide

Comprehensive guide for testing trained cat breeds classification models and API endpoints.

## Quick Links

- **Model Testing:** Evaluate checkpoint performance
- **API Testing:** Test Phase 01 API endpoints
- **Integration Testing:** Full pipeline validation

---

## API Testing (Phase 01-05)

See detailed documentation:
- [`docs/api-phase05.md`](./api-phase05.md) - **Phase 05: Testing & Validation (40 tests, 89% coverage)**
- [`docs/api-phase01.md`](./api-phase01.md) - Phase 01: Core API & Model Loading
- [`docs/api-phase02.md`](./api-phase02.md) - Phase 02: Image Validation & Preprocessing
- [`docs/api-phase03.md`](./api-phase03.md) - Phase 03: Inference Pipeline
- [`docs/api-phase04.md`](./api-phase04.md) - Phase 04: Response Formatting & Metrics

### Run API Tests (Phase 05)

```bash
# Run all tests with coverage report
./scripts/run_api_tests.sh

# Or run specific test suites
pytest tests/api/ -v

# Run unit tests only (fast, no model)
pytest tests/api/test_image_service.py tests/api/test_inference_service.py -v

# Run integration tests (requires model)
pytest tests/api/test_health.py tests/api/test_predict.py tests/api/test_model.py -v

# Run specific test class
pytest tests/api/test_image_service.py::TestImageServiceValidation -v

# Run single test
pytest tests/api/test_health.py::TestHealthEndpoints::test_liveness -v

# Generate coverage report
pytest tests/api/ --cov=api --cov-report=term-missing --cov-report=html
```

### API Test Coverage (Phase 05)

**Total:** 40 comprehensive tests covering all phases
- **Code Coverage:** 89%
- **Health Endpoints:** 4 tests
- **Image Service Validation:** 15 tests
- **Inference Service:** 5 tests
- **Predict Endpoint:** 10 tests
- **Model Endpoints:** 6 tests

See [`docs/api-phase05.md`](./api-phase05.md) for detailed test breakdown by component.

### API Quick Start

```bash
# Start API
python -m uvicorn api.main:app --reload

# In another terminal, test endpoints
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/api/v1/model/info

# View interactive docs
# Browser: http://localhost:8000/docs

# Run test suite
./scripts/run_api_tests.sh
```

---

## Quick Start

### Test with Best Checkpoint (Easiest)

```bash
./test_model.sh [FOLD] [SPLIT] [MODEL_NAME]
```

**Examples:**
```bash
# Test fold 0 on validation set with ResNet50
./test_model.sh 0 val resnet50

# Test fold 1 on validation set
./test_model.sh 1 val resnet50

# Test all defaults (fold 0, val split, resnet50)
./test_model.sh
```

**Parameters:**
- `FOLD`: Fold number (0-4), default: 0
- `SPLIT`: Dataset split ('val' or 'test'), default: 'val'
- `MODEL_NAME`: Model architecture, default: 'resnet50'

---

## Manual Testing

For more control, use `scripts/test.py` directly:

```bash
python scripts/test.py \
    --checkpoint outputs/checkpoints/fold_0/best_model.pt \
    --model_name resnet50 \
    --split val \
    --fold 0 \
    --batch_size 64 \
    --num_workers 4 \
    --save_confusion_matrix \
    --save_per_class_metrics \
    --num_inference_samples 1000
```

---

## Command-Line Arguments

### Required Arguments
| Argument | Description | Example |
|----------|-------------|---------|
| `--checkpoint` | Path to checkpoint | `outputs/checkpoints/fold_0/best_model.pt` |

### Model Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `resnet50` | Model architecture |

### Data Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/images` | Path to images directory |
| `--batch_size` | `64` | Batch size for testing |
| `--num_workers` | `4` | Number of data loading workers |
| `--image_size` | `224` | Input image size |

### Test Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | `val` | Which split to evaluate ('val' or 'test') |
| `--fold` | `0` | Which fold to evaluate (0-4) |
| `--num_folds` | `5` | Total number of folds |

### Output Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `outputs/test_results` | Directory to save results |
| `--save_confusion_matrix` | `True` | Save confusion matrix plot |
| `--save_per_class_metrics` | `True` | Save per-class metrics JSON |
| `--save_roc_curves` | `True` | Save ROC curves plot |
| `--save_pr_curves` | `True` | Save Precision-Recall curves plot |
| `--num_inference_samples` | `1000` | Samples for speed test |

### Device
| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `cuda` | Device ('cuda', 'cpu', 'mps') |

---

## Output Files

All results are saved to `outputs/test_results/fold_{FOLD}/{SPLIT}/`:

### 1. `test_metrics.json`
Overall performance metrics:
```json
{
  "checkpoint": "outputs/checkpoints/fold_0/best_model.pt",
  "model_name": "resnet50",
  "split": "val",
  "fold": 0,
  "metrics": {
    "accuracy": 0.5496,
    "balanced_accuracy": 0.2310,
    "precision_macro": 0.3271,
    "recall_macro": 0.2310,
    "f1_macro": 0.2580,
    "precision_weighted": 0.5154,
    "recall_weighted": 0.5496,
    "f1_weighted": 0.5219,
    "top_1_accuracy": 0.5496,
    "top_3_accuracy": 0.7950,
    "top_5_accuracy": 0.8673,
    "roc_auc_micro": 0.9668,
    "roc_auc_macro": "NaN",
    "average_precision_micro": 0.5308,
    "average_precision_macro": 0.2256
  },
  "speed_metrics": {
    "samples_tested": 1024,
    "avg_time_per_sample_ms": 0.88,
    "std_time_per_sample_ms": 0.00,
    "min_time_per_sample_ms": 0.88,
    "max_time_per_sample_ms": 0.89,
    "median_time_per_sample_ms": 0.88,
    "throughput_samples_per_sec": 1141.8,
    "device": "cuda"
  }
}
```

### 2. `per_class_metrics.json`
Detailed metrics for each breed:
```json
{
  "Persian": {
    "precision": 0.7940,
    "recall": 0.7940,
    "f1": 0.7940,
    "support": 500
  },
  "Siamese": {
    "precision": 0.6361,
    "recall": 0.6361,
    "f1": 0.6361,
    "support": 350
  },
  ...
}
```

### 3. `class_performance.json`
Best and worst performing classes:
```json
{
  "worst_performing": [
    {"rank": 1, "class": "American Wirehair", "f1_score": 0.0000},
    {"rank": 2, "class": "Burmilla", "f1_score": 0.0000},
    ...
  ],
  "best_performing": [
    {"rank": 1, "class": "Sphynx - Hairless Cat", "f1_score": 0.8462},
    {"rank": 2, "class": "Persian", "f1_score": 0.7940},
    ...
  ]
}
```

### 4. `confusion_matrix.png`
Normalized confusion matrix heatmap (16x14 inches, 300 DPI)

### 5. `roc_curves.png`
ROC (Receiver Operating Characteristic) curves visualization:
- **Micro-average ROC curve:** Aggregated performance across all classes
- **Macro-average ROC curve:** Average performance treating all classes equally
- **Top 10 individual class curves:** Best performing breeds by AUC
- **Random baseline:** Diagonal line (AUC = 0.5)

### 6. `pr_curves.png`
Precision-Recall curves visualization:
- **Micro-average PR curve:** Aggregated precision-recall tradeoff
- **Macro-average score:** Average precision across all classes
- **Top 10 individual class curves:** Best performing breeds by Average Precision
- **Baseline:** Varies by class frequency in dataset

---

## Evaluation Metrics Explained

### Overall Metrics

**Accuracy**
- Percentage of correct predictions
- Range: 0-1 (higher is better)
- **Limitation:** Can be misleading for imbalanced datasets

**Balanced Accuracy**
- Average recall across all classes
- Range: 0-1 (higher is better)
- **Better for imbalanced data** - treats all breeds equally

### Macro Metrics
Calculated per-class, then averaged (all classes weighted equally):
- **Precision Macro:** Average precision across breeds
- **Recall Macro:** Average recall across breeds
- **F1 Macro:** Harmonic mean of precision & recall

### Weighted Metrics
Calculated per-class, weighted by class size:
- **Precision Weighted:** Weighted by breed frequency
- **Recall Weighted:** Weighted by breed frequency
- **F1 Weighted:** Weighted harmonic mean

### Top-K Accuracy
- **Top-1:** Standard accuracy (correct class is #1 prediction)
- **Top-3:** Success if correct class in top 3 predictions
- **Top-5:** Success if correct class in top 5 predictions

**Use case:** Useful when similar breeds are acceptable (e.g., Maine Coon vs Norwegian Forest)

### ROC AUC (Area Under ROC Curve)
Measures model's ability to distinguish between classes at various probability thresholds:
- **ROC AUC Micro:** Aggregates all classes into one curve (0-1, higher is better)
- **ROC AUC Macro:** Averages AUC across all classes (treats all breeds equally)
- **Range:** 0-1 (0.5 = random, 1.0 = perfect)
- **Interpretation:**
  - AUC > 0.9: Excellent discrimination
  - AUC 0.8-0.9: Good discrimination
  - AUC 0.7-0.8: Acceptable
  - AUC < 0.7: Poor discrimination

**Note:** Macro-average may be NaN when some classes have no positive samples

### Average Precision (AP)
Summarizes Precision-Recall curve as weighted mean of precisions:
- **AP Micro:** Aggregates precision-recall across all classes
- **AP Macro:** Averages AP across all classes
- **Range:** 0-1 (higher is better)
- **Better for imbalanced data** than ROC AUC
- **Interpretation:**
  - AP > 0.8: Excellent performance
  - AP 0.6-0.8: Good performance
  - AP 0.4-0.6: Moderate performance
  - AP < 0.4: Poor performance

### Inference Speed
- **Average time/sample:** Mean inference time per image
- **Throughput:** Samples processed per second
- **Device:** Hardware used (CUDA/CPU/MPS)

---

## Understanding Results

### Good Performance Indicators
✅ **Balanced Accuracy > 0.70** - Model works well across all breeds
✅ **Top-3 Accuracy > 0.85** - Model captures similar breeds
✅ **F1 Macro > 0.65** - Good precision-recall balance
✅ **No zero-F1 classes** - All breeds have some performance

### Warning Signs
⚠️ **Balanced Accuracy << Accuracy** - Model biased toward common breeds
⚠️ **Many zero-F1 classes** - Some breeds never predicted correctly
⚠️ **High precision, low recall** - Model is too conservative
⚠️ **Low precision, high recall** - Model predicts too liberally

### Example Interpretation

```
Overall Metrics:
  Accuracy:           0.5496 (54.96%)  ← Overall correctness
  Balanced Accuracy:  0.2310 (23.10%)  ← ⚠️ Much lower! Model struggles with rare breeds

Macro Metrics:
  F1 Score:           0.2580  ← ⚠️ Poor average performance per breed

Top-K Accuracy:
  Top-3:              0.7950 (79.50%)  ← ✅ Better! Model captures similar breeds
  Top-5:              0.8673 (86.73%)  ← ✅ Even better with more options

ROC & Precision-Recall:
  ROC AUC (Micro):    0.9668 (96.68%)  ← ✅ Excellent class separation overall
  ROC AUC (Macro):    NaN              ← ⚠️ Some classes have no samples
  AP (Micro):         0.5308 (53.08%)  ← Moderate precision-recall tradeoff
  AP (Macro):         0.2256 (22.56%)  ← ⚠️ Poor average precision per breed

Inference Speed:
  Throughput:         1141.8 samples/sec  ← ✅ Fast enough for real-time
```

**Analysis:** Model has excellent ROC AUC micro (96.68%) indicating strong class separation ability, but struggles with rare breeds (low balanced accuracy, AP macro). The gap between micro and macro metrics suggests performance is driven by common classes. Consider:
1. More data for underrepresented breeds
2. Class-balanced sampling during training
3. Data augmentation for rare classes
4. Reviewing ROC/PR curves to identify problem classes

---

## Interpreting ROC and PR Curve Plots

### ROC Curves (`roc_curves.png`)

**What it shows:**
- X-axis: False Positive Rate (FPR) - fraction of negatives incorrectly classified as positive
- Y-axis: True Positive Rate (TPR/Recall) - fraction of positives correctly classified
- Each curve represents one-vs-rest classification for a class

**How to read:**
- **Perfect classifier:** Curve hugs top-left corner (TPR=1, FPR=0)
- **Random classifier:** Diagonal line from (0,0) to (1,1) - AUC = 0.5
- **Better curves:** Higher and more to the left
- **AUC interpretation:** Area under the curve (0.5 = random, 1.0 = perfect)

**Multi-class specifics:**
- Plot shows top 10 classes by AUC (avoiding visual clutter)
- Micro-average (pink dashed): Overall performance across all samples
- Macro-average (navy dashed): Average performance per class

**Example insights:**
- High micro-AUC but low macro-AUC → Good on common breeds, poor on rare ones
- Wide variation in individual curves → Some breeds much easier to classify
- Curves near diagonal → Model struggles to distinguish these classes

### PR Curves (`pr_curves.png`)

**What it shows:**
- X-axis: Recall (fraction of positives found)
- Y-axis: Precision (fraction of predictions that are correct)
- Trade-off between precision and recall at various thresholds

**How to read:**
- **Perfect classifier:** Curve stays at top (Precision=1) across all recall values
- **Baseline:** Horizontal line at class frequency in dataset
- **Better curves:** Higher and more to the right
- **AP interpretation:** Weighted mean of precisions (0-1, higher is better)

**When to use PR over ROC:**
- **Imbalanced datasets** (like cat breeds) - PR curves more informative
- **When positive class is rare** - ROC can be overly optimistic
- **Cost of false positives matters** - PR highlights precision trade-offs

**Multi-class specifics:**
- Plot shows top 10 classes by Average Precision
- Each class has different baseline (varies by class frequency)
- Micro-average: Overall precision-recall trade-off
- Macro-average score shown in legend

**Example insights:**
- Low AP for rare breeds → Model predicts them infrequently
- Steep drops in curve → Poor calibration at certain thresholds
- Curves below baseline → Model worse than random for that class

---

## Comparing Multiple Checkpoints

Test multiple epochs or folds to find best checkpoint:

```bash
# Test all folds
for fold in {0..4}; do
  ./test_model.sh $fold val resnet50
done

# Test different epochs
python scripts/test.py \
  --checkpoint outputs/checkpoints/fold_0/checkpoint_epoch_35.pt \
  --fold 0 --split val --model_name resnet50

python scripts/test.py \
  --checkpoint outputs/checkpoints/fold_0/checkpoint_epoch_40.pt \
  --fold 0 --split val --model_name resnet50
```

Compare `test_metrics.json` files to select best checkpoint.

---

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python scripts/test.py --checkpoint <path> --batch_size 32
```

### Slow Inference
Check device:
```bash
python scripts/test.py --checkpoint <path> --device cuda
```

### Wrong Fold Data
Ensure `--fold` matches checkpoint:
```bash
# Checkpoint from fold 2 → use --fold 2
python scripts/test.py \
  --checkpoint outputs/checkpoints/fold_2/best_model.pt \
  --fold 2
```

### Missing Checkpoint
List available checkpoints:
```bash
find outputs/checkpoints -name "*.pt" | sort
```

---

## Best Practices

1. **Always test on validation set** during development
2. **Test on test set ONLY ONCE** at the very end
3. **Use same fold** for checkpoint and data split
4. **Compare balanced accuracy** for imbalanced datasets
5. **Check top-k accuracy** for similar breeds
6. **Analyze worst-performing classes** for improvement ideas
7. **Save all results** for reproducibility

---

## Next Steps

After testing:
1. **Review confusion matrix** for misclassification patterns
2. **Analyze ROC curves** to identify classes with poor discrimination
3. **Analyze PR curves** to understand precision-recall trade-offs
4. **Review worst-performing breeds** (from `class_performance.json`)
5. Consider targeted improvements:
   - More data for underrepresented breeds
   - Focused augmentation strategies
   - Architecture changes (e.g., attention mechanisms)
   - Hyperparameter tuning
   - Class-balanced sampling
   - Threshold optimization based on ROC/PR curves
6. Test ensemble models (combine multiple folds)
7. Deploy best checkpoint for inference
