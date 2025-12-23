#!/bin/bash
# Test model with best checkpoint

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠️  Warning: No virtual environment found (.venv or venv)"
fi

# Default parameters
FOLD=${1:-0}
SPLIT=${2:-val}
MODEL_NAME=${3:-convnext_base}

# Find best checkpoint for fold
BEST_CHECKPOINT="outputs/checkpoints/fold_${FOLD}/best_model.pt"

if [ ! -f "$BEST_CHECKPOINT" ]; then
    echo "❌ Best checkpoint not found: $BEST_CHECKPOINT"
    echo ""
    echo "Available checkpoints in fold_${FOLD}:"
    ls -lh outputs/checkpoints/fold_${FOLD}/*.pt 2>/dev/null | tail -5
    exit 1
fi

echo "==========================================================================================================="
echo "TESTING MODEL"
echo "==========================================================================================================="
echo ""
echo "Checkpoint: $BEST_CHECKPOINT"
echo "Fold: $FOLD"
echo "Split: $SPLIT"
echo "Model: $MODEL_NAME"
echo ""
echo "==========================================================================================================="
echo ""

# Run test script
python scripts/test.py \
    --checkpoint "$BEST_CHECKPOINT" \
    --model_name "$MODEL_NAME" \
    --split "$SPLIT" \
    --fold "$FOLD" \
    --batch_size 128 \
    --num_workers 4 \
    --save_confusion_matrix \
    --save_per_class_metrics \
    --num_inference_samples 1000

echo ""
echo "==========================================================================================================="
echo "Testing complete! Results saved to: outputs/test_results/fold_${FOLD}/${SPLIT}/"
echo "==========================================================================================================="
