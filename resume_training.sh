#!/bin/bash
# Resume training from latest checkpoint

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

# Find latest checkpoint
LATEST_CHECKPOINT=$(find outputs/checkpoints/fold_0 -name "checkpoint_epoch_*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ No checkpoint found in outputs/checkpoints/fold_0/"
    echo "Available checkpoints:"
    find outputs/checkpoints -name "*.pt" -type f 2>/dev/null
    exit 1
fi

echo "Found latest checkpoint: $LATEST_CHECKPOINT"
echo ""

# Extract epoch number
EPOCH=$(basename "$LATEST_CHECKPOINT" | grep -oP '\d+')
echo "Resuming from epoch: $EPOCH"
echo ""

# Resume training with same or modified parameters
echo "Resuming training..."
echo "You can modify batch_size, lr, etc. by adding arguments"
echo ""

python scripts/train.py --resume "$LATEST_CHECKPOINT" "$@"  --model_name resnet50 --fast_dev --batch_size 128 --lr 0.0002
