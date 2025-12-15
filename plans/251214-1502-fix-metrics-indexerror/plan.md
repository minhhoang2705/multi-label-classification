# Fix IndexError in metrics.py

**Date:** 2024-12-14
**Status:** Ready for Implementation
**Priority:** High (blocks training)

## Problem Summary

`IndexError` in `compute_per_class_metrics()` at lines 122-135 when validation set doesn't contain all 67 classes.

**Root Cause:** Missing `labels` parameter in `precision_recall_fscore_support()` call causes sklearn to return arrays sized by unique classes in data rather than all expected classes.

## Current Code (Line 122-124)

```python
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0
)
```

## Required Fix

```python
precision, recall, f1, support = precision_recall_fscore_support(
    targets, preds, average=None, zero_division=0,
    labels=np.arange(self.num_classes)
)
```

## Pre-Implementation Checklist

| Item | Status | Notes |
|------|--------|-------|
| numpy imported | OK | Line 6: `import numpy as np` |
| `self.num_classes` exists | OK | Set in `__init__` line 32 |
| Similar pattern in codebase | OK | Line 103, 146 already use `labels=np.arange(self.num_classes)` |

## Implementation Steps

### Step 1: Apply Fix
- File: `/home/minh-ubs-k8s/multi-label-classification/src/metrics.py`
- Location: Line 122-124
- Change: Add `labels=np.arange(self.num_classes)` parameter

### Step 2: Verify Fix
Run training to confirm no IndexError:
```bash
python scripts/train.py --max_epochs 1 --fast_dev_run
```

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Breaking other metrics | Low | Same pattern used elsewhere in file |
| Performance impact | None | No additional computation |
| Side effects | None | Only affects array sizing |

## Success Criteria

1. No `IndexError` during validation
2. Per-class metrics computed for all 67 classes
3. Classes not present in validation batch show 0.0 metrics (expected behavior)

## Files Modified

- `src/metrics.py` (1 line change)

## Estimated Time

5 minutes implementation + verification
