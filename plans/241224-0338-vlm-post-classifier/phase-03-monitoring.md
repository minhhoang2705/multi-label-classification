# Phase 03: Monitoring & Production

**Parent:** [plan.md](./plan.md)
**Depends on:** [phase-02-disagreement-strategy.md](./phase-02-disagreement-strategy.md)
**Status:** Pending | **Priority:** Medium | **Effort:** 0.5 day

## Overview

Add monitoring for VLM verification pipeline: track agreement rates, latency, and errors.

## Metrics to Track

| Metric | Purpose |
|--------|---------|
| Agreement rate | % predictions where CNN == VLM |
| Disagreement rate | % predictions where CNN != VLM |
| VLM error rate | % failed VLM calls |
| CNN accuracy on verified | Accuracy when VLM agrees |
| CNN accuracy on uncertain | Accuracy when VLM disagrees |
| VLM latency p50/p95/p99 | API response times |

## Implementation

### Step 1: Add Metrics Collection

**File:** `api/services/metrics_service.py`

```python
"""Metrics collection for hybrid inference pipeline."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Aggregated prediction metrics."""
    total_predictions: int = 0
    verified_count: int = 0
    uncertain_count: int = 0
    cnn_only_count: int = 0
    error_count: int = 0

    # Latency tracking (in ms)
    cnn_latencies: List[float] = field(default_factory=list)
    vlm_latencies: List[float] = field(default_factory=list)

    # Per-breed disagreement tracking
    breed_disagreements: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, result):
        """Record a prediction result."""
        self.total_predictions += 1

        if result.status == "verified":
            self.verified_count += 1
        elif result.status == "uncertain":
            self.uncertain_count += 1
            self.breed_disagreements[result.cnn_prediction] += 1
        elif result.status == "cnn_only":
            self.cnn_only_count += 1
        else:
            self.error_count += 1

        self.cnn_latencies.append(result.cnn_time_ms)
        if result.vlm_time_ms:
            self.vlm_latencies.append(result.vlm_time_ms)

    def summary(self) -> dict:
        """Get metrics summary."""
        import statistics

        return {
            "total_predictions": self.total_predictions,
            "agreement_rate": self.verified_count / max(1, self.total_predictions),
            "disagreement_rate": self.uncertain_count / max(1, self.total_predictions),
            "error_rate": self.error_count / max(1, self.total_predictions),
            "cnn_latency_p50": statistics.median(self.cnn_latencies) if self.cnn_latencies else 0,
            "vlm_latency_p50": statistics.median(self.vlm_latencies) if self.vlm_latencies else 0,
            "vlm_latency_p95": (
                statistics.quantiles(self.vlm_latencies, n=20)[18]
                if len(self.vlm_latencies) >= 20 else
                max(self.vlm_latencies) if self.vlm_latencies else 0
            ),
            "top_disagreement_breeds": dict(
                sorted(self.breed_disagreements.items(), key=lambda x: -x[1])[:10]
            )
        }


# Global metrics instance
_metrics = PredictionMetrics()


def get_metrics() -> PredictionMetrics:
    return _metrics


def reset_metrics():
    global _metrics
    _metrics = PredictionMetrics()
```

### Step 2: Add Metrics Endpoint

**File:** `api/routers/metrics.py`

```python
"""Metrics endpoint for monitoring."""

from fastapi import APIRouter

from ..services.metrics_service import get_metrics

router = APIRouter()


@router.get("/metrics")
async def get_prediction_metrics():
    """Get prediction pipeline metrics."""
    metrics = get_metrics()
    return metrics.summary()


@router.post("/metrics/reset")
async def reset_prediction_metrics():
    """Reset prediction metrics (admin only)."""
    from ..services.metrics_service import reset_metrics
    reset_metrics()
    return {"status": "reset"}
```

### Step 3: Integrate Metrics Collection

Update `hybrid_inference_service.py`:

```python
from .metrics_service import get_metrics

# At end of predict() method:
get_metrics().record(result)
```

## Logging Configuration

**File:** `api/logging_config.py`

```python
import logging

# Log disagreements for analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create disagreement logger
disagreement_logger = logging.getLogger('disagreements')
disagreement_handler = logging.FileHandler('logs/disagreements.jsonl')
disagreement_logger.addHandler(disagreement_handler)
```

Log disagreements:
```python
if result.status == "uncertain":
    disagreement_logger.info(json.dumps({
        "timestamp": time.time(),
        "cnn_prediction": result.cnn_prediction,
        "cnn_confidence": result.cnn_confidence,
        "vlm_prediction": result.vlm_prediction,
        "vlm_reasoning": result.vlm_reasoning
    }))
```

## Dashboard Queries

For analyzing logged data:

```python
# Agreement rate over time
df.groupby(df['timestamp'].dt.hour)['status'].value_counts(normalize=True)

# Breeds with highest disagreement
df[df['status'] == 'uncertain']['cnn_prediction'].value_counts().head(10)

# VLM accuracy when disagreeing
df[df['status'] == 'uncertain'].apply(
    lambda x: x['ground_truth'] == x['vlm_prediction']
).mean()
```

## Success Criteria

- [ ] `/metrics` endpoint returns valid data
- [ ] Disagreements logged to file
- [ ] Can identify high-disagreement breeds
- [ ] Latency percentiles available

## Production Checklist

- [ ] Set `ZAI_API_KEY` environment variable
- [ ] Create `logs/` directory
- [ ] Configure log rotation
- [ ] Set up alerting on error rate >5%
- [ ] Monitor VLM API costs

---
**Complete:** Ready for production deployment
