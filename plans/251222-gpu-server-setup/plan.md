# GPU Server Setup Automation Plan

**Date:** 2025-12-22
**Goal:** Single-command setup from fresh Ubuntu to training in <15 min
**Approach:** Shell-based with idempotent phases + state tracking

## Problem Statement

Current manual 6-step setup (venv, deps, data download, config) takes ~30+ min and is error-prone. Need automated, resumable solution.

## Architecture Overview

```
setup_training_server.sh (master orchestrator)
    |
    +-- check_system()     -> verify Ubuntu, NVIDIA drivers
    +-- install_uv()       -> fast Python env manager
    +-- setup_python()     -> create venv with Python 3.12
    +-- download_data()    -> scripts/download_dataset.sh (Kaggle)
    +-- install_deps()     -> uv pip install requirements
    +-- validate_env()     -> scripts/validate_env.py (GPU test)
    +-- test_training()    -> quick 1-epoch smoke test

State: ~/.cache/ml-setup/cat-breeds.state (phase completion tracking)
```

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package manager | **uv** | 10-100x faster than pip, resumable |
| State location | `~/.cache/ml-setup/` | User-portable, no sudo needed |
| Dataset source | Kaggle API | Official, versioned, resumable |
| Validation | GPU matmul + spot check | Catches driver/data issues early |

## Phases

| Phase | File | Est. Lines | Key Deliverable |
|-------|------|------------|-----------------|
| 01 | phase-01-master-setup-script.md | ~250 | `setup_training_server.sh` |
| 02 | phase-02-dataset-download.md | ~150 | `scripts/download_dataset.sh` |
| 03 | phase-03-environment-validation.md | ~180 | `scripts/validate_env.py` |
| 04 | phase-04-documentation-update.md | ~50 | README.md updates |

## Success Criteria

1. Fresh Ubuntu 22.04 + NVIDIA drivers -> training in <15 min
2. Idempotent: re-run skips completed phases
3. Resume: interrupted download continues from checkpoint
4. Validation: catches GPU/data issues before training starts

## Constraints

- Ubuntu 20.04/22.04 with existing NVIDIA drivers
- No sudo in main flow (except driver install if missing)
- Preserve existing scripts (start_mlflow.sh, test_model.sh)
- Follow existing patterns: venv activation, checkpoint discovery

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Kaggle rate limits | Retry logic + exponential backoff |
| NVIDIA driver mismatch | Clear error message + version check |
| Disk space exhaustion | Pre-check: need ~15GB free |
| Network interruption | State file + resumable downloads |

## Implementation Order

Execute phases 01-04 sequentially. Each phase self-contained but builds on prior.
