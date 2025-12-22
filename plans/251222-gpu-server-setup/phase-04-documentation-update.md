# Phase 04: Documentation Update

## Context

Update README.md with single-command setup instructions and add quick-start section for remote GPU server deployment.

## Overview

Modify existing `README.md` to add:
- Quick Start section for automated setup
- Prerequisites section with GPU requirements
- Troubleshooting section for common issues
- Keep existing content (EDA, manual setup, etc.)

## Architecture

```
README.md (updated)
    |
    +-- Quick Start (NEW)
    |   |-- Prerequisites
    |   |-- One-command setup
    |   |-- Verification
    |
    +-- Remote GPU Server Setup (NEW)
    |   |-- Fresh server setup
    |   |-- Resume interrupted setup
    |   |-- Validation
    |
    +-- Existing Sections
    |   |-- Dataset info
    |   |-- Manual setup (reference)
    |   |-- EDA notebook contents
    |
    +-- Troubleshooting (NEW)
        |-- Common issues
        |-- Recovery steps
```

## Implementation Steps

### Step 1: Quick Start Section (Insert After Title)

Add after the main title, before "Dataset" section:

```markdown
## Quick Start (Automated Setup)

For fresh Ubuntu 20.04/22.04 servers with NVIDIA GPU drivers installed:

```bash
# Clone repository
git clone https://github.com/your-repo/multi-label-classification.git
cd multi-label-classification

# Run automated setup (installs everything, downloads data, validates)
./setup_training_server.sh

# Start training
source .venv/bin/activate
python scripts/train.py --fast_dev
```

Setup completes in ~10-15 minutes and includes:
- Python 3.12 environment via uv
- All dependencies (PyTorch, timm, etc.)
- Cat breeds dataset (~67K images)
- GPU and data validation

### Prerequisites

| Requirement | Details |
|-------------|---------|
| OS | Ubuntu 20.04 or 22.04 |
| GPU | NVIDIA with drivers installed |
| Disk | 15GB free space |
| Network | For dataset download (~4GB) |

**Check GPU drivers:**
```bash
nvidia-smi  # Should show GPU info and CUDA version
```
```

### Step 2: Remote GPU Server Section

Add new section after Quick Start:

```markdown
## Remote GPU Server Setup

### Fresh Server Deployment

```bash
# 1. SSH into your GPU server
ssh user@gpu-server

# 2. Install NVIDIA drivers (if not present)
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot

# 3. After reboot, clone and setup
git clone https://github.com/your-repo/multi-label-classification.git
cd multi-label-classification
./setup_training_server.sh
```

### Resume Interrupted Setup

Setup is idempotent and tracks state. If interrupted:

```bash
./setup_training_server.sh  # Continues from last completed phase
```

To force re-run all phases:

```bash
./setup_training_server.sh --force
```

### Validate Environment

```bash
source .venv/bin/activate
python scripts/validate_env.py
```

### Start Training

```bash
source .venv/bin/activate

# Quick test (2 epochs, 2 folds)
python scripts/train.py --fast_dev

# Full training
python scripts/train.py --model_name efficientnet_b3 --num_epochs 50
```

### Monitor with MLflow

```bash
./start_mlflow.sh
# Access at http://localhost:5000
```
```

### Step 3: Troubleshooting Section

Add at the end of README:

```markdown
## Troubleshooting

### Setup Issues

**"NVIDIA drivers not installed"**
```bash
sudo apt update && sudo apt install nvidia-driver-535
sudo reboot
```

**"Kaggle credentials not found"**
1. Go to https://www.kaggle.com/settings
2. Click "Create New Token" under API
3. Move file: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

**"uv command not found"**
```bash
export PATH="$HOME/.local/bin:$PATH"
# Add to ~/.bashrc for persistence
```

**Resume interrupted download**
```bash
./scripts/download_dataset.sh  # Automatically resumes
```

### Training Issues

**"CUDA out of memory"**
```bash
python scripts/train.py --batch_size 16 --image_size 224
```

**Reset setup state**
```bash
./setup_training_server.sh --reset
./setup_training_server.sh
```

### Validation Issues

**Run validation independently**
```bash
source .venv/bin/activate
python scripts/validate_env.py --skip-benchmark
```
```

### Step 4: Update Existing Setup Section

Modify the existing "Setup" section to reference automated setup first:

```markdown
## Setup

### Recommended: Automated Setup

See [Quick Start](#quick-start-automated-setup) above for one-command setup.

### Manual Setup (Alternative)

If you prefer manual installation:

#### 1. Install Dependencies
...
```

## Todo List

- [ ] Add "Quick Start" section after title
- [ ] Add "Remote GPU Server Setup" section
- [ ] Add "Troubleshooting" section at end
- [ ] Update existing "Setup" section to reference automated setup
- [ ] Update table of contents if present
- [ ] Test all code blocks are correctly formatted
- [ ] Verify all script paths are correct

## Success Criteria

1. New user can follow README to setup in <15 min
2. Troubleshooting covers top 5 common issues
3. Existing documentation preserved
4. Code blocks are copy-pasteable
5. Links to relevant files work

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing docs | Medium | Only add, don't remove content |
| Outdated paths | Low | Use relative paths, test |
| Missing edge cases | Low | Cover most common issues |

## Sample Updated README Structure

```
# Cat Breeds Dataset - Multi-Label Classification

## Quick Start (Automated Setup)     <-- NEW
  - Prerequisites
  - One-command setup
  - What's included

## Remote GPU Server Setup           <-- NEW
  - Fresh server deployment
  - Resume interrupted setup
  - Validation
  - Start training
  - Monitor with MLflow

## Dataset                           <-- EXISTING
  - Statistics
  - Structure

## Setup                             <-- MODIFIED
  - Recommended: Automated Setup (link to Quick Start)
  - Manual Setup (Alternative)

## Files                             <-- EXISTING

## Notebook Contents                 <-- EXISTING

## Key Findings                      <-- EXISTING

## Recommendations                   <-- EXISTING

## Troubleshooting                   <-- NEW
  - Setup issues
  - Training issues
  - Validation issues
```
