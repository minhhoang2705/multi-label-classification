# Phase 01: Master Setup Script

## Context

Master orchestrator script that coordinates all setup phases with idempotent execution and state tracking.

## Overview

Create `setup_training_server.sh` - a single entry point that:
- Checks system requirements (Ubuntu, NVIDIA drivers)
- Installs uv package manager
- Creates Python virtual environment
- Downloads dataset via Kaggle API
- Installs dependencies
- Validates environment (GPU + data)
- Runs smoke test

## Architecture

```
setup_training_server.sh
    |
    +-- State Management
    |   |-- STATE_DIR=~/.cache/ml-setup
    |   |-- STATE_FILE=$STATE_DIR/cat-breeds.state
    |   |-- is_phase_done() / mark_phase_done()
    |
    +-- Phase Functions (idempotent)
    |   |-- check_system
    |   |-- install_uv
    |   |-- setup_python
    |   |-- download_data
    |   |-- install_deps
    |   |-- validate_env
    |   |-- test_training
    |
    +-- CLI Interface
        |-- --force: re-run all phases
        |-- --skip-validation: skip GPU/data validation
        |-- --skip-test: skip smoke test
        |-- --help: show usage
```

## Implementation Steps

### Step 1: Script Header & Configuration

```bash
#!/usr/bin/env bash
set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
STATE_DIR="${HOME}/.cache/ml-setup"
STATE_FILE="${STATE_DIR}/cat-breeds.state"
PYTHON_VERSION="3.12"
VENV_DIR="${PROJECT_DIR}/.venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timing
START_TIME=$(date +%s)
```

### Step 2: State Management Functions

```bash
init_state() {
    mkdir -p "$STATE_DIR"
    touch "$STATE_FILE"
}

is_phase_done() {
    local phase="$1"
    grep -q "^${phase}=done$" "$STATE_FILE" 2>/dev/null
}

mark_phase_done() {
    local phase="$1"
    echo "${phase}=done" >> "$STATE_FILE"
}

reset_state() {
    rm -f "$STATE_FILE"
    init_state
}
```

### Step 3: Logging Functions

```bash
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}
```

### Step 4: System Check Phase

```bash
check_system() {
    print_header "Phase 1/7: System Check"

    if is_phase_done "check_system" && [[ "$FORCE" != "true" ]]; then
        log_success "System check already completed (skipping)"
        return 0
    fi

    # Check Ubuntu version
    if [[ ! -f /etc/os-release ]]; then
        log_error "Cannot detect OS. Expected Ubuntu 20.04/22.04"
        exit 1
    fi

    source /etc/os-release
    if [[ "$ID" != "ubuntu" ]]; then
        log_error "Expected Ubuntu, got: $ID"
        exit 1
    fi

    log_info "Ubuntu version: $VERSION_ID"

    # Check NVIDIA drivers
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA drivers not installed. Please install first:"
        echo "  sudo apt update && sudo apt install nvidia-driver-535"
        exit 1
    fi

    # Capture GPU info
    log_info "GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

    # Check CUDA availability
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || echo "unknown")
    log_info "CUDA version: $CUDA_VERSION"

    # Check disk space (need ~15GB)
    AVAILABLE_GB=$(df -BG "$PROJECT_DIR" | tail -1 | awk '{print $4}' | tr -d 'G')
    if [[ "$AVAILABLE_GB" -lt 15 ]]; then
        log_error "Insufficient disk space. Need 15GB, have ${AVAILABLE_GB}GB"
        exit 1
    fi
    log_info "Disk space: ${AVAILABLE_GB}GB available"

    mark_phase_done "check_system"
    log_success "System check passed"
}
```

### Step 5: UV Installation Phase

```bash
install_uv() {
    print_header "Phase 2/7: Install uv Package Manager"

    if is_phase_done "install_uv" && [[ "$FORCE" != "true" ]]; then
        log_success "uv already installed (skipping)"
        return 0
    fi

    if command -v uv &> /dev/null; then
        log_info "uv already installed: $(uv --version)"
    else
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add to PATH for this session
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Verify installation
    if ! command -v uv &> /dev/null; then
        log_error "uv installation failed"
        exit 1
    fi

    log_info "uv version: $(uv --version)"
    mark_phase_done "install_uv"
    log_success "uv installed"
}
```

### Step 6: Python Environment Setup

```bash
setup_python() {
    print_header "Phase 3/7: Setup Python Environment"

    if is_phase_done "setup_python" && [[ "$FORCE" != "true" ]]; then
        log_success "Python environment already configured (skipping)"
        return 0
    fi

    cd "$PROJECT_DIR"

    # Install Python if needed
    log_info "Ensuring Python ${PYTHON_VERSION}..."
    uv python install "$PYTHON_VERSION" 2>/dev/null || true

    # Create virtual environment
    if [[ ! -d "$VENV_DIR" ]]; then
        log_info "Creating virtual environment..."
        uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    else
        log_info "Virtual environment exists"
    fi

    # Verify
    source "$VENV_DIR/bin/activate"
    log_info "Python: $(python --version)"
    log_info "Location: $(which python)"

    mark_phase_done "setup_python"
    log_success "Python environment ready"
}
```

### Step 7: Data Download Phase

```bash
download_data() {
    print_header "Phase 4/7: Download Dataset"

    if is_phase_done "download_data" && [[ "$FORCE" != "true" ]]; then
        log_success "Dataset already downloaded (skipping)"
        return 0
    fi

    # Call dedicated download script
    if [[ -x "${PROJECT_DIR}/scripts/download_dataset.sh" ]]; then
        "${PROJECT_DIR}/scripts/download_dataset.sh"
    else
        log_error "Download script not found: scripts/download_dataset.sh"
        exit 1
    fi

    mark_phase_done "download_data"
    log_success "Dataset downloaded"
}
```

### Step 8: Dependencies Installation

```bash
install_deps() {
    print_header "Phase 5/7: Install Dependencies"

    if is_phase_done "install_deps" && [[ "$FORCE" != "true" ]]; then
        log_success "Dependencies already installed (skipping)"
        return 0
    fi

    cd "$PROJECT_DIR"
    source "$VENV_DIR/bin/activate"

    log_info "Installing dependencies with uv (this may take 1-2 minutes)..."

    # Use uv pip for fast installation
    uv pip install -r requirements.txt

    # Verify key packages
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
    python -c "import timm; print(f'timm: {timm.__version__}')"

    mark_phase_done "install_deps"
    log_success "Dependencies installed"
}
```

### Step 9: Environment Validation Phase

```bash
validate_env() {
    print_header "Phase 6/7: Validate Environment"

    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        log_warn "Skipping validation (--skip-validation)"
        return 0
    fi

    if is_phase_done "validate_env" && [[ "$FORCE" != "true" ]]; then
        log_success "Environment already validated (skipping)"
        return 0
    fi

    cd "$PROJECT_DIR"
    source "$VENV_DIR/bin/activate"

    # Call validation script
    if [[ -f "${PROJECT_DIR}/scripts/validate_env.py" ]]; then
        python "${PROJECT_DIR}/scripts/validate_env.py"
    else
        log_error "Validation script not found: scripts/validate_env.py"
        exit 1
    fi

    mark_phase_done "validate_env"
    log_success "Environment validated"
}
```

### Step 10: Smoke Test Phase

```bash
test_training() {
    print_header "Phase 7/7: Training Smoke Test"

    if [[ "$SKIP_TEST" == "true" ]]; then
        log_warn "Skipping smoke test (--skip-test)"
        return 0
    fi

    if is_phase_done "test_training" && [[ "$FORCE" != "true" ]]; then
        log_success "Smoke test already passed (skipping)"
        return 0
    fi

    cd "$PROJECT_DIR"
    source "$VENV_DIR/bin/activate"

    log_info "Running 1-epoch smoke test..."

    # Quick training test with minimal config
    python scripts/train.py \
        --fast_dev \
        --num_epochs 1 \
        --num_folds 1 \
        --batch_size 16 \
        --use_mlflow false \
        2>&1 | head -50

    mark_phase_done "test_training"
    log_success "Smoke test passed"
}
```

### Step 11: Main Function & CLI

```bash
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated GPU server setup for cat breeds classification training.

OPTIONS:
    --force             Force re-run all phases (ignore state)
    --skip-validation   Skip GPU/data validation phase
    --skip-test         Skip training smoke test
    --reset             Reset state and start fresh
    -h, --help          Show this help message

EXAMPLES:
    $0                  # Normal setup (idempotent)
    $0 --force          # Force complete re-setup
    $0 --skip-test      # Skip final smoke test

STATE FILE:
    $STATE_FILE
EOF
}

main() {
    # Parse arguments
    FORCE="false"
    SKIP_VALIDATION="false"
    SKIP_TEST="false"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --force) FORCE="true"; shift ;;
            --skip-validation) SKIP_VALIDATION="true"; shift ;;
            --skip-test) SKIP_TEST="true"; shift ;;
            --reset) reset_state; log_info "State reset"; exit 0 ;;
            -h|--help) show_usage; exit 0 ;;
            *) log_error "Unknown option: $1"; show_usage; exit 1 ;;
        esac
    done

    # Initialize
    init_state

    echo ""
    echo "=========================================="
    echo "  Cat Breeds Training Server Setup"
    echo "=========================================="
    echo "Project: $PROJECT_DIR"
    echo "State: $STATE_FILE"
    echo ""

    # Run phases
    check_system
    install_uv
    setup_python
    download_data
    install_deps
    validate_env
    test_training

    # Summary
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "=========================================="
    echo "  Setup Complete!"
    echo "=========================================="
    echo "Duration: ${DURATION} seconds"
    echo ""
    echo "To start training:"
    echo "  source .venv/bin/activate"
    echo "  python scripts/train.py --fast_dev"
    echo ""
    echo "To start MLflow UI:"
    echo "  ./start_mlflow.sh"
    echo ""
}

main "$@"
```

## Todo List

- [ ] Create `setup_training_server.sh` with script header and config
- [ ] Implement state management functions (init, check, mark, reset)
- [ ] Implement logging functions with colors
- [ ] Implement `check_system` phase (Ubuntu, NVIDIA, disk space)
- [ ] Implement `install_uv` phase with PATH handling
- [ ] Implement `setup_python` phase with uv venv
- [ ] Implement `download_data` phase (calls Phase 02 script)
- [ ] Implement `install_deps` phase with uv pip
- [ ] Implement `validate_env` phase (calls Phase 03 script)
- [ ] Implement `test_training` phase (1-epoch smoke test)
- [ ] Implement CLI argument parsing (--force, --skip-*, --help)
- [ ] Add final summary with timing and next steps
- [ ] Make script executable: `chmod +x setup_training_server.sh`
- [ ] Test idempotency: run twice, second should skip all

## Success Criteria

1. Script runs without sudo in normal flow
2. Idempotent: second run skips all phases
3. `--force` flag re-runs all phases
4. Clear error messages with recovery suggestions
5. Total execution <15 min on fresh server
6. State persists across terminal sessions

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| uv install fails | High | Fallback to pip, clear error message |
| State file corruption | Medium | Simple format, easy to reset |
| Phase fails mid-way | Medium | Resume from last successful phase |
| PATH not updated | Medium | Export PATH in script, instructions for shell reload |
