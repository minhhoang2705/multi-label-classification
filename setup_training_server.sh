#!/usr/bin/env bash
# ============================================================================
# Cat Breeds Classification - Training Server Setup Script
# ============================================================================
# This script automates the complete setup of a remote GPU training server.
#
# Usage:
#   ./setup_training_server.sh              # Normal setup
#   ./setup_training_server.sh --force      # Force re-run all phases
#   ./setup_training_server.sh --skip-test  # Skip smoke test
#   ./setup_training_server.sh --help       # Show usage
#
# Phases:
#   1. System check (Ubuntu, NVIDIA drivers)
#   2. Install uv package manager
#   3. Setup Python virtual environment
#   4. Download dataset (Kaggle API)
#   5. Install dependencies
#   6. Validate environment (GPU + data)
#   7. Run smoke test
# ============================================================================

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
STATE_DIR="${HOME}/.cache/ml-setup"
STATE_FILE="${STATE_DIR}/cat-breeds.state"
LOG_FILE="${STATE_DIR}/setup.log"
PYTHON_VERSION="3.12"
VENV_DIR="${PROJECT_DIR}/.venv"
DATA_DIR="${PROJECT_DIR}/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Timing
START_TIME=$(date +%s)

# ============================================================================
# CLI Flags
# ============================================================================

FORCE=false
SKIP_VALIDATION=false
SKIP_TEST=false

for arg in "$@"; do
    case $arg in
        --force)
            FORCE=true
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            ;;
        --skip-test)
            SKIP_TEST=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force             Re-run all phases (ignore state)"
            echo "  --skip-validation   Skip GPU and data validation"
            echo "  --skip-test         Skip smoke test"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Phases:"
            echo "  1. System check (Ubuntu, NVIDIA drivers)"
            echo "  2. Install uv package manager"
            echo "  3. Setup Python virtual environment"
            echo "  4. Download dataset (Kaggle API)"
            echo "  5. Install dependencies"
            echo "  6. Validate environment (GPU + data)"
            echo "  7. Run smoke test"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# State Management Functions
# ============================================================================

init_state() {
    mkdir -p "$STATE_DIR"
    touch "$STATE_FILE"
    touch "$LOG_FILE"
}

is_phase_done() {
    local phase="$1"
    grep -q "^${phase}=done$" "$STATE_FILE" 2>/dev/null
}

mark_phase_done() {
    local phase="$1"
    # Atomic write: create temp file, then replace
    {
        grep -v "^${phase}=" "$STATE_FILE" 2>/dev/null || true
        echo "${phase}=done"
    } > "${STATE_FILE}.tmp"
    mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

reset_state() {
    rm -f "$STATE_FILE"
    init_state
    log_info "State reset. All phases will re-run."
}

# ============================================================================
# Logging Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[⚠]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1" | tee -a "$LOG_FILE" >&2
}

log_phase() {
    echo -e "${CYAN}[PHASE]${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
}

print_separator() {
    echo "----------------------------------------" | tee -a "$LOG_FILE"
}

# ============================================================================
# Phase Functions
# ============================================================================

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
        log_warn "Expected Ubuntu, detected: $ID $VERSION_ID"
        log_info "Continuing anyway (may work on Debian-based distros)"
    else
        log_info "Ubuntu version: $VERSION_ID"
    fi

    # Check NVIDIA drivers
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA drivers not installed. Please install first:"
        echo "  sudo apt update"
        echo "  sudo apt install nvidia-driver-535  # or newer"
        exit 1
    fi

    # Capture GPU info
    log_info "GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
        echo "  $line" | tee -a "$LOG_FILE"
    done

    # Check CUDA availability
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || echo "unknown")
    log_info "CUDA version: $CUDA_VERSION"

    # Check disk space
    AVAIL_DISK_GB=$(df "$PROJECT_DIR" | awk 'NR==2 {printf "%.1f", $4/1024/1024}')
    log_info "Available disk space: ${AVAIL_DISK_GB} GB"

    if (( $(echo "$AVAIL_DISK_GB < 50" | bc -l) )); then
        log_warn "Low disk space (${AVAIL_DISK_GB} GB). Recommended: 100+ GB"
    fi

    # Check RAM
    TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    log_info "Total RAM: ${TOTAL_RAM_GB} GB"

    if [[ "$TOTAL_RAM_GB" -lt 16 ]]; then
        log_warn "Low RAM (${TOTAL_RAM_GB} GB). Recommended: 16+ GB for training"
    fi

    log_success "System check passed"
    mark_phase_done "check_system"
}

install_uv() {
    print_header "Phase 2/7: Install uv Package Manager"

    if is_phase_done "install_uv" && [[ "$FORCE" != "true" ]]; then
        log_success "uv already installed (skipping)"
        return 0
    fi

    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version | awk '{print $2}')
        log_info "uv already installed: v$UV_VERSION"
        mark_phase_done "install_uv"
        return 0
    fi

    log_info "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version | awk '{print $2}')
        log_success "uv installed: v$UV_VERSION"
    else
        log_error "uv installation failed"
        exit 1
    fi

    mark_phase_done "install_uv"
}

setup_python() {
    print_header "Phase 3/7: Setup Python Virtual Environment"

    if is_phase_done "setup_python" && [[ "$FORCE" != "true" ]]; then
        log_success "Python environment already set up (skipping)"
        return 0
    fi

    # Ensure uv is in PATH
    export PATH="$HOME/.cargo/bin:$PATH"

    log_info "Creating virtual environment with Python $PYTHON_VERSION..."

    if [[ -d "$VENV_DIR" ]]; then
        log_warn "Virtual environment already exists at $VENV_DIR"
        log_info "Using existing environment"
    else
        # Use uv to create venv (much faster than python -m venv)
        uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
        log_success "Virtual environment created"
    fi

    # Verify venv
    if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
        log_error "Virtual environment creation failed"
        exit 1
    fi

    log_success "Python environment ready"
    mark_phase_done "setup_python"
}

download_data() {
    print_header "Phase 4/7: Download Dataset"

    if is_phase_done "download_data" && [[ "$FORCE" != "true" ]]; then
        log_success "Dataset already downloaded (skipping)"
        return 0
    fi

    log_info "Checking for dataset at $DATA_DIR/images..."

    # Check if data already exists
    if [[ -d "$DATA_DIR/images" ]]; then
        NUM_IMAGES=$(find "$DATA_DIR/images" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
        if [[ "$NUM_IMAGES" -gt 1000 ]]; then
            log_info "Dataset already present ($NUM_IMAGES images)"
            mark_phase_done "download_data"
            return 0
        fi
    fi

    log_info "Dataset not found. Downloading..."
    log_warn "This requires Kaggle API credentials"

    # Delegate to download script
    if [[ ! -f "$PROJECT_DIR/scripts/download_dataset.sh" ]]; then
        log_error "Download script not found: scripts/download_dataset.sh"
        log_info "Please manually download the Cat Breeds dataset to $DATA_DIR/"
        log_info "Skipping automatic download..."
        return 0
    fi

    bash "$PROJECT_DIR/scripts/download_dataset.sh" || {
        log_warn "Automatic download failed"
        log_info "Please manually download dataset to: $DATA_DIR/images/"
        log_info "You can continue setup and download later"
        return 0
    }

    log_success "Dataset downloaded"
    mark_phase_done "download_data"
}

install_deps() {
    print_header "Phase 5/7: Install Dependencies"

    if is_phase_done "install_deps" && [[ "$FORCE" != "true" ]]; then
        log_success "Dependencies already installed (skipping)"
        return 0
    fi

    log_info "Installing Python dependencies with uv..."

    # Ensure uv and venv are available
    export PATH="$HOME/.cargo/bin:$PATH"
    source "$VENV_DIR/bin/activate"

    # Use uv pip for 10-100x faster installs
    log_info "This may take 5-10 minutes (downloading PyTorch, CUDA libs, etc.)"
    uv pip install -r "$PROJECT_DIR/requirements.txt"

    # Verify key dependencies
    log_info "Verifying installations..."
    python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
    python -c "import torchvision; print(f'  Torchvision: {torchvision.__version__}')"
    python -c "import timm; print(f'  TIMM: {timm.__version__}')"
    python -c "import fastapi; print(f'  FastAPI: {fastapi.__version__}')"

    log_success "Dependencies installed"
    mark_phase_done "install_deps"
}

validate_env() {
    print_header "Phase 6/7: Validate Environment"

    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        log_warn "Skipping validation (--skip-validation flag)"
        mark_phase_done "validate_env"
        return 0
    fi

    if is_phase_done "validate_env" && [[ "$FORCE" != "true" ]]; then
        log_success "Environment already validated (skipping)"
        return 0
    fi

    source "$VENV_DIR/bin/activate"

    # Delegate to validation script
    if [[ ! -f "$PROJECT_DIR/scripts/validate_env.py" ]]; then
        log_warn "Validation script not found: scripts/validate_env.py"
        log_info "Skipping automated validation"
        mark_phase_done "validate_env"
        return 0
    fi

    log_info "Running environment validation..."
    python "$PROJECT_DIR/scripts/validate_env.py" || {
        log_error "Environment validation failed"
        exit 1
    }

    log_success "Environment validated"
    mark_phase_done "validate_env"
}

test_training() {
    print_header "Phase 7/7: Smoke Test"

    if [[ "$SKIP_TEST" == "true" ]]; then
        log_warn "Skipping smoke test (--skip-test flag)"
        mark_phase_done "test_training"
        return 0
    fi

    if is_phase_done "test_training" && [[ "$FORCE" != "true" ]]; then
        log_success "Smoke test already completed (skipping)"
        return 0
    fi

    source "$VENV_DIR/bin/activate"

    log_info "Running 1-epoch training test..."
    log_info "This verifies GPU training works end-to-end"

    # Run minimal training (1 epoch, 1 fold, small batch)
    python scripts/train.py \
        --num_epochs 1 \
        --num_folds 1 \
        --batch_size 16 \
        --num_workers 2 \
        --model_name resnet50 \
        --fast_dev || {
            log_error "Smoke test failed"
            exit 1
        }

    log_success "Smoke test passed"
    mark_phase_done "test_training"
}

# ============================================================================
# Cleanup Handler
# ============================================================================

cleanup() {
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        echo ""
        log_error "Setup failed with exit code: $exit_code"
        log_info "Check logs at: $LOG_FILE"
        log_info "To retry, run: $0"
        log_info "To reset state: rm -rf $STATE_DIR"
    fi
}

trap cleanup EXIT

# ============================================================================
# Main Execution
# ============================================================================

main() {
    print_header "Cat Breeds Training Server Setup"

    log_info "Project directory: $PROJECT_DIR"
    log_info "State directory: $STATE_DIR"
    log_info "Log file: $LOG_FILE"

    # Initialize state
    init_state

    # Reset state if --force flag
    if [[ "$FORCE" == "true" ]]; then
        reset_state
    fi

    # Execute phases sequentially
    check_system
    install_uv
    setup_python
    download_data
    install_deps
    validate_env
    test_training

    # Success summary
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))

    print_header "Setup Complete!"

    log_success "All phases completed successfully"
    log_info "Total time: ${MINUTES}m ${SECONDS}s"
    echo ""

    echo "Next steps:"
    echo "  1. Activate environment:  source .venv/bin/activate"
    echo "  2. Start training:        python scripts/train.py --fast_dev"
    echo "  3. Start MLflow UI:       ./start_mlflow.sh"
    echo "  4. Test model:            ./test_model.sh"
    echo "  5. Start API:             uvicorn api.main:app --host 0.0.0.0 --port 8000"
    echo ""
    echo "For more options: python scripts/train.py --help"
}

# Run main
main
