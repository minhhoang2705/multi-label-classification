#!/usr/bin/env bash
# ============================================================================
# Cat Breeds Dataset Download Script
# ============================================================================
# Downloads Cat Breeds Dataset from Kaggle with retry logic and verification.
#
# Usage:
#   ./scripts/download_dataset.sh
#
# Requirements:
#   - Kaggle API credentials at ~/.kaggle/kaggle.json
#   - kaggle CLI tool (installed via pip install kaggle)
# ============================================================================

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data"
IMAGES_DIR="${DATA_DIR}/images"
STATE_DIR="${HOME}/.cache/ml-setup"
META_FILE="${STATE_DIR}/dataset.meta"

# Kaggle dataset info
KAGGLE_DATASET="ma7555/cat-breeds-dataset"
EXPECTED_BREEDS=67
MIN_IMAGES=50000

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# Logging Functions
# ============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1" >&2; }

print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

# ============================================================================
# Credential Management
# ============================================================================

check_kaggle_credentials() {
    local kaggle_json="$HOME/.kaggle/kaggle.json"

    if [[ ! -f "$kaggle_json" ]]; then
        log_error "Kaggle credentials not found: $kaggle_json"
        echo ""
        echo "To set up Kaggle API:"
        echo "  1. Go to https://www.kaggle.com/settings"
        echo "  2. Click 'Create New Token' under API section"
        echo "  3. Move downloaded kaggle.json:"
        echo "     mkdir -p ~/.kaggle"
        echo "     mv ~/Downloads/kaggle.json ~/.kaggle/"
        echo "  4. Set permissions:"
        echo "     chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        exit 1
    fi

    # Ensure correct permissions
    chmod 600 "$kaggle_json"

    # Check if kaggle CLI is installed
    if ! command -v kaggle &>/dev/null; then
        log_error "Kaggle CLI not installed"
        echo ""
        echo "Install with:"
        echo "  pip install kaggle"
        echo ""
        exit 1
    fi

    # Validate credentials with test request
    log_info "Validating Kaggle credentials..."
    if ! kaggle datasets list --max-size 1 &>/dev/null; then
        log_error "Kaggle credentials invalid. Please regenerate token."
        exit 1
    fi

    log_success "Kaggle credentials valid"
}

# ============================================================================
# Download Functions
# ============================================================================

download_with_retry() {
    local max_attempts=3
    local attempt=1
    local wait_time=5

    while [[ $attempt -le $max_attempts ]]; do
        log_info "Download attempt $attempt/$max_attempts..."

        if kaggle datasets download -d "$KAGGLE_DATASET" -p "$DATA_DIR" --unzip; then
            log_success "Download completed"
            return 0
        else
            if [[ $attempt -lt $max_attempts ]]; then
                log_warn "Download failed. Retrying in ${wait_time}s..."
                sleep $wait_time
                ((attempt++))
                wait_time=$((wait_time * 2))  # Exponential backoff
            else
                log_error "Download failed after $max_attempts attempts"
                return 1
            fi
        fi
    done

    return 1
}

# ============================================================================
# Verification Functions
# ============================================================================

verify_dataset() {
    log_info "Verifying dataset..."

    # Check if images directory exists
    if [[ ! -d "$IMAGES_DIR" ]]; then
        log_error "Images directory not found: $IMAGES_DIR"
        return 1
    fi

    # Count breed directories
    local num_breeds=$(find "$IMAGES_DIR" -maxdepth 1 -type d | tail -n +2 | wc -l)
    log_info "Found $num_breeds breed directories"

    if [[ $num_breeds -lt $EXPECTED_BREEDS ]]; then
        log_warn "Expected $EXPECTED_BREEDS breeds, found $num_breeds"
    fi

    # Count total images
    local num_images=$(find "$IMAGES_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
    log_info "Found $num_images images"

    if [[ $num_images -lt $MIN_IMAGES ]]; then
        log_warn "Expected at least $MIN_IMAGES images, found $num_images"
    fi

    # Spot check: verify 5 random images are readable
    log_info "Spot checking 5 random images..."
    local sample_files=($(find "$IMAGES_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | shuf -n 5))

    for img in "${sample_files[@]}"; do
        if command -v identify &>/dev/null; then
            if identify "$img" &>/dev/null; then
                log_info "  ✓ $(basename "$img")"
            else
                log_warn "  ✗ Corrupted: $(basename "$img")"
            fi
        else
            # If ImageMagick not available, just check file exists and has size
            if [[ -f "$img" && -s "$img" ]]; then
                log_info "  ✓ $(basename "$img") (basic check)"
            else
                log_warn "  ✗ Invalid: $(basename "$img")"
            fi
        fi
    done

    # Record successful download
    mkdir -p "$STATE_DIR"
    echo "DOWNLOAD_TIME=$(date +%s)" > "$META_FILE"
    echo "NUM_BREEDS=$num_breeds" >> "$META_FILE"
    echo "NUM_IMAGES=$num_images" >> "$META_FILE"

    log_success "Dataset verification complete"
    return 0
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    print_header "Cat Breeds Dataset Download"

    log_info "Target directory: $IMAGES_DIR"

    # Check if dataset already exists
    if [[ -d "$IMAGES_DIR" ]]; then
        local num_images=$(find "$IMAGES_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
        if [[ $num_images -gt $MIN_IMAGES ]]; then
            log_success "Dataset already present ($num_images images)"
            log_info "To re-download, remove: $IMAGES_DIR"
            exit 0
        else
            log_warn "Dataset incomplete ($num_images images). Re-downloading..."
            rm -rf "$IMAGES_DIR"
        fi
    fi

    # Check credentials
    check_kaggle_credentials

    # Create data directory
    mkdir -p "$DATA_DIR"

    # Download dataset
    log_info "Downloading Cat Breeds Dataset (~4GB)..."
    log_warn "This may take 10-30 minutes depending on network speed"

    if ! download_with_retry; then
        log_error "Dataset download failed"
        exit 1
    fi

    # Verify dataset
    if ! verify_dataset; then
        log_error "Dataset verification failed"
        exit 1
    fi

    # Success
    print_header "Download Complete"
    log_success "Dataset ready at: $IMAGES_DIR"

    # Show summary
    if [[ -f "$META_FILE" ]]; then
        echo ""
        echo "Dataset Summary:"
        cat "$META_FILE" | while IFS='=' read key value; do
            echo "  $key: $value"
        done
    fi
}

# Run main
main
