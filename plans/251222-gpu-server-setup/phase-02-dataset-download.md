# Phase 02: Dataset Download Script

## Context

Automated dataset download with Kaggle API, checkpoint tracking, and retry logic for resilient downloads.

## Overview

Create `scripts/download_dataset.sh` that:
- Validates/configures Kaggle API credentials
- Downloads Cat Breeds Dataset (~4GB) with resume support
- Extracts to `data/images/` directory
- Tracks download state for idempotency
- Verifies data integrity post-extraction

## Architecture

```
scripts/download_dataset.sh
    |
    +-- Credential Management
    |   |-- Check ~/.kaggle/kaggle.json
    |   |-- Validate API key with test request
    |   |-- Interactive prompt if missing
    |
    +-- Download Logic
    |   |-- Checkpoint: ~/.cache/ml-setup/dataset.meta
    |   |-- kaggle datasets download with --unzip
    |   |-- Retry logic (3 attempts, exponential backoff)
    |
    +-- Verification
        |-- Check expected directory structure
        |-- Count images per breed
        |-- Validate sample images readable
```

## Implementation Steps

### Step 1: Script Header & Configuration

```bash
#!/usr/bin/env bash
set -euo pipefail

# Configuration
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

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
```

### Step 2: Kaggle Credential Management

```bash
check_kaggle_credentials() {
    local kaggle_json="$HOME/.kaggle/kaggle.json"

    if [[ ! -f "$kaggle_json" ]]; then
        log_error "Kaggle credentials not found: $kaggle_json"
        echo ""
        echo "To set up Kaggle API:"
        echo "1. Go to https://www.kaggle.com/settings"
        echo "2. Click 'Create New Token' under API section"
        echo "3. Move downloaded file:"
        echo "   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
        echo "4. Set permissions:"
        echo "   chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        exit 1
    fi

    # Ensure correct permissions
    chmod 600 "$kaggle_json"

    # Validate credentials with test request
    log_info "Validating Kaggle credentials..."
    if ! kaggle datasets list --max-size 1 &>/dev/null; then
        log_error "Kaggle credentials invalid. Please regenerate token."
        exit 1
    fi

    log_success "Kaggle credentials valid"
}

install_kaggle_cli() {
    if ! command -v kaggle &>/dev/null; then
        log_info "Installing Kaggle CLI..."
        pip install --quiet kaggle
    fi
    log_info "Kaggle CLI version: $(kaggle --version)"
}
```

### Step 3: Download State Management

```bash
is_download_complete() {
    # Check metadata file
    if [[ ! -f "$META_FILE" ]]; then
        return 1
    fi

    # Verify images directory exists with content
    if [[ ! -d "$IMAGES_DIR" ]]; then
        return 1
    fi

    local breed_count=$(find "$IMAGES_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    if [[ "$breed_count" -lt "$EXPECTED_BREEDS" ]]; then
        log_warn "Found $breed_count breeds, expected $EXPECTED_BREEDS"
        return 1
    fi

    return 0
}

save_download_metadata() {
    mkdir -p "$STATE_DIR"
    cat > "$META_FILE" << EOF
dataset=$KAGGLE_DATASET
download_date=$(date -Iseconds)
breed_count=$(find "$IMAGES_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
image_count=$(find "$IMAGES_DIR" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
EOF
    log_success "Download metadata saved: $META_FILE"
}
```

### Step 4: Download with Retry Logic

```bash
download_dataset() {
    local max_retries=3
    local retry_delay=5

    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"

    for attempt in $(seq 1 $max_retries); do
        log_info "Download attempt $attempt/$max_retries..."

        if kaggle datasets download -d "$KAGGLE_DATASET" --unzip --force; then
            log_success "Download completed"
            return 0
        fi

        if [[ $attempt -lt $max_retries ]]; then
            log_warn "Download failed, retrying in ${retry_delay}s..."
            sleep $retry_delay
            retry_delay=$((retry_delay * 2))
        fi
    done

    log_error "Download failed after $max_retries attempts"
    return 1
}
```

### Step 5: Alternative Download Method (curl fallback)

```bash
download_dataset_curl() {
    log_info "Attempting curl download (fallback)..."

    local zip_file="${DATA_DIR}/cat-breeds-dataset.zip"

    # Read credentials
    local kaggle_json="$HOME/.kaggle/kaggle.json"
    local username=$(jq -r '.username' "$kaggle_json")
    local key=$(jq -r '.key' "$kaggle_json")

    # Download with resume support (-C -)
    curl -L -C - \
        -u "${username}:${key}" \
        -o "$zip_file" \
        "https://www.kaggle.com/api/v1/datasets/download/${KAGGLE_DATASET}"

    if [[ ! -f "$zip_file" ]]; then
        log_error "Curl download failed"
        return 1
    fi

    # Extract
    log_info "Extracting dataset..."
    cd "$DATA_DIR"
    unzip -q -o "$zip_file"

    # Cleanup zip
    rm -f "$zip_file"

    log_success "Curl download completed"
}
```

### Step 6: Data Verification

```bash
verify_dataset() {
    log_info "Verifying dataset..."

    # Check directory structure
    if [[ ! -d "$IMAGES_DIR" ]]; then
        log_error "Images directory not found: $IMAGES_DIR"
        return 1
    fi

    # Count breeds
    local breed_count=$(find "$IMAGES_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    log_info "Breeds found: $breed_count"

    if [[ "$breed_count" -lt "$EXPECTED_BREEDS" ]]; then
        log_error "Expected $EXPECTED_BREEDS breeds, found $breed_count"
        return 1
    fi

    # Count total images
    local image_count=$(find "$IMAGES_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
    log_info "Images found: $image_count"

    if [[ "$image_count" -lt "$MIN_IMAGES" ]]; then
        log_error "Expected at least $MIN_IMAGES images, found $image_count"
        return 1
    fi

    # Spot check: verify 5 random images are readable
    log_info "Spot checking random images..."
    local test_images=$(find "$IMAGES_DIR" -type f -name "*.jpg" | shuf | head -5)

    for img in $test_images; do
        if ! file "$img" | grep -q "image data"; then
            log_error "Corrupted image: $img"
            return 1
        fi
    done

    log_success "Dataset verification passed"
    return 0
}
```

### Step 7: Main Function

```bash
main() {
    echo ""
    echo "========================================"
    echo "  Cat Breeds Dataset Download"
    echo "========================================"
    echo ""

    # Check if already downloaded
    if is_download_complete; then
        log_success "Dataset already downloaded and verified"
        cat "$META_FILE"
        return 0
    fi

    # Setup
    install_kaggle_cli
    check_kaggle_credentials

    # Download
    if ! download_dataset; then
        log_warn "Kaggle CLI failed, trying curl fallback..."
        download_dataset_curl
    fi

    # Verify
    verify_dataset

    # Save metadata
    save_download_metadata

    echo ""
    log_success "Dataset ready at: $IMAGES_DIR"
}

main "$@"
```

## Todo List

- [ ] Create `scripts/download_dataset.sh` with header and config
- [ ] Implement Kaggle credential check with user-friendly error message
- [ ] Implement `install_kaggle_cli` function
- [ ] Implement download state management (check, save metadata)
- [ ] Implement `download_dataset` with retry logic (3 attempts, exp backoff)
- [ ] Implement curl fallback with resume support (-C -)
- [ ] Implement `verify_dataset` (breed count, image count, spot check)
- [ ] Implement main function with idempotency check
- [ ] Make script executable: `chmod +x scripts/download_dataset.sh`
- [ ] Test with valid and invalid credentials
- [ ] Test resume after interrupted download

## Success Criteria

1. Downloads ~4GB dataset in <10 min on good connection
2. Resumes interrupted downloads (via kaggle --force or curl -C)
3. Clear instructions when credentials missing
4. Idempotent: skips if already downloaded
5. Verifies data integrity before marking complete

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Kaggle rate limits | Medium | Retry with exponential backoff |
| Credential setup confusion | Medium | Clear step-by-step instructions |
| Corrupt zip download | Medium | Verify images post-extraction |
| Disk space exhaustion mid-download | High | Pre-check disk space in Phase 01 |
| Dataset structure changes | Low | Flexible breed count check (>=67) |

## Data Structure Expected

```
data/
├── images/
│   ├── Abyssinian/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   ├── American Bobtail/
│   │   └── ...
│   └── ... (67 breed folders)
├── cats.csv (optional metadata)
└── clean_metadata.csv (if present)
```
