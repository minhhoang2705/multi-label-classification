#!/usr/bin/env bash
#
# docker-stop.sh - Stop Cat Breeds Classification Docker container
#
# Usage:
#   ./scripts/docker-stop.sh [OPTIONS]
#
# Options:
#   --clean       Remove volumes (logs, cache)
#   --prune       Prune unused volumes and images
#   --dry-run     Print commands without executing
#   --help        Show this help message

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
CONTAINER_NAME="cat-breeds-api"
CLEAN_VOLUMES=false
PRUNE=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_VOLUMES=true
            shift
            ;;
        --prune)
            PRUNE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            grep '^#' "$0" | cut -c 3-
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

# Stop container
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_info "Stopping container: ${CONTAINER_NAME}"
    run_cmd docker stop "$CONTAINER_NAME"
else
    log_info "Container not running: ${CONTAINER_NAME}"
fi

# Remove container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_info "Removing container: ${CONTAINER_NAME}"
    run_cmd docker rm "$CONTAINER_NAME"
fi

# Clean volumes if requested
if [ "$CLEAN_VOLUMES" = true ]; then
    log_warn "Cleaning volumes (logs and cache will be deleted)..."
    run_cmd docker volume rm cat_breeds_logs cat_breeds_cache_v1 2>/dev/null || true
    log_info "Volumes cleaned"
fi

# Prune if requested
if [ "$PRUNE" = true ]; then
    log_info "Pruning unused volumes and images..."
    run_cmd docker volume prune -f
    run_cmd docker image prune -f
    log_info "Prune completed"
fi

log_info "Done!"
