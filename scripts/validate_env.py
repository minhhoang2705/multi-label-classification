#!/usr/bin/env python3
"""
Environment validation script for GPU training setup.
Validates dependencies, GPU, data, and measures throughput.

Usage:
    python scripts/validate_env.py
    python scripts/validate_env.py --quick  # Skip benchmark
"""

import sys
import time
import importlib
from pathlib import Path
from typing import Dict, Tuple
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Color Codes
# ============================================================================

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'


def log_info(msg: str) -> None:
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")


def log_success(msg: str) -> None:
    print(f"{Colors.GREEN}[✓ PASS]{Colors.NC} {msg}")


def log_warn(msg: str) -> None:
    print(f"{Colors.YELLOW}[⚠ WARN]{Colors.NC} {msg}")


def log_error(msg: str) -> None:
    print(f"{Colors.RED}[✗ FAIL]{Colors.NC} {msg}", file=sys.stderr)


def print_header(msg: str) -> None:
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


# ============================================================================
# Dependency Validator
# ============================================================================

class DependencyValidator:
    """Validates required package installations."""

    REQUIRED_PACKAGES = {
        'torch': '2.0.0',
        'torchvision': '0.15.0',
        'timm': '0.9.0',
        'albumentations': '1.3.0',
        'pandas': '1.5.0',
        'numpy': '1.23.0',
        'PIL': None,  # Pillow (no version check)
        'scikit-learn': '1.2.0',
        'mlflow': '2.8.0',
        'fastapi': '0.115.0',
    }

    def check_all(self) -> bool:
        """Check all required packages. Returns True if all pass."""
        log_info("Checking dependencies...")
        all_pass = True

        for package, min_version in self.REQUIRED_PACKAGES.items():
            try:
                if package == 'PIL':
                    mod = importlib.import_module('PIL')
                    package_name = 'Pillow'
                else:
                    mod = importlib.import_module(package)
                    package_name = package

                version = getattr(mod, '__version__', 'N/A')
                log_success(f"{package_name}: {version}")

            except ImportError:
                log_error(f"{package}: NOT INSTALLED")
                all_pass = False

        return all_pass


# ============================================================================
# GPU Validator
# ============================================================================

class GPUValidator:
    """Validates GPU and CUDA functionality."""

    def check_all(self) -> bool:
        """Run all GPU checks. Returns True if all pass."""
        log_info("Checking GPU...")

        try:
            import torch
        except ImportError:
            log_error("PyTorch not installed")
            return False

        # Check CUDA availability
        if not torch.cuda.is_available():
            log_error("CUDA not available")
            log_info("  Training will run on CPU (very slow)")
            return False

        # GPU info
        gpu_count = torch.cuda.device_count()
        log_success(f"GPU Count: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            log_info(f"  GPU {i}: {gpu_name}")

        # CUDA/cuDNN versions
        cuda_version = torch.version.cuda if torch.version.cuda else "N/A"
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
        log_success(f"CUDA: {cuda_version} | cuDNN: {cudnn_version}")

        # Matmul test
        if not self._run_matmul_test():
            return False

        return True

    def _run_matmul_test(self) -> bool:
        """Test GPU computation with matrix multiplication."""
        import torch

        log_info("Running GPU matmul test...")

        try:
            # Allocate tensors on GPU
            x = torch.randn(1024, 1024).cuda()
            y = torch.randn(1024, 1024).cuda()

            # Perform matmul
            torch.cuda.synchronize()
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            log_success(f"GPU matmul: {elapsed*1000:.2f}ms")
            return True

        except RuntimeError as e:
            log_error(f"GPU matmul failed: {e}")
            return False


# ============================================================================
# Data Validator
# ============================================================================

class DataValidator:
    """Validates dataset structure and accessibility."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"

    def check_all(self) -> bool:
        """Run all data checks. Returns True if all pass."""
        log_info("Checking dataset...")

        # Check directory exists
        if not self.images_dir.exists():
            log_error(f"Images directory not found: {self.images_dir}")
            log_info("  Run: ./scripts/download_dataset.sh")
            return False

        # Count breed folders
        breed_dirs = [d for d in self.images_dir.iterdir() if d.is_dir()]
        num_breeds = len(breed_dirs)
        log_success(f"Breed directories: {num_breeds}")

        if num_breeds < 67:
            log_warn(f"Expected 67 breeds, found {num_breeds}")

        # Count images
        image_files = list(self.images_dir.rglob("*.jpg")) + \
                      list(self.images_dir.rglob("*.jpeg")) + \
                      list(self.images_dir.rglob("*.png"))
        num_images = len(image_files)
        log_success(f"Total images: {num_images}")

        if num_images < 50000:
            log_warn(f"Expected 50,000+ images, found {num_images}")

        # Spot check 5 random images
        if not self._spot_check_images(image_files[:5]):
            return False

        return True

    def _spot_check_images(self, sample_files) -> bool:
        """Verify sample images are readable."""
        import torch
        from PIL import Image

        log_info("Spot checking sample images...")

        for img_path in sample_files:
            try:
                img = Image.open(img_path)
                img.verify()  # Verify image integrity
                log_info(f"  ✓ {img_path.name}: {img.size} {img.mode}")
            except Exception as e:
                log_error(f"  ✗ {img_path.name}: {e}")
                return False

        log_success("All sample images valid")
        return True


# ============================================================================
# Throughput Benchmark
# ============================================================================

class ThroughputBenchmark:
    """Measures inference throughput."""

    def run(self) -> bool:
        """Run inference benchmark. Returns True on success."""
        import torch
        import torch.nn as nn

        log_info("Running throughput benchmark...")

        try:
            # Create simple CNN model
            model = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ).cuda()

            model.eval()

            # Warmup
            for _ in range(5):
                x = torch.randn(4, 3, 224, 224).cuda()
                with torch.no_grad():
                    _ = model(x)

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()

            for _ in range(100):
                x = torch.randn(4, 3, 224, 224).cuda()
                with torch.no_grad():
                    _ = model(x)

            torch.cuda.synchronize()
            elapsed = time.time() - start

            throughput = (100 * 4) / elapsed  # images/sec
            avg_time_ms = (elapsed / 100) * 1000  # ms per batch

            log_success(f"Throughput: {throughput:.1f} images/sec")
            log_info(f"  Avg time: {avg_time_ms:.2f}ms per batch")

            return True

        except Exception as e:
            log_error(f"Benchmark failed: {e}")
            return False


# ============================================================================
# Main Validation
# ============================================================================

def main():
    """Run all validation checks."""
    parser = argparse.ArgumentParser(description="Validate training environment")
    parser.add_argument('--quick', action='store_true',
                        help='Skip throughput benchmark (faster)')
    args = parser.parse_args()

    print_header("Environment Validation")

    results = {}

    # 1. Dependencies
    print_header("1. Dependency Check")
    validator = DependencyValidator()
    results['dependencies'] = validator.check_all()

    # 2. GPU
    print_header("2. GPU Check")
    gpu_validator = GPUValidator()
    results['gpu'] = gpu_validator.check_all()

    # 3. Dataset
    print_header("3. Dataset Check")
    data_dir = PROJECT_ROOT / "data"
    data_validator = DataValidator(data_dir)
    results['data'] = data_validator.check_all()

    # 4. Throughput (optional)
    if not args.quick and results['gpu']:
        print_header("4. Throughput Benchmark")
        benchmark = ThroughputBenchmark()
        results['benchmark'] = benchmark.run()
    else:
        results['benchmark'] = True  # Skip

    # Summary
    print_header("Validation Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nResults: {passed}/{total} checks passed")
    for check, status in results.items():
        status_str = f"{Colors.GREEN}PASS{Colors.NC}" if status else f"{Colors.RED}FAIL{Colors.NC}"
        print(f"  {check:15s}: {status_str}")

    print()

    if all(results.values()):
        log_success("Environment validation PASSED")
        log_info("Ready to start training!")
        return 0
    else:
        log_error("Environment validation FAILED")
        log_info("Fix errors above before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
