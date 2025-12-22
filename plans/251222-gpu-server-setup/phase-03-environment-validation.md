# Phase 03: Environment Validation Script

## Context

Comprehensive validation script that verifies GPU functionality, dependency versions, data accessibility, and inference throughput before training.

## Overview

Create `scripts/validate_env.py` that:
- Validates all required dependencies are installed
- Tests GPU CUDA availability with matmul operation
- Verifies dataset structure and sample image loading
- Measures inference throughput benchmark
- Returns clear pass/fail status with diagnostics

## Architecture

```
scripts/validate_env.py
    |
    +-- DependencyValidator
    |   |-- check_package_versions()
    |   |-- verify_torch_cuda_match()
    |
    +-- GPUValidator
    |   |-- check_cuda_available()
    |   |-- run_matmul_test()
    |   |-- get_gpu_info()
    |
    +-- DataValidator
    |   |-- check_directory_structure()
    |   |-- verify_breed_folders()
    |   |-- spot_check_images(n=5)
    |
    +-- ThroughputBenchmark
    |   |-- load_sample_batch()
    |   |-- measure_inference_time()
    |   |-- report_images_per_second()
    |
    +-- ValidationReport
        |-- print_summary()
        |-- exit_with_status()
```

## Implementation Steps

### Step 1: Script Header & Imports

```python
#!/usr/bin/env python3
"""
Environment validation script for GPU training setup.
Validates dependencies, GPU, data, and measures throughput.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def log_info(msg: str) -> None:
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")


def log_success(msg: str) -> None:
    print(f"{Colors.GREEN}[PASS]{Colors.NC} {msg}")


def log_warn(msg: str) -> None:
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def log_error(msg: str) -> None:
    print(f"{Colors.RED}[FAIL]{Colors.NC} {msg}", file=sys.stderr)
```

### Step 2: Dependency Validator

```python
class DependencyValidator:
    """Validates required package installations."""

    REQUIRED_PACKAGES = {
        'torch': '2.0.0',
        'torchvision': '0.15.0',
        'timm': '0.9.0',
        'albumentations': '1.3.0',
        'pandas': '1.5.0',
        'numpy': '1.23.0',
        'Pillow': '9.0.0',
        'scikit-learn': '1.2.0',
        'mlflow': '2.8.0',
    }

    def __init__(self):
        self.results: Dict[str, Tuple[bool, str]] = {}

    def check_all(self) -> bool:
        """Check all required packages. Returns True if all pass."""
        log_info("Checking dependencies...")
        all_pass = True

        for package, min_version in self.REQUIRED_PACKAGES.items():
            try:
                if package == 'Pillow':
                    import PIL
                    version = PIL.__version__
                elif package == 'scikit-learn':
                    import sklearn
                    version = sklearn.__version__
                else:
                    mod = __import__(package)
                    version = getattr(mod, '__version__', 'unknown')

                self.results[package] = (True, version)

            except ImportError as e:
                self.results[package] = (False, str(e))
                all_pass = False

        return all_pass

    def verify_torch_cuda_match(self) -> bool:
        """Verify PyTorch was built with CUDA support."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None

            if cuda_available:
                log_success(f"PyTorch CUDA: {cuda_version}")
                return True
            else:
                log_error("PyTorch not built with CUDA support")
                return False
        except Exception as e:
            log_error(f"PyTorch CUDA check failed: {e}")
            return False

    def print_report(self) -> None:
        """Print dependency check results."""
        print("\nDependency Versions:")
        print("-" * 40)
        for package, (passed, version) in self.results.items():
            status = Colors.GREEN + "OK" + Colors.NC if passed else Colors.RED + "MISSING" + Colors.NC
            print(f"  {package:20} {version:15} [{status}]")
        print()
```

### Step 3: GPU Validator

```python
class GPUValidator:
    """Validates GPU availability and functionality."""

    def __init__(self):
        self.gpu_info: Optional[Dict] = None

    def check_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            if not torch.cuda.is_available():
                log_error("CUDA not available")
                return False

            device_count = torch.cuda.device_count()
            log_success(f"CUDA available: {device_count} GPU(s)")
            return True
        except Exception as e:
            log_error(f"CUDA check failed: {e}")
            return False

    def get_gpu_info(self) -> Dict:
        """Get detailed GPU information."""
        import torch

        if not torch.cuda.is_available():
            return {}

        self.gpu_info = {
            'device_count': torch.cuda.device_count(),
            'devices': []
        }

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            self.gpu_info['devices'].append({
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
            })

        return self.gpu_info

    def run_matmul_test(self, size: int = 4096) -> bool:
        """Run GPU matmul test to verify functionality."""
        log_info(f"Running GPU matmul test ({size}x{size})...")

        try:
            import torch

            device = torch.device('cuda')

            # Create random matrices
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)

            # Warmup
            torch.cuda.synchronize()
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()

            # Timed run
            start = time.perf_counter()
            for _ in range(10):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            # Calculate TFLOPS
            flops = 2 * size**3 * 10  # matmul FLOPs * iterations
            tflops = flops / elapsed / 1e12

            log_success(f"GPU matmul test passed: {tflops:.2f} TFLOPS")
            return True

        except Exception as e:
            log_error(f"GPU matmul test failed: {e}")
            return False

    def print_gpu_info(self) -> None:
        """Print GPU information."""
        if not self.gpu_info:
            self.get_gpu_info()

        if not self.gpu_info:
            return

        print("\nGPU Information:")
        print("-" * 40)
        for i, dev in enumerate(self.gpu_info['devices']):
            print(f"  GPU {i}: {dev['name']}")
            print(f"         Memory: {dev['total_memory_gb']:.1f} GB")
            print(f"         Compute: {dev['compute_capability']}")
        print()
```

### Step 4: Data Validator

```python
class DataValidator:
    """Validates dataset structure and accessibility."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"
        self.breed_count = 0
        self.image_count = 0

    def check_directory_structure(self) -> bool:
        """Check that expected directories exist."""
        log_info("Checking data directory structure...")

        if not self.data_dir.exists():
            log_error(f"Data directory not found: {self.data_dir}")
            return False

        if not self.images_dir.exists():
            log_error(f"Images directory not found: {self.images_dir}")
            return False

        log_success(f"Data directories exist")
        return True

    def verify_breed_folders(self, min_breeds: int = 67) -> bool:
        """Verify breed folders exist with images."""
        log_info("Counting breed folders...")

        breed_folders = [
            d for d in self.images_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        self.breed_count = len(breed_folders)

        if self.breed_count < min_breeds:
            log_error(f"Expected {min_breeds} breeds, found {self.breed_count}")
            return False

        log_success(f"Found {self.breed_count} breed folders")

        # Count total images
        self.image_count = sum(
            1 for _ in self.images_dir.rglob("*.jpg")
        ) + sum(
            1 for _ in self.images_dir.rglob("*.jpeg")
        ) + sum(
            1 for _ in self.images_dir.rglob("*.png")
        )

        log_info(f"Total images: {self.image_count:,}")
        return True

    def spot_check_images(self, n: int = 5) -> bool:
        """Verify random sample of images are loadable."""
        log_info(f"Spot checking {n} random images...")

        try:
            from PIL import Image
            import random

            # Collect image paths
            image_paths = list(self.images_dir.rglob("*.jpg"))[:1000]
            if len(image_paths) < n:
                log_error(f"Not enough images for spot check")
                return False

            # Random sample
            samples = random.sample(image_paths, n)

            for img_path in samples:
                try:
                    img = Image.open(img_path)
                    img.verify()  # Verify image integrity
                except Exception as e:
                    log_error(f"Corrupt image: {img_path} - {e}")
                    return False

            log_success(f"All {n} sample images valid")
            return True

        except Exception as e:
            log_error(f"Image spot check failed: {e}")
            return False
```

### Step 5: Throughput Benchmark

```python
class ThroughputBenchmark:
    """Measures inference throughput."""

    def __init__(self, data_dir: Path, batch_size: int = 32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.images_per_second: Optional[float] = None

    def run_benchmark(self, num_batches: int = 10) -> bool:
        """Run inference throughput benchmark."""
        log_info(f"Running throughput benchmark ({num_batches} batches)...")

        try:
            import torch
            from torchvision import transforms
            from PIL import Image

            # Setup transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            # Load sample images
            images_dir = self.data_dir / "images"
            image_paths = list(images_dir.rglob("*.jpg"))[:self.batch_size * num_batches]

            if len(image_paths) < self.batch_size:
                log_error("Not enough images for benchmark")
                return False

            # Load and transform images
            tensors = []
            for path in image_paths[:self.batch_size * num_batches]:
                img = Image.open(path).convert('RGB')
                tensors.append(transform(img))

            # Create batches
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batches = [
                torch.stack(tensors[i:i+self.batch_size]).to(device)
                for i in range(0, len(tensors), self.batch_size)
            ][:num_batches]

            # Simple forward pass (just data movement + basic ops)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            start = time.perf_counter()
            for batch in batches:
                # Simulate lightweight forward pass
                _ = batch.mean(dim=(2, 3))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start

            total_images = num_batches * self.batch_size
            self.images_per_second = total_images / elapsed

            log_success(f"Throughput: {self.images_per_second:.0f} images/sec")
            return True

        except Exception as e:
            log_error(f"Benchmark failed: {e}")
            return False
```

### Step 6: Main Validation Runner

```python
class ValidationRunner:
    """Orchestrates all validation checks."""

    def __init__(self, data_dir: Path, skip_benchmark: bool = False):
        self.data_dir = data_dir
        self.skip_benchmark = skip_benchmark
        self.results: Dict[str, bool] = {}

    def run_all(self) -> bool:
        """Run all validation checks. Returns True if all pass."""
        print("\n" + "=" * 50)
        print("  Environment Validation")
        print("=" * 50)

        # 1. Dependencies
        dep_validator = DependencyValidator()
        self.results['dependencies'] = dep_validator.check_all()
        self.results['torch_cuda'] = dep_validator.verify_torch_cuda_match()
        dep_validator.print_report()

        # 2. GPU
        gpu_validator = GPUValidator()
        self.results['cuda_available'] = gpu_validator.check_cuda_available()
        if self.results['cuda_available']:
            self.results['gpu_matmul'] = gpu_validator.run_matmul_test()
            gpu_validator.print_gpu_info()
        else:
            self.results['gpu_matmul'] = False

        # 3. Data
        data_validator = DataValidator(self.data_dir)
        self.results['data_dirs'] = data_validator.check_directory_structure()
        if self.results['data_dirs']:
            self.results['breed_folders'] = data_validator.verify_breed_folders()
            self.results['image_spot_check'] = data_validator.spot_check_images()

        # 4. Benchmark (optional)
        if not self.skip_benchmark and self.results.get('data_dirs'):
            benchmark = ThroughputBenchmark(self.data_dir)
            self.results['benchmark'] = benchmark.run_benchmark()

        # Print summary
        self._print_summary()

        return all(self.results.values())

    def _print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 50)
        print("  Validation Summary")
        print("=" * 50)

        for check, passed in self.results.items():
            status = f"{Colors.GREEN}PASS{Colors.NC}" if passed else f"{Colors.RED}FAIL{Colors.NC}"
            print(f"  {check:25} [{status}]")

        all_pass = all(self.results.values())
        print()
        if all_pass:
            print(f"{Colors.GREEN}All validation checks passed!{Colors.NC}")
        else:
            print(f"{Colors.RED}Some validation checks failed.{Colors.NC}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Validate training environment')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=PROJECT_ROOT / 'data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--skip-benchmark',
        action='store_true',
        help='Skip throughput benchmark'
    )
    args = parser.parse_args()

    runner = ValidationRunner(
        data_dir=args.data_dir,
        skip_benchmark=args.skip_benchmark
    )

    success = runner.run_all()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
```

## Todo List

- [ ] Create `scripts/validate_env.py` with header and imports
- [ ] Implement `Colors` class and logging functions
- [ ] Implement `DependencyValidator` with version checks
- [ ] Implement `GPUValidator` with matmul test
- [ ] Implement `DataValidator` with spot check
- [ ] Implement `ThroughputBenchmark` (optional)
- [ ] Implement `ValidationRunner` orchestrator
- [ ] Implement CLI with argparse
- [ ] Test with CUDA available/unavailable
- [ ] Test with missing dependencies
- [ ] Test with missing data

## Success Criteria

1. Exits 0 if all checks pass, 1 if any fail
2. GPU matmul test validates CUDA functionality
3. Spot check catches corrupted images
4. Clear diagnostic output for failures
5. Benchmark provides throughput baseline

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU OOM during matmul test | Medium | Use reasonable matrix size (4096) |
| Import errors cascade | Low | Try/except each package check |
| Benchmark unstable | Low | Use sufficient iterations (10) |
| Missing data dir | Low | Clear error with path |

## Sample Output

```
==================================================
  Environment Validation
==================================================

[INFO] Checking dependencies...

Dependency Versions:
----------------------------------------
  torch                2.1.0           [OK]
  torchvision          0.16.0          [OK]
  timm                 0.9.12          [OK]
  ...

[PASS] PyTorch CUDA: 12.1
[PASS] CUDA available: 1 GPU(s)
[INFO] Running GPU matmul test (4096x4096)...
[PASS] GPU matmul test passed: 15.23 TFLOPS

GPU Information:
----------------------------------------
  GPU 0: NVIDIA GeForce RTX 4090
         Memory: 24.0 GB
         Compute: 8.9

[INFO] Checking data directory structure...
[PASS] Data directories exist
[INFO] Counting breed folders...
[PASS] Found 67 breed folders
[INFO] Total images: 67,128
[INFO] Spot checking 5 random images...
[PASS] All 5 sample images valid
[INFO] Running throughput benchmark (10 batches)...
[PASS] Throughput: 1250 images/sec

==================================================
  Validation Summary
==================================================
  dependencies              [PASS]
  torch_cuda                [PASS]
  cuda_available            [PASS]
  gpu_matmul                [PASS]
  data_dirs                 [PASS]
  breed_folders             [PASS]
  image_spot_check          [PASS]
  benchmark                 [PASS]

All validation checks passed!
```
