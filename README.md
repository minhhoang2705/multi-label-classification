# Cat Breeds Dataset - Multi-Label Classification

This project contains a comprehensive training pipeline for cat breed classification using deep learning, featuring automated GPU server setup and production-ready FastAPI inference.

## Quick Start (Automated Setup)

For fresh Ubuntu 20.04/22.04 servers with NVIDIA GPU drivers installed:

```bash
# Clone repository
git clone <your-repo-url>
cd multi-label-classification

# Run automated setup (installs everything, downloads data, validates)
./setup_training_server.sh

# Start training
source .venv/bin/activate
python scripts/train.py --fast_dev
```

Setup completes in **~10-15 minutes** and includes:
- Python 3.12 environment via uv
- All dependencies (PyTorch, timm, FastAPI, MLflow)
- Cat breeds dataset (~67K images, ~4GB)
- GPU and data validation
- Smoke test (1-epoch training run)

### Prerequisites

| Requirement | Details |
|-------------|---------|
| OS | Ubuntu 20.04 or 22.04 |
| GPU | NVIDIA with drivers installed |
| RAM | 16GB+ recommended |
| Disk | 15GB free space |
| Network | For dataset download (~4GB) |
| Kaggle API | Credentials at `~/.kaggle/kaggle.json` |

**Check GPU drivers:**
```bash
nvidia-smi  # Should show GPU info and CUDA version
```

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
git clone <your-repo-url>
cd multi-label-classification
./setup_training_server.sh
```

### Resume Interrupted Setup

Setup is idempotent and tracks state in `~/.cache/ml-setup/`. If interrupted:

```bash
./setup_training_server.sh  # Continues from last completed phase
```

To force re-run all phases:

```bash
./setup_training_server.sh --force
```

To skip validation/testing:

```bash
./setup_training_server.sh --skip-validation --skip-test
```

### Validate Environment

```bash
source .venv/bin/activate
python scripts/validate_env.py

# Skip throughput benchmark for faster validation
python scripts/validate_env.py --quick
```

### Start Training

```bash
source .venv/bin/activate

# Quick test (2 epochs, 2 folds)
python scripts/train.py --fast_dev

# Full training with EfficientNet-B3
python scripts/train.py --model_name efficientnet_b3 --num_epochs 50 --num_folds 5

# Multi-GPU training
python scripts/train.py --model_name convnext_base --batch_size 64 --num_workers 8
```

### Monitor with MLflow

```bash
./start_mlflow.sh
# Access at http://localhost:5000
```

### Test Trained Model

```bash
./test_model.sh checkpoints/best_model_fold0.pth data/images/Abyssinian/001.jpg
```

### Start FastAPI Inference Server

```bash
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Access at http://localhost:8000/docs
```

## Dataset

The dataset contains images of 67 different cat breeds with varying sample sizes per breed. The dataset exhibits significant class imbalance, making it an interesting challenge for multi-label classification tasks.

**Dataset Statistics:**
- Total images: ~67,000+
- Number of breeds: 67
- Image format: JPG
- Additional metadata: CSV file with breed information

## Files

- `eda_cat_breeds.ipynb` - Comprehensive EDA Jupyter notebook
- `requirements.txt` - Python dependencies
- `data/` - Dataset directory
  - `images/` - Images organized by breed folders
  - `data/cats.csv` - Metadata CSV file

## Setup

### Recommended: Automated Setup

See [Quick Start](#quick-start-automated-setup) above for one-command setup that installs everything in ~10-15 minutes.

### Manual Setup (Alternative)

If you prefer manual installation or are developing locally:

#### 1. Create Virtual Environment

```bash
# Using uv (recommended - 10-100x faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.12
source .venv/bin/activate

# Or using standard venv
python3.12 -m venv .venv
source .venv/bin/activate
```

#### 2. Install Dependencies

```bash
# With uv (faster)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

#### 3. Download Dataset

Using the automated script:

```bash
./scripts/download_dataset.sh
```

Or manually with Kaggle CLI:

```bash
kaggle datasets download -d ma7555/cat-breeds-dataset -p data/ --unzip
```

#### 4. Run the EDA Notebook

```bash
jupyter notebook eda_cat_breeds.ipynb
```

Or use Jupyter Lab:

```bash
jupyter lab eda_cat_breeds.ipynb
```

## Notebook Contents

The EDA notebook includes:

1. **Setup and Imports** - Library imports and configuration
2. **Data Loading and Overview** - Dataset structure exploration
3. **Dataset Statistics** - Basic statistics and distributions
4. **Class Imbalance Analysis** - Comprehensive analysis of class distribution
   - Imbalance ratios and metrics
   - Class weights calculation
   - Visualizations (bar charts, box plots, histograms, pie charts)
   - Recommendations for handling imbalance
5. **Image Analysis** - Image properties analysis
   - Dimensions distribution
   - Aspect ratios
   - File sizes
   - Scatter plots
6. **Visualizations** - Sample images from different breeds
7. **Data Quality Checks** - Identifying corrupted or problematic images
8. **Summary and Recommendations** - Key findings and next steps

## Key Findings

- **Severe class imbalance**: The dataset has significant imbalance with some breeds having thousands of images while others have only a few
- **Variable image dimensions**: Images vary significantly in size
- **Multiple color modes**: Predominantly RGB images
- **Data quality issues**: Some corrupted or very small images identified

## Recommendations

### Preprocessing
- Standardize image sizes (e.g., 224x224 or 299x299)
- Apply data augmentation to minority classes
- Normalize pixel values
- Use stratified train/val/test splits

### Modeling
- Use transfer learning (ResNet, EfficientNet, ViT)
- Apply class weights during training
- Use focal loss or weighted cross-entropy
- Implement stratified k-fold cross-validation
- Monitor per-class metrics

### Class Imbalance Handling
- Data augmentation for minority classes
- Class weighting
- Oversampling strategies
- Consider ensemble methods
- Use focal loss

## Next Steps

1. Implement data preprocessing pipeline
2. Create train/val/test splits with stratification
3. Build baseline model with transfer learning
4. Experiment with different augmentation strategies
5. Fine-tune hyperparameters
6. Evaluate model performance with per-class metrics

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- Pillow
- scikit-learn
- jupyter

See `requirements.txt` for specific versions.

## Troubleshooting

### Setup Issues

**"NVIDIA drivers not installed"**
```bash
# Install NVIDIA drivers
sudo apt update && sudo apt install nvidia-driver-535
sudo reboot

# Verify installation
nvidia-smi
```

**"Kaggle credentials not found"**
1. Go to https://www.kaggle.com/settings
2. Click "Create New Token" under API section
3. Move downloaded file:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Re-run setup: `./setup_training_server.sh`

**"uv command not found" after installation**
```bash
# Add to PATH for current session
export PATH="$HOME/.cargo/bin:$PATH"

# Add to ~/.bashrc for persistence
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Resume interrupted download**
```bash
./scripts/download_dataset.sh  # Automatically resumes from checkpoint
```

**Check setup state**
```bash
cat ~/.cache/ml-setup/cat-breeds.state
```

**Reset setup state and start over**
```bash
rm -rf ~/.cache/ml-setup/
./setup_training_server.sh --force
```

### Training Issues

**"CUDA out of memory"**
```bash
# Reduce batch size and image size
python scripts/train.py --batch_size 16 --image_size 224

# Or use gradient accumulation
python scripts/train.py --batch_size 8 --gradient_accumulation_steps 4
```

**"Too many open files"**
```bash
# Reduce number of data loader workers
python scripts/train.py --num_workers 4
```

**Check GPU utilization**
```bash
watch -n 1 nvidia-smi  # Monitor GPU in real-time
```

### Validation Issues

**Run validation independently**
```bash
source .venv/bin/activate
python scripts/validate_env.py

# Skip throughput benchmark for faster check
python scripts/validate_env.py --quick
```

**Re-validate after fixing issues**
```bash
# Re-run only validation phase
./setup_training_server.sh --skip-test
```

### API Issues

**"Model file not found"**
```bash
# Ensure you have a trained model
ls checkpoints/

# Or train a quick model first
python scripts/train.py --fast_dev
```

**Check API health**
```bash
curl http://localhost:8000/health
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: CUDA out of memory` | Batch size too large | Reduce `--batch_size` |
| `FileNotFoundError: data/images/` | Dataset not downloaded | Run `./scripts/download_dataset.sh` |
| `ImportError: No module named 'torch'` | Dependencies not installed | Run `uv pip install -r requirements.txt` |
| `Permission denied: kaggle.json` | Wrong file permissions | Run `chmod 600 ~/.kaggle/kaggle.json` |
| `CUDA not available` | PyTorch CPU version installed | Reinstall with CUDA support |

### Getting Help

1. Check logs: `~/.cache/ml-setup/setup.log`
2. Run validation: `python scripts/validate_env.py`
3. Check disk space: `df -h`
4. Check GPU: `nvidia-smi`
5. Open an issue with error logs

## License

This project uses the Cat Breeds Dataset from Kaggle. Please refer to the original dataset's license for usage terms.

