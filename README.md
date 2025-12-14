# Cat Breeds Dataset - Exploratory Data Analysis

This project contains a comprehensive Exploratory Data Analysis (EDA) of the Cat Breeds Dataset from Kaggle.

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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

The dataset has already been downloaded and extracted to the `data/` directory. If you need to re-download:

```bash
curl -L -o data/cat-breeds-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/ma7555/cat-breeds-dataset

cd data && unzip cat-breeds-dataset.zip
```

### 3. Run the Notebook

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

## License

This project uses the Cat Breeds Dataset from Kaggle. Please refer to the original dataset's license for usage terms.

