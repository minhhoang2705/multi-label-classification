"""
PyTorch Dataset and DataLoader utilities.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .config import DataConfig, AugmentationConfig
from .augmentations import get_train_transforms, get_val_transforms


class CatBreedsDataset(Dataset):
    """PyTorch Dataset for Cat Breeds."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        label_encoder: Optional[LabelEncoder] = None
    ):
        """
        Initialize dataset.

        Args:
            image_paths: List of image file paths
            labels: List of integer labels
            transform: Torchvision transforms
            label_encoder: Sklearn LabelEncoder for breed names
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (image, label)
        """
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

    def get_breed_name(self, label: int) -> str:
        """Get breed name from label."""
        if self.label_encoder:
            return self.label_encoder.inverse_transform([label])[0]
        return str(label)


def load_dataset_metadata(data_dir: str) -> Tuple[List[str], List[str], LabelEncoder]:
    """
    Load dataset metadata from directory structure.

    Args:
        data_dir: Path to images directory

    Returns:
        Tuple of (image_paths, breed_names, label_encoder)
    """
    data_path = Path(data_dir)

    image_paths = []
    breed_names = []

    # Iterate through breed folders
    for breed_folder in sorted(data_path.iterdir()):
        if not breed_folder.is_dir():
            continue

        breed_name = breed_folder.name

        # Get all images in this breed folder
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in breed_folder.glob(ext):
                image_paths.append(str(img_path))
                breed_names.append(breed_name)

    # Encode breed names to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(breed_names)

    print(
        f"Loaded {len(image_paths)} images from {len(label_encoder.classes_)} breeds")

    return image_paths, breed_names, label_encoder


def create_stratified_splits(
    image_paths: List[str],
    labels: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """
    Create stratified train/val/test splits.

    Args:
        image_paths: List of image paths
        labels: Array of integer labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (paths, labels) tuples
    """
    from sklearn.model_selection import train_test_split

    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_seed
    )

    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_ratio_adjusted,
        stratify=train_val_labels,
        random_state=random_seed
    )

    print(
        f"Split sizes - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }


def create_stratified_kfold_splits(
    image_paths: List[str],
    labels: np.ndarray,
    n_splits: int = 5,
    random_seed: int = 42
) -> List[Dict[str, Tuple[List[str], np.ndarray]]]:
    """
    Create stratified k-fold cross-validation splits.

    Args:
        image_paths: List of image paths
        labels: Array of integer labels
        n_splits: Number of folds
        random_seed: Random seed

    Returns:
        List of dictionaries, each with 'train' and 'val' keys
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        folds.append({
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels)
        })

        print(
            f"Fold {fold_idx + 1}: Train={len(train_paths)}, Val={len(val_paths)}")

    return folds


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.

    Args:
        labels: Array of integer labels
        num_classes: Number of classes

    Returns:
        Tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=labels
    )

    return torch.FloatTensor(class_weights)


def get_dataloaders(
    data_config: DataConfig,
    aug_config: AugmentationConfig,
    splits: Dict[str, Tuple[List[str], np.ndarray]],
    label_encoder: LabelEncoder,
    use_train_augmentation: bool = True
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test splits.

    Args:
        data_config: Data configuration
        aug_config: Augmentation configuration
        splits: Dictionary with split data
        label_encoder: Label encoder
        use_train_augmentation: Whether to use augmentation for training

    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {}

    for split_name, (paths, labels) in splits.items():
        # Choose transforms
        if split_name == 'train' and use_train_augmentation:
            transform = get_train_transforms(
                aug_config, data_config.image_size)
        else:
            transform = get_val_transforms(aug_config, data_config.image_size)

        # Create dataset
        dataset = CatBreedsDataset(
            image_paths=paths,
            labels=labels.tolist(),
            transform=transform,
            label_encoder=label_encoder
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=data_config.batch_size if split_name == 'train' else data_config.batch_size * 2,
            shuffle=(split_name == 'train'),
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
            drop_last=(split_name == 'train')
        )

        dataloaders[split_name] = dataloader

    return dataloaders


def prepare_data_for_training(
    data_config: DataConfig,
    aug_config: AugmentationConfig,
    use_kfold: bool = False
) -> Tuple:
    """
    Prepare all data for training.

    Args:
        data_config: Data configuration
        aug_config: Augmentation configuration
        use_kfold: Whether to use k-fold CV

    Returns:
        If use_kfold=False: (dataloaders, label_encoder, class_weights, num_classes)
        If use_kfold=True: (folds, label_encoder, class_weights, num_classes)
    """
    # Load dataset
    image_paths, breed_names, label_encoder = load_dataset_metadata(
        data_config.data_dir)
    labels = label_encoder.transform(breed_names)
    num_classes = len(label_encoder.classes_)

    # Compute class weights
    class_weights = compute_class_weights(labels, num_classes)

    if use_kfold:
        # Create k-fold splits
        folds = create_stratified_kfold_splits(
            image_paths,
            labels,
            n_splits=data_config.num_folds if hasattr(
                data_config, 'num_folds') else 5,
            random_seed=data_config.random_seed
        )
        return folds, label_encoder, class_weights, num_classes
    else:
        # Create single train/val/test split
        splits = create_stratified_splits(
            image_paths,
            labels,
            train_ratio=data_config.train_ratio,
            val_ratio=data_config.val_ratio,
            test_ratio=data_config.test_ratio,
            random_seed=data_config.random_seed
        )

        dataloaders = get_dataloaders(
            data_config,
            aug_config,
            splits,
            label_encoder
        )

        return dataloaders, label_encoder, class_weights, num_classes


class DatasetStatistics:
    """Compute and display dataset statistics."""

    @staticmethod
    def compute_statistics(labels: np.ndarray, label_encoder: LabelEncoder) -> pd.DataFrame:
        """Compute per-class statistics."""
        unique, counts = np.unique(labels, return_counts=True)

        df = pd.DataFrame({
            'breed': label_encoder.inverse_transform(unique),
            'count': counts,
            'percentage': counts / len(labels) * 100
        })

        df = df.sort_values('count', ascending=False).reset_index(drop=True)

        return df

    @staticmethod
    def print_statistics(labels: np.ndarray, label_encoder: LabelEncoder):
        """Print dataset statistics."""
        df = DatasetStatistics.compute_statistics(labels, label_encoder)

        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"\nTotal samples: {len(labels):,}")
        print(f"Number of classes: {len(label_encoder.classes_)}")
        print(f"\nClass distribution:")
        print(f"  Max: {df['count'].max():,} ({df.iloc[0]['breed']})")
        print(f"  Min: {df['count'].min():,} ({df.iloc[-1]['breed']})")
        print(f"  Mean: {df['count'].mean():.1f}")
        print(f"  Median: {df['count'].median():.1f}")
        print(
            f"  Imbalance ratio: {df['count'].max() / df['count'].min():.2f}:1")
        print("\nTop 10 classes:")
        print(df.head(10).to_string(index=False))
        print("\nBottom 10 classes:")
        print(df.tail(10).to_string(index=False))
