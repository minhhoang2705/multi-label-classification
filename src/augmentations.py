"""
Data augmentation transforms for training and validation.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import autoaugment, transforms
from typing import Tuple, Optional
import random
import numpy as np

from .config import AugmentationConfig


class MixUp:
    """Mixup augmentation."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch.

        Args:
            batch: Tuple of (images, labels)

        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lambda)
        """
        images, labels = batch
        batch_size = images.size(0)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """CutMix augmentation."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply cutmix to a batch.

        Args:
            batch: Tuple of (images, labels)

        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lambda)
        """
        images, labels = batch
        batch_size = images.size(0)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        index = torch.randperm(batch_size).to(images.device)

        # Get random box
        _, _, h, w = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2,
                     bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

        labels_a, labels_b = labels, labels[index]

        return mixed_images, labels_a, labels_b, lam


def get_train_transforms(config: AugmentationConfig, image_size: int = 224) -> T.Compose:
    """
    Get training transforms.

    Args:
        config: Augmentation configuration
        image_size: Target image size

    Returns:
        Composed transforms
    """
    transforms_list = []

    # Resize
    transforms_list.append(T.Resize((image_size, image_size)))

    # Horizontal flip
    if config.horizontal_flip:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    # Vertical flip
    if config.vertical_flip:
        transforms_list.append(T.RandomVerticalFlip(p=0.5))

    # Rotation
    if config.rotation_degrees > 0:
        transforms_list.append(T.RandomRotation(
            degrees=config.rotation_degrees))

    # AutoAugment
    if config.use_autoaugment:
        transforms_list.append(autoaugment.AutoAugment(
            policy=autoaugment.AutoAugmentPolicy.IMAGENET
        ))

    # RandAugment
    if config.use_randaugment:
        transforms_list.append(autoaugment.RandAugment(
            num_ops=config.randaugment_n,
            magnitude=config.randaugment_m
        ))

    # Color jitter
    if config.color_jitter:
        transforms_list.append(T.ColorJitter(
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue=config.hue
        ))

    # Convert to tensor
    transforms_list.append(T.ToTensor())

    # Random erasing (applied after ToTensor)
    if config.random_erasing:
        transforms_list.append(T.RandomErasing(
            p=config.random_erasing_prob,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3)
        ))

    # Normalization
    transforms_list.append(T.Normalize(
        mean=config.normalize_mean,
        std=config.normalize_std
    ))

    return T.Compose(transforms_list)


def get_val_transforms(config: AugmentationConfig, image_size: int = 224) -> T.Compose:
    """
    Get validation/test transforms.

    Args:
        config: Augmentation configuration
        image_size: Target image size

    Returns:
        Composed transforms
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std
        )
    ])


def get_test_time_augmentation_transforms(
    config: AugmentationConfig,
    image_size: int = 224,
    n_augmentations: int = 5
) -> list:
    """
    Get test-time augmentation transforms.

    Args:
        config: Augmentation configuration
        image_size: Target image size
        n_augmentations: Number of augmented versions

    Returns:
        List of transform compositions
    """
    tta_transforms = []

    # Original
    tta_transforms.append(get_val_transforms(config, image_size))

    # Horizontal flip
    tta_transforms.append(T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
    ]))

    # Slight rotations
    for angle in [-10, 10]:
        tta_transforms.append(T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=(angle, angle)),
            T.ToTensor(),
            T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        ]))

    # Center crop variations
    tta_transforms.append(T.Compose([
        T.Resize((int(image_size * 1.1), int(image_size * 1.1))),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=config.normalize_mean, std=config.normalize_std)
    ]))

    return tta_transforms[:n_augmentations]


class Denormalize:
    """Denormalize images for visualization."""

    def __init__(self, mean: list, std: list):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor image.

        Args:
            tensor: Normalized image tensor

        Returns:
            Denormalized image tensor
        """
        return tensor * self.std + self.mean
