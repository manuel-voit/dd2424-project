import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from src.data.transforms import get_train_transforms, get_val_test_transforms


# Download dataset (if not available) and return DataLoaders for binary and multi-class classification
def get_pet_dataloaders(
    data_dir='./data',
    image_size: int = 224,
    batch_size=32,
    num_workers=2,
    seed: int = 42,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    binary=False,
    imbalanced=False,
    imbalance_factor=0.2,
    augmentation=True,
    train_fraction=1.0,
    oversample=False
):
    """   
    Args:
        data_dir (str): Directory where the dataset will be stored
        batch_size: Number of images per batch
        num_workers: Number of CPU for data loading
        imbalanced: Whether to create an imbalanced version of the dataset
        imbalance_factor: Fraction in (0, 1] that controls the downsampling strength (e.g. 0.2 means keeping 20% of the cats and all dogs)
        train_fraction: Fraction in (0, 1] that controls the amount of training data used
        augmentation: Whether to apply data augmentation to the training set
        
    Returnns:
        dict: dictionary containing the train and test loaders for both tasks.
    """
    # ensure target directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    if augmentation:
        train_transform = get_train_transforms(image_size=image_size)
    else:
        train_transform = get_val_test_transforms(image_size=image_size)
        
    test_transform = get_val_test_transforms(image_size=image_size)

    # Load Base datasets
    binary_train_trans = datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='binary-category', 
        transform=train_transform, download=True
    )
    binary_val_trans = datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='binary-category', 
        transform=test_transform, download=True
    )
    multi_train_trans = datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='category', 
        transform=train_transform, download=True
    )
    multi_val_trans = datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='category', 
        transform=test_transform, download=True
    )

    # Split Train and Val (stratify)
    dataset_size = len(multi_train_trans)
    indices = list(range(dataset_size))
    
    # Calculate train and val sizes based on train_fraction
    # Default is 80% train, 20% val
    # WE change this ratio (and reduce train size) with train_fraction < 1.0
    # Rest goes to val
    train_size = int(dataset_size * 0.8 * train_fraction)
    val_size = dataset_size - train_size
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=multi_train_trans._labels,
    )

    # Apply imbalance to training set
    if imbalanced:
        if not 0 < imbalance_factor <= 1:
            raise ValueError(
                f"imbalance_factor must be in the range (0, 1], got {imbalance_factor}"
            )

        # Find idx of cats (0) and dogs (1) in the training set
        cat_indices = [i for i in train_indices if binary_train_trans._bin_labels[i] == 0]
        dog_indices = [i for i in train_indices if binary_train_trans._bin_labels[i] == 1]

        # Downsample cats, stratifying by breed (even reduction)
        cat_breeds = [multi_train_trans._labels[i] for i in cat_indices]
        reduced_cat_indices, _ = train_test_split(
            cat_indices,
            train_size=imbalance_factor,
            random_state=seed,
            stratify=cat_breeds
        )

        # Recombine indices
        train_indices = reduced_cat_indices + dog_indices
        print(
            "Dataset imbalanced. "
            f"Kept {len(reduced_cat_indices)} cats ({imbalance_factor:.0%} of cats) "
            f"and {len(dog_indices)} dogs."
        )

    # Create Subsets
    binary_train = Subset(binary_train_trans, train_indices)
    binary_val = Subset(binary_val_trans, val_indices)
    multi_train = Subset(multi_train_trans, train_indices)
    multi_val = Subset(multi_val_trans, val_indices)

    # Load Test Datasets
    binary_test = datasets.OxfordIIITPet(
        root=data_dir, split='test', target_types='binary-category', 
        transform=test_transform, download=True
    )
    multi_test = datasets.OxfordIIITPet(
        root=data_dir, split='test', target_types='category', 
        transform=test_transform, download=True
    )

    # Dataloader Kwargs
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': bool(pin_memory and torch.cuda.is_available()),
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    # Dataloaders
    if binary:
        train_sampler = None
        if oversample:
            targets = [binary_train_trans._bin_labels[i] for i in train_indices]
            sample_weights = compute_sample_weight('balanced', y=targets)
            train_sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True
            )
            
        kwargs_train = loader_kwargs.copy()
        if train_sampler is not None:
            kwargs_train['sampler'] = train_sampler
        else:
            kwargs_train['shuffle'] = True

        loaders = {
            'train': DataLoader(binary_train, **kwargs_train),
            'val': DataLoader(binary_val, shuffle=False, **loader_kwargs),
            'test': DataLoader(binary_test, shuffle=False, **loader_kwargs),
        }
    else:    
        train_sampler = None
        if oversample:
            targets = [multi_train_trans._labels[i] for i in train_indices]
            sample_weights = compute_sample_weight('balanced', y=targets)
            train_sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True
            )

        kwargs_train = loader_kwargs.copy()
        if train_sampler is not None:
            kwargs_train['sampler'] = train_sampler
        else:
            kwargs_train['shuffle'] = True

        loaders = {
                'train': DataLoader(multi_train, **kwargs_train),
                'val': DataLoader(multi_val, shuffle=False, **loader_kwargs),
                'test': DataLoader(multi_test, shuffle=False, **loader_kwargs),
        }
        
    return loaders

# Testing block
if __name__ == "__main__":
    # Test standard loader
    print("\nSTANDARD DATASET\n")
    loaders_standard = get_pet_dataloaders(batch_size=16, imbalanced=False)
    print(f"Standard Train Size: {len(loaders_standard['train'].dataset)}")
    
    # Test imbalanced loader
    print("\nIMBALANCED DATASET\n")
    loaders_imbalanced = get_pet_dataloaders(batch_size=16, imbalanced=True)
    print(f"Imbalanced Train Size: {len(loaders_imbalanced['train'].dataset)}")
