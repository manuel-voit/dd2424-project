import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from src.data.transforms import get_train_transforms, get_val_test_transforms


# Download dataset (if not available) and return DataLoaders for binary and multi-class classification
def get_pet_dataloaders(
    data_dir='./data',
    image_size: int = 224,
    batch_size=32,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    binary=False,
    imbalanced=False
):
    """   
    Args:
        data_dir (str): Directory where the dataset will be stored
        batch_size: Number of images per batch
        num_workers: Number of CPU for data loading
        imbalanced: Whether to create an imbalanced version of the dataset (e.g. 20% of the training images for each cat breed)
        
    Returnns:
        dict: dictionary containing the train and test loaders for both tasks.
    """
    # ensure target directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    train_transform = get_train_transforms(image_size=image_size)
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
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=multi_train_trans._labels,
    )

    # Apply imbalance to training set
    if imbalanced:
        # Find idx of cats (0) and dogs (1) in the training set
        cat_indices = [i for i in train_indices if binary_train_trans._bin_labels[i] == 0]
        dog_indices = [i for i in train_indices if binary_train_trans._bin_labels[i] == 1]

        # Downsample cats to 20%, stratifying by breed (even reduction)
        cat_breeds = [multi_train_trans._labels[i] for i in cat_indices]
        reduced_cat_indices, _ = train_test_split(
            cat_indices, train_size=0.2, random_state=42, stratify=cat_breeds
        )

        # Recombine indices
        train_indices = reduced_cat_indices + dog_indices
        print(f"Dataset Imbalanced. Kept {len(reduced_cat_indices)} cats and {len(dog_indices)} dogs.")

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
        loaders = {
            'train': DataLoader(binary_train, shuffle=True, **loader_kwargs),
            'val': DataLoader(binary_val, shuffle=False, **loader_kwargs),
            'test': DataLoader(binary_test, shuffle=False, **loader_kwargs),
        }
    else:    
        loaders = {
                'train': DataLoader(multi_train, shuffle=True, **loader_kwargs),
                'val': DataLoader(multi_val, shuffle=False, **loader_kwargs),
                'test': DataLoader(multi_test, shuffle=False, **loader_kwargs),
        }
        
    return loaders

# Testing block
if __name__ == "__main__":
    # Test standard loader
    print("\nSTANDARD DATASET\n")
    loaders_standard = get_pet_dataloaders(batch_size=16, imbalanced=False)
    print(f"Standard Train Size: {len(loaders_standard['multi']['train'].dataset)}")
    
    # Test imbalanced loader
    print("\nIMBALANCED DATASET\n")
    loaders_imbalanced = get_pet_dataloaders(batch_size=16, imbalanced=True)
    print(f"Imbalanced Train Size: {len(loaders_imbalanced['multi']['train'].dataset)}")