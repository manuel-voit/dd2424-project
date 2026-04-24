import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from src.data.transforms import get_train_transforms, get_test_transforms

# Download dataset (if not available) and return DataLoaders for binary and multi-class classification
def get_pet_dataloaders(
    data_dir='./data',
    batch_size=32,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
):
    """   
    Args:
        data_dir (str): Directory where the dataset will be stored
        batch_size: Number of images per batch
        num_workers: Number of CPU for data loading
        
    Returnns:
        dict: dictionary containing the train and test loaders for both tasks.
    """
    
    # ensure target directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    # Binary dataset (0=Cat, 1=Dog)
    # Different transforms for train and val    
    binary_train_trans = datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='binary-category', 
        transform=train_transform, download=True
    )
    binary_val_trans = datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='binary-category', 
        transform=test_transform, download=True
    )
    # Splitting train and val set from trainval in OxfordIIIT Dataset
    dataset_size = len(binary_train_trans)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=binary_train_trans._bin_labels,
    )
    binary_train = Subset(binary_train_trans, train_indices)
    binary_val = Subset(binary_val_trans, val_indices)

    binary_test = datasets.OxfordIIITPet(
        root=data_dir, split='test', target_types='binary-category', 
        transform=test_transform, download=True
    )

    # Multi-class dataset (0-36)
    # Different transforms for train and val    

    multi_train_trans = datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='category', 
        transform=train_transform, download=True
    )
    multi_val_trans = datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='category', 
        transform=test_transform, download=True
    )

    dataset_size = len(multi_train_trans)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=multi_train_trans._labels,
    )
    multi_train = Subset(multi_train_trans, train_indices)
    multi_val = Subset(multi_val_trans, val_indices)

    multi_test = datasets.OxfordIIITPet(
        root=data_dir, split='test', target_types='category', 
        transform=test_transform, download=True
    )

    # Keep workers alive between epochs to avoid costly process respawn on Windows.
    # Pinning host memory speeds up CPU->GPU transfers when using CUDA.
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': bool(pin_memory and torch.cuda.is_available()),
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    # Create data loaders for both tasks
    # Pulls a batch-size nr of samples from the dataset objects, shuffles, and stacks them into torch tensor
    loaders = {
        'binary': {
            'train': DataLoader(binary_train, shuffle=True, **loader_kwargs),
            'val': DataLoader(binary_val, shuffle=False, **loader_kwargs),
            'test': DataLoader(binary_test, shuffle=False, **loader_kwargs),
        },
        'multi': {
            'train': DataLoader(multi_train, shuffle=True, **loader_kwargs),
            'val': DataLoader(multi_val, shuffle=False, **loader_kwargs),
            'test': DataLoader(multi_test, shuffle=False, **loader_kwargs),
        },
    }
    
    return loaders

# Testing block
if __name__ == "__main__":
    # Test loader function
    loaders = get_pet_dataloaders(batch_size=16)
    
    # Fetch a single batch to verify
    binary_images, binary_labels = next(iter(loaders['binary']['train']))
    print("\nDATALOADER SANITY CHECK:\n")
    print(f"Binary Batch Image Shape: {binary_images.shape}")
    print(f"Binary Batch Labels Shape: {binary_labels.shape}")
    print(f"Binary Labels Example (0=Cat, 1=Dog): {binary_labels[:5].tolist()}")
    
    multi_images, multi_labels = next(iter(loaders['multi']['train']))
    print(f"Multi-Class Batch Labels Shape: {multi_labels.shape}")
    print(f"Multi-Class Labels Example (0-36): {multi_labels[:5].tolist()}")