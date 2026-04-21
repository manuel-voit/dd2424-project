import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# return std transforms for ResNet models, include resizing and ImageNet normalization
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and standard deviation
    ])

# Download dataset (if not available) and return DataLoaders for binary and multi-class classification
def get_pet_dataloaders(data_dir='./dataset', batch_size=32, num_workers=2):
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
    
    transform = get_transforms()

    # Binary dataset (0=Cat, 1=Dog)
    binary_train = datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='binary-category', 
        transform=transform, download=True
    )
    binary_test = datasets.OxfordIIITPet(
        root=data_dir, split='test', target_types='binary-category', 
        transform=transform, download=True
    )

    # Multi-class dataset (0-36)
    multi_train = datasets.OxfordIIITPet(
        root=data_dir, split='trainval', target_types='category', 
        transform=transform, download=True
    )
    multi_test = datasets.OxfordIIITPet(
        root=data_dir, split='test', target_types='category', 
        transform=transform, download=True
    )

    # Create data loaders for both tasks
    # Pulls a batch-size nr of samples from the dataset objects, shuffles, and stacks them into torch tensor
    loaders = {
        'binary': {
            'train': DataLoader(binary_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test': DataLoader(binary_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        },
        'multi': {
            'train': DataLoader(multi_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'test': DataLoader(multi_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }
    }
    
    return loaders

# Testing block
if __name__ == "__main__":
    # Test loader function
    loaders = get_pet_dataloaders(batch_size=16)
    
    # Fetch a single batch to verify
    binary_images, binary_labels = next(iter(loaders['binary']['train']))
    print("\n--- Sanity Check ---")
    print(f"Binary Batch Image Shape: {binary_images.shape}")
    print(f"Binary Batch Labels Shape: {binary_labels.shape}")
    print(f"Binary Labels Example (0=Cat, 1=Dog): {binary_labels[:5].tolist()}")
    
    multi_images, multi_labels = next(iter(loaders['multi']['train']))
    print(f"Multi-Class Batch Labels Shape: {multi_labels.shape}")
    print(f"Multi-Class Labels Example (0-36): {multi_labels[:5].tolist()}")