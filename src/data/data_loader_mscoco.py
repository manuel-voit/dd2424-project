# Not yet tested!!!

import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from src.data.transforms import get_train_transforms, get_test_transforms

# --- Custom Target Transforms for MS COCO ---

class CocoBinaryTargetTransform:
    """
    Converts COCO annotations to a binary label.
    1 if 'person' (category_id=1) is present in the image, 0 otherwise.
    """
    def __call__(self, target):
        for ann in target:
            if ann['category_id'] == 1:
                return torch.tensor(1, dtype=torch.long)
        return torch.tensor(0, dtype=torch.long)

class CocoMultiLabelTargetTransform:
    """
    Converts COCO annotations to a multi-hot vector.
    COCO has 80 classes, but category IDs go up to 90. We use a 91-dim tensor.
    """
    def __call__(self, target):
        labels = torch.zeros(91, dtype=torch.float32)
        for ann in target:
            labels[ann['category_id']] = 1.0
        return labels

# --- DataLoader Function ---

def get_coco_dataloaders(
    data_dir='./data/coco',
    batch_size=32,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
):
    """   
    Args:
        data_dir (str): Directory where the COCO dataset is stored.
        batch_size: Number of images per batch
        num_workers: Number of CPU workers for data loading
        
    Returns:
        dict: dictionary containing the train and val loaders for both tasks.
    """
    
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    # COCO File Paths
    train_img_dir = os.path.join(data_dir, 'train2017')
    train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    val_img_dir = os.path.join(data_dir, 'val2017')
    val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')

    # Ensure paths exist to prevent cryptic errors
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"COCO data not found at {train_img_dir}. Please download it.")

    # 1. Binary Datasets (Person vs. No Person)
    binary_train = datasets.CocoDetection(
        root=train_img_dir, annFile=train_ann_file, 
        transform=train_transform, target_transform=CocoBinaryTargetTransform()
    )
    binary_val = datasets.CocoDetection(
        root=val_img_dir, annFile=val_ann_file, 
        transform=test_transform, target_transform=CocoBinaryTargetTransform()
    )

    # 2. Multi-Label Datasets (Multi-hot encoded 91-dim vector)
    multi_train = datasets.CocoDetection(
        root=train_img_dir, annFile=train_ann_file, 
        transform=train_transform, target_transform=CocoMultiLabelTargetTransform()
    )
    multi_val = datasets.CocoDetection(
        root=val_img_dir, annFile=val_ann_file, 
        transform=test_transform, target_transform=CocoMultiLabelTargetTransform()
    )

    # Loader arguments
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': bool(pin_memory and torch.cuda.is_available()),
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    # Create data loaders
    # Note: We are using standard COCO train/val splits instead of train_test_split. 
    # MS COCO train2017 is ~118k images; val2017 is ~5k images.
    loaders = {
        'binary': {
            'train': DataLoader(binary_train, shuffle=True, **loader_kwargs),
            'val': DataLoader(binary_val, shuffle=False, **loader_kwargs),
        },
        'multi': {
            'train': DataLoader(multi_train, shuffle=True, **loader_kwargs),
            'val': DataLoader(multi_val, shuffle=False, **loader_kwargs),
        },
    }
    
    return loaders

# --- Testing Block ---

if __name__ == "__main__":
    # Test loader function
    # Note: Requires COCO to be downloaded to ./data/coco
    try:
        loaders = get_coco_dataloaders(batch_size=16)
        
        # Fetch a single batch to verify
        binary_images, binary_labels = next(iter(loaders['binary']['train']))
        print("\n--- DATALOADER SANITY CHECK ---\n")
        print(f"Binary Batch Image Shape: {binary_images.shape}")
        print(f"Binary Batch Labels Shape: {binary_labels.shape}")
        print(f"Binary Labels Example (0=No Person, 1=Person): {binary_labels[:5].tolist()}\n")
        
        multi_images, multi_labels = next(iter(loaders['multi']['train']))
        print(f"Multi-Class Batch Labels Shape: {multi_labels.shape}")
        # Displaying a subset of the multi-hot vector for readability
        print(f"Multi-Class Example (First 5 items, categories 0-10): \n{multi_labels[:5, :10].tolist()}")
    
    except FileNotFoundError as e:
        print(e)
