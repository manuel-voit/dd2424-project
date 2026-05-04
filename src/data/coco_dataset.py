import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Assuming you have these in your project
from src.data.transforms import get_train_transforms, get_val_test_transforms

# --- Custom Target Transforms for MS COCO ---

class CocoBinaryTargetTransform:
    """1 if 'person' (category_id=1) is present, 0 otherwise."""
    def __call__(self, target):
        for ann in target:
            if ann['category_id'] == 1:
                return torch.tensor(1, dtype=torch.long)
        return torch.tensor(0, dtype=torch.long)

class CocoMultiLabelTargetTransform:
    """Converts COCO annotations to a multi-hot 91-dim vector."""
    def __call__(self, target):
        labels = torch.zeros(91, dtype=torch.float32)
        for ann in target:
            labels[ann['category_id']] = 1.0
        return labels

# --- Helper Function ---

def _get_coco_binary_labels(coco_dataset):
    """
    Extracts binary labels (Person vs No Person) quickly using the COCO API.
    Used for stratification and imbalancing without loading images.
    """
    coco = coco_dataset.coco
    img_ids = coco_dataset.ids
    labels = []
    for img_id in img_ids:
        # Check if 'person' (id 1) is in this image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[1])
        labels.append(1 if len(ann_ids) > 0 else 0)
    return labels

# --- Main DataLoader Function ---

def get_coco_dataloaders(
    data_dir='./data/coco',
    image_size: int = 224,
    batch_size=32,
    num_workers=2,
    seed: int = 42,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    binary=False,
    imbalanced=False
):
    """   
    Args:
        data_dir (str): Directory where the COCO dataset is stored.
        image_size: Target size for image transforms.
        batch_size: Number of images per batch.
        num_workers: Number of CPU workers for data loading.
        imbalanced: If True, downsamples the 'Person' class to 20% in the training set.
        binary: If True, returns loaders for binary classification; else multi-label.
        
    Returns:
        dict: dictionary containing the train, val, and test loaders.
    """
    # File Paths (Ensure annotations are downloaded and extracted as discussed!)
    train_img_dir = os.path.join(data_dir, 'train2017')
    train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    test_img_dir = os.path.join(data_dir, 'val2017') # Using COCO val2017 as our test set
    test_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')

    if not os.path.exists(train_ann_file):
        raise FileNotFoundError(f"Annotations not found at {train_ann_file}. Please download annotations_trainval2017.zip")

    # Transforms
    train_transform = get_train_transforms(image_size=image_size)
    test_transform = get_val_test_transforms(image_size=image_size)

    # 1. Load Base Train Datasets
    binary_train_trans = datasets.CocoDetection(
        root=train_img_dir, annFile=train_ann_file, 
        transform=train_transform, target_transform=CocoBinaryTargetTransform()
    )
    binary_val_trans = datasets.CocoDetection(
        root=train_img_dir, annFile=train_ann_file, # Same source, different transform for Val
        transform=test_transform, target_transform=CocoBinaryTargetTransform()
    )
    multi_train_trans = datasets.CocoDetection(
        root=train_img_dir, annFile=train_ann_file, 
        transform=train_transform, target_transform=CocoMultiLabelTargetTransform()
    )
    multi_val_trans = datasets.CocoDetection(
        root=train_img_dir, annFile=train_ann_file, 
        transform=test_transform, target_transform=CocoMultiLabelTargetTransform()
    )

    # 2. Extract labels and Split Train/Val (stratify by Person vs No-Person)
    print("Extracting COCO labels for stratification... (This takes a few seconds)")
    bin_labels = _get_coco_binary_labels(binary_train_trans)
    
    dataset_size = len(binary_train_trans)
    indices = list(range(dataset_size))
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=bin_labels, # Stratify based on Person vs No Person
    )

    # 3. Apply imbalance to training set (Downsample 'Person' images)
    if imbalanced:
        # Separate the train indices based on binary label
        no_person_indices = [i for i in train_indices if bin_labels[i] == 0]
        person_indices = [i for i in train_indices if bin_labels[i] == 1]

        # Downsample 'Person' images to 20%
        reduced_person_indices, _ = train_test_split(
            person_indices, train_size=0.2, random_state=seed
        )

        # Recombine indices
        train_indices = no_person_indices + reduced_person_indices
        print(f"Dataset Imbalanced. Kept {len(no_person_indices)} No-Person and {len(reduced_person_indices)} Person images.")

    # 4. Create Subsets using the split indices
    binary_train = Subset(binary_train_trans, train_indices)
    binary_val = Subset(binary_val_trans, val_indices)
    multi_train = Subset(multi_train_trans, train_indices)
    multi_val = Subset(multi_val_trans, val_indices)

    # 5. Load Test Datasets (using COCO's val2017 split)
    binary_test = datasets.CocoDetection(
        root=test_img_dir, annFile=test_ann_file, 
        transform=test_transform, target_transform=CocoBinaryTargetTransform()
    )
    multi_test = datasets.CocoDetection(
        root=test_img_dir, annFile=test_ann_file, 
        transform=test_transform, target_transform=CocoMultiLabelTargetTransform()
    )

    # 6. Build Dataloaders
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': bool(pin_memory and torch.cuda.is_available()),
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

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

# --- Testing block ---
if __name__ == "__main__":
    try:
        # Test standard loader (Binary Mode)
        print("\n--- STANDARD COCO DATASET (BINARY) ---")
        loaders_standard = get_coco_dataloaders(batch_size=16, imbalanced=False, binary=True)
        print(f"Standard Train Size: {len(loaders_standard['train'].dataset)}")
        print(f"Standard Val Size: {len(loaders_standard['val'].dataset)}")
        print(f"Standard Test Size: {len(loaders_standard['test'].dataset)}")
        
        # Test imbalanced loader (Multi-Label Mode)
        print("\n--- IMBALANCED COCO DATASET (MULTI-LABEL) ---")
        loaders_imbalanced = get_coco_dataloaders(batch_size=16, imbalanced=True, binary=False)
        print(f"Imbalanced Train Size: {len(loaders_imbalanced['train'].dataset)}")
        
    except FileNotFoundError as e:
        print(e)