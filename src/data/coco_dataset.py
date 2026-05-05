import os
import torch
import random
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import IterativeStratification
from src.data.transforms import get_train_transforms, get_val_test_transforms

# Mapping of 90 ids to 80 actual classes (0-79)
VALID_COCO_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
]
COCO_ID_TO_INDEX = {coco_id: index for index, coco_id in enumerate(VALID_COCO_IDS)}

# --- Custom Target Transforms for MS COCO ---

class CocoBinaryTargetTransform:
    """1 if 'person' (category_id=1) is present, 0 otherwise."""
    def __call__(self, target):
        for ann in target:
            if ann['category_id'] == 1:
                return torch.tensor(1, dtype=torch.float32) # Better FOR BCE loss
        return torch.tensor(0, dtype=torch.float32)

class CocoMultiLabelTargetTransform:
    """Converts COCO annotations to a multi-hot 91-dim vector."""
    def __call__(self, target):
        labels = torch.zeros(80, dtype=torch.float32)
        for ann in target:
            cat_id = ann['category_id']
            if cat_id in COCO_ID_TO_INDEX:
                mapped_idx = COCO_ID_TO_INDEX[cat_id]
                labels[mapped_idx] = 1.0
        return labels

def create_multilabel_stratified_split(dataset, multi_hot_labels, test_size=0.2):
    if isinstance(multi_hot_labels, torch.Tensor):
        multi_hot_labels = multi_hot_labels.numpy()

    stratifier = IterativeStratification(
        n_splits=2, 
        order=1, 
        sample_distribution_per_fold=[test_size, 1.0 - test_size]
    )
    
    dummy_X = np.zeros((len(multi_hot_labels), 1))
    train_indices, val_indices = next(stratifier.split(dummy_X, multi_hot_labels))
    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    
    return train_subset, val_subset

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

def _get_coco_multilabel_labels(coco_dataset):
    """
    Extracts multi-label targets quickly using the COCO API without loading images.
    """
    coco = coco_dataset.coco
    img_ids = coco_dataset.ids
    labels = []
    
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        label = np.zeros(80, dtype=np.float32)
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in COCO_ID_TO_INDEX:
                mapped_idx = COCO_ID_TO_INDEX[cat_id]
                label[mapped_idx] = 1.0
        labels.append(label)
        
    return np.stack(labels)


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
    imbalanced=False,
    imbalance_factor=0.2,
    augmentation=True,
    train_fraction=1.0
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
    # File Paths Sanity Check
    train_img_dir = os.path.join(data_dir, 'train2017')
    train_ann_file = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    test_img_dir = os.path.join(data_dir, 'val2017') # Using COCO val2017 as our test set
    test_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
    val_img_dir = os.path.join(data_dir, 'val2017') # Using COCO val2017 as our test set
    val_ann_file = os.path.join(data_dir, 'annotations', 'instances_val2017.json')

    paths_to_check = [train_img_dir, train_ann_file, val_img_dir, val_ann_file]
    for path in paths_to_check:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing COCO data file or directory: {path}")

    # Transforms
    if augmentation:
        train_transform = get_train_transforms(image_size=image_size)
    else:
        train_transform = get_val_test_transforms(image_size=image_size)
    test_transform = get_val_test_transforms(image_size=image_size)

    train_dataset_raw = datasets.CocoDetection(
        root=train_img_dir, annFile=train_ann_file, 
        transform=train_transform, 
        target_transform=CocoBinaryTargetTransform() if binary else CocoMultiLabelTargetTransform()
    )
    val_dataset_raw = datasets.CocoDetection(
        root=train_img_dir, annFile=train_ann_file, 
        transform=test_transform, 
        target_transform=CocoBinaryTargetTransform() if binary else CocoMultiLabelTargetTransform()
    )

    if binary:
        bin_labels = _get_coco_binary_labels(train_dataset_raw)
        indices = list(range(len(train_dataset_raw)))
        
        if train_fraction < 1.0:
            indices, _, bin_labels, _ = train_test_split(
                indices, bin_labels, train_size=train_fraction, random_state=seed, stratify=bin_labels
            )

        train_indices, val_indices = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=bin_labels
        )

        if imbalanced:
            print("Applying binary imbalance logic...")
            no_person_indices = [i for i in train_indices if bin_labels[i] == 0]
            person_indices = [i for i in train_indices if bin_labels[i] == 1]

            reduced_person_indices, _ = train_test_split(
                person_indices, train_size=imbalance_factor, random_state=seed
            )
            train_indices = no_person_indices + reduced_person_indices
            print(f"Dataset Imbalanced. Kept {len(no_person_indices)} No-Person and {len(reduced_person_indices)} Person images.")

        train_subset = Subset(train_dataset_raw, train_indices)
        val_subset = Subset(val_dataset_raw, val_indices)

    else:
        if imbalanced:
            print("Not yet implemented!")
        
        # Fast multi-label extraction
        all_labels_matrix = _get_coco_multilabel_labels(train_dataset_raw)

        train_subset, val_subset = create_multilabel_stratified_split(
            dataset=train_dataset_raw, 
            multi_hot_labels=all_labels_matrix, 
            test_size=0.2 
        )

        if train_fraction < 1.0:
            random.seed(seed)
            
            # Subsample train indices
            train_k = int(len(train_subset) * train_fraction)
            new_train_indices = random.sample(train_subset.indices, train_k)
            train_subset = Subset(train_dataset_raw, new_train_indices)
            
            # Subsample val indices
            val_k = int(len(val_subset) * train_fraction)
            new_val_indices = random.sample(val_subset.indices, val_k)
            val_subset = Subset(train_dataset_raw, new_val_indices)

        # Apply the Val transforms to the val_subset
        val_subset.dataset = val_dataset_raw

    # Load Test Datasets (using COCO's val2017 split)
    test_dataset = datasets.CocoDetection(
        root=test_img_dir, annFile=test_ann_file, 
        transform=test_transform, 
        target_transform=CocoBinaryTargetTransform() if binary else CocoMultiLabelTargetTransform()
    )

    # Build Dataloaders
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': bool(pin_memory and torch.cuda.is_available()),
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor

    loaders = {
        'train': DataLoader(train_subset, shuffle=True, **loader_kwargs),
        'val': DataLoader(val_subset, shuffle=False, **loader_kwargs),
        'test': DataLoader(test_dataset, shuffle=False, **loader_kwargs),
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