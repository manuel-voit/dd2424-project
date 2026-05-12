import torch
from torchvision.transforms import v2 as transforms

# Transformation pipeline for training data (augmentation + normalization)
def get_train_transforms(image_size: int = 224):
    return transforms.Compose([
        # Standard crop augmentation while preserving plausible pet framing
        transforms.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),

        # Horizontal flip is safe for pets and usually helps
        transforms.RandomHorizontalFlip(p=0.5),

        # Mild color jitter keeps color cues useful for breed recognition
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.02,
        ),

        # Small rotations add robustness without distorting breed traits too much
        transforms.RandomRotation(degrees=10),

        # Convert to pytorch tensor
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),

        # Light occlusion regularization after conversion to tensor space
        transforms.RandomErasing(
            p=0.15,
            scale=(0.02, 0.12),
            ratio=(0.3, 3.3),
            value="random",
        ),

        # ImageNet mean and standard deviation
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_test_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
