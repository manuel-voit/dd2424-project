import torch
from torchvision.transforms import v2 as transforms

# Transformation pipeline for training data (augmentation + normalization)
def get_train_transforms():
    return transforms.Compose([
        # Random size scaling and cropping
        # Randomly crop portion of the image (80% to 100% of original area) and resize to 224x224
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        
        # horizontal flip (50% chance)
        transforms.RandomHorizontalFlip(p=0.5),

        # Random convert to grayscale
        transforms.RandomGrayscale(p=0.1),
        
        # Random rotations ( -15 to +15 degrees)
        transforms.RandomRotation(degrees=15),

        # Random perspective transformation
        transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
        
        # Convert to pytorch tensor
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        
        # ImageNet mean and standard deviation
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_transforms():
    """
    Returns the transformation pipeline for TESTING/VALIDATION data.
    Strictly deterministic operations: only resizing and normalizing.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])