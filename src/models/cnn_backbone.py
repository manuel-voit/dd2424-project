import torch.nn as nn
from torchvision import models

def get_binary_resnet():
    """
    Loads pre-trained ResNet-50, freezes its convolutional base, 
    and replaces the final classification layer for binary classification
    """
        
    # Load pretrained model
    # 'DEFAULT' automatically pulls the most up-to-date, state-of-the-art ImageNet weights
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze base network (loop through all parameters and disable gradient calculations)
    for param in model.parameters():
        param.requires_grad = False

    #Replace the final layer (is called 'fc')
    # First, get Nr. of input features of this layer
    num_ftrs = model.fc.in_features
    
    # Overwrite fc with new Linear layer
    # Output features = 2, since binary task (0: Cat, 1: Dog)
    # when initializing, requires_grad is true by default -> no need to 'unfreeze'
    model.fc = nn.Linear(in_features=num_ftrs, out_features=2)
    
    return model

# Testing block
if __name__ == "__main__":
    # Instantiate model
    binary_model = get_binary_resnet()
    
    # Sanity check
    print("\nMODEL SANITY CHECK:\n")
    print(f"Final layer structure: {binary_model.fc}")
    
    # Check which parameters are actually trainable
    trainable_params = sum(p.numel() for p in binary_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in binary_model.parameters())
    
    print(f"Total parameters in network: {total_params}")
    print(f"Trainable parameters: {trainable_params}") # Should be 4098 = 2048*2