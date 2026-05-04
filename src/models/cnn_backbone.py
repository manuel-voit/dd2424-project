import torch.nn as nn
from torchvision import models


def get_resnet(num_classes: int, model_name: str = "resnet50"):
    """
    Loads pre-trained ResNet, freezes its convolutional base, 
    and replaces the classification head.
    Available options: resnet18, resnet34, resnet50, resnet101.
    """
    # Load pretrained model
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # ~11.7M params
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT) # ~21.8M params
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # ~25.6M params
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT) # ~44.5M params
    else:
        raise ValueError(f"Unsupported ResNet model: {model_name}")

    # Freeze base network (loop through all parameters and disable gradient calculations)
    for param in model.parameters():
        param.requires_grad = False

    #Replace the final layer (is called 'fc')
    # First, get Nr. of input features of this layer
    num_ftrs = model.fc.in_features
    
    # Overwrite fc with new Linear layer
    # when initializing, requires_grad is true by default -> no need to 'unfreeze'
    model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    
    return model


# Testing block
if __name__ == "__main__":
    # Instantiate model
    binary_model = get_resnet(num_classes=2)
    
    # Sanity check
    print("\nMODEL SANITY CHECK:\n")
    print(f"Final layer structure: {binary_model.fc}")
    
    # Check which parameters are actually trainable
    trainable_params = sum(p.numel() for p in binary_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in binary_model.parameters())
    
    print(f"Total parameters in network: {total_params}")
    print(f"Trainable parameters: {trainable_params}") # Should be 4098 = 2048*2
