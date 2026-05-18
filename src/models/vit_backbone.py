import torch.nn as nn
from torchvision import models


def get_vit(num_classes: int, model_name: str = "swin_t"):
    """
    Loads a pre-trained vision transformer backbone, freezes its feature
    extractor, and replaces the classification head.
    Available options: swin_t, swin_s, swin_b, swin_v2_t, vit_b_16.
    """
    if model_name == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)  # ~28.3M params
        head_module = "head"
    elif model_name == "swin_s":
        model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)  # ~49.6M params
        head_module = "head"
    elif model_name == "swin_b":
        model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)  # ~87.8M params
        head_module = "head"
    elif model_name == "swin_v2_t":
        model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)  # ~28.4M params
        head_module = "head"
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT) # ~86.6M params
        head_module = "heads"
    else:
        raise ValueError(f"Unsupported ViT/Swin model: {model_name}")

    for param in model.parameters():
        param.requires_grad = False

    if head_module == "head":
        num_ftrs = model.head.in_features
        model.head = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    else:
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features=num_ftrs, out_features=num_classes)

    return model


def get_swin(num_classes: int, model_name: str = "swin_t"):
    """
    Backward-compatible wrapper for existing imports/configs.
    """
    return get_vit(num_classes=num_classes, model_name=model_name)


# Testing block
if __name__ == "__main__":
    binary_model = get_vit(num_classes=2)

    print("\nMODEL SANITY CHECK:\n")
    print(binary_model)

    trainable_params = sum(p.numel() for p in binary_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in binary_model.parameters())

    print(f"Total parameters in network: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
