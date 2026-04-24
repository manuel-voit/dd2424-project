import torch

from src.models.cnn_backbone import get_resnet
from src.models.vit_backbone import get_swin_t
from src.models.lora import inject_lora


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint from {checkpoint_path} ...")
    
    # Load the dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    weights = checkpoint['model_state_dict']

    # Rebuild the base architecture based on the saved config
    model_type = config['model']['type']
    num_classes = config['model']['num_classes']
    if model_type == 'resnet':
        model = get_resnet(num_classes=num_classes)
    elif model_type == 'vit':
        model = get_swin_t(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")

    # Inject LoRA if the config says it was used during training
    if 'lora' in config and config['lora']:
        model = inject_lora(
            model,
            target_layer_names=config['lora']['targets'],
            r=config['lora']['r'],
            alpha=config['lora']['alpha']
        )

    # Load the weights
    # strict=False ignores the frozen base layers
    model.load_state_dict(weights, strict=False)
    
    # Finalize the model for inference
    model = model.to(device)
    model.eval()
    
    return model, config
