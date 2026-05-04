import torch

from src.models.cnn_backbone import get_resnet
from src.models.vit_backbone import get_swin
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
    model_name = config['model'].get('name', None)
    
    if model_type == 'resnet':
        model = get_resnet(num_classes=num_classes, model_name=model_name or "resnet50")
    elif model_type == 'vit':
        model = get_swin(num_classes=num_classes, model_name=model_name or "swin_t")
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
    # strict=False allows checkpoints that only store trainable parameters
    missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)

    trainable_keys = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_keys.add(name)

    missing_trainable = sorted(set(missing_keys) & trainable_keys)

    if missing_trainable:
        print(f"Warning: missing trainable keys when loading checkpoint: {missing_trainable}")
    elif missing_keys:
        print(
            "Checkpoint loaded with partial weights as expected: "
            f"{len(missing_keys)} frozen/base parameter(s) were not present in the checkpoint."
        )

    if unexpected_keys:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected_keys}")
    
    # Finalize the model for inference
    model = model.to(device)
    model.eval()
    
    return model, config
