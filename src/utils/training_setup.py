import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm

def split_trainable_parameters(model: nn.Module):
    lora_params = []
    non_lora_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".lora_" in name:
            lora_params.append(param)
        else:
            non_lora_params.append(param)

    return non_lora_params, lora_params

def build_optimizer(model: nn.Module, config: dict):
    training_cfg = config.get('training', {})
    optimizer_cfg = config.get('optimizer', {})

    optimizer_name = optimizer_cfg.get('name', 'adamw').lower()
    base_lr = optimizer_cfg.get('lr', training_cfg.get('learning_rate', 1e-3))
    lora_lr = optimizer_cfg.get('lora_lr', base_lr)
    weight_decay = optimizer_cfg.get('weight_decay', 0.0)

    non_lora_params, lora_params = split_trainable_parameters(model)

    param_groups = []
    if non_lora_params:
        param_groups.append({'params': non_lora_params, 'lr': base_lr})
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': lora_lr})

    optimizer_kwargs = {'params': param_groups}

    if optimizer_name == 'adamw':
        optimizer_kwargs['weight_decay'] = weight_decay
        return optim.AdamW(**optimizer_kwargs)
    elif optimizer_name == 'adam':
        optimizer_kwargs['weight_decay'] = weight_decay
        betas = optimizer_cfg.get('betas', [0.9, 0.999])
        optimizer_kwargs['betas'] = tuple(betas)
        return optim.Adam(**optimizer_kwargs)
    elif optimizer_name == 'sgd':
        optimizer_kwargs['weight_decay'] = weight_decay
        optimizer_kwargs['momentum'] = optimizer_cfg.get('momentum', 0.9)
        optimizer_kwargs['nesterov'] = optimizer_cfg.get('nesterov', False)
        return optim.SGD(**optimizer_kwargs)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def build_loss(config: dict, class_weights: torch.Tensor = None):
    loss_cfg = config.get('loss', {})
    loss_name = loss_cfg.get('name', 'cross_entropy').lower()

    if loss_name == 'cross_entropy':
        kwargs = {}
        if 'label_smoothing' in loss_cfg:
            kwargs['label_smoothing'] = loss_cfg['label_smoothing']
        if class_weights is not None:
            kwargs['weight'] = class_weights
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'bce_with_logits':
        kwargs = {}
        pos_weight = loss_cfg.get('pos_weight')
        if pos_weight is not None:
            kwargs['pos_weight'] = pos_weight
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == 'mse':
        return nn.MSELoss(reduction=loss_cfg.get('reduction', 'mean'))

    raise ValueError(f"Unsupported loss: {loss_name}")

def build_scheduler(optimizer, config: dict):
    scheduler_cfg = config.get('scheduler', {})
    scheduler_name = scheduler_cfg.get('name', 'none').lower()

    if scheduler_name in ('none', ''):
        return None, False

    if scheduler_name == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get('step_size', 5),
            gamma=scheduler_cfg.get('gamma', 0.1),
        )
        return scheduler, False
    elif scheduler_name == 'cosine_annealing_lr':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get('t_max', config.get('training', {}).get('epochs', 1)),
            eta_min=scheduler_cfg.get('eta_min', 0.0),
        )
        return scheduler, False
    elif scheduler_name == 'exponential_lr':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_cfg.get('gamma', 0.95),
        )
        return scheduler, False
    elif scheduler_name == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get('mode', 'min'),
            factor=scheduler_cfg.get('factor', 0.1),
            patience=scheduler_cfg.get('patience', 2),
            min_lr=scheduler_cfg.get('min_lr', 0.0),
        )
        return scheduler, True

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def build_early_stopping(config: dict, save_path: str):
    from src.utils.saving import EarlyStopping

    early_cfg = config.get('early_stopping', {})
    if not early_cfg.get('enabled', True):
        return None

    return EarlyStopping(
        config=config,
        patience=early_cfg.get('patience', 3),
        min_delta=early_cfg.get('min_delta', 0.0),
        save_path=save_path,
    )

def apply_finetuning_strategy(model: nn.Module, config: dict, current_epoch: int = 0):
    ft_config = config.get('model', {}).get('fine_tuning', {})
    strategy = ft_config.get('strategy', 'none').lower()
    
    anything_unfrozen = False
    
    if strategy != 'none':
        num_layers = ft_config.get('num_layers', 0)
        num_layers += 1
        model_type = config.get('model', {}).get('type', 'resnet').lower()
        
        # Organize the top-level feature extraction blocks chronologically backwards
        blocks = []
        if model_type == 'resnet':
            if hasattr(model, 'fc'): blocks.append(model.fc)
            if hasattr(model, 'layer4'): blocks.append(model.layer4)
            if hasattr(model, 'layer3'): blocks.append(model.layer3)
            if hasattr(model, 'layer2'): blocks.append(model.layer2)
            if hasattr(model, 'layer1'): blocks.append(model.layer1)
        elif model_type == 'vit':
            if hasattr(model, 'head'): blocks.append(model.head)
            if hasattr(model, 'layers'):
                for layer in reversed(model.layers):
                    blocks.append(layer)
            else:
                blocks = list(model.children())[::-1]
        else:
            blocks = list(model.children())[::-1]
            
        # Calculate how many layers deep to unfreeze
        if strategy == 'simultaneous':
            blocks_to_unfreeze = min(num_layers, len(blocks))
        elif strategy == 'gradual':
            unfreeze_interval = ft_config.get('unfreeze_every_n_epochs', 3)
            blocks_to_unfreeze = 1 + (current_epoch // unfreeze_interval)
            blocks_to_unfreeze = min(blocks_to_unfreeze, num_layers, len(blocks))
        else:
            blocks_to_unfreeze = 0 # Fallback

        # Apply requires_grad to our chosen blocks
        for block in blocks[:blocks_to_unfreeze]:
            for name, param in block.named_parameters():
                # Leave LoRA components alone (handled independently)
                if ".lora_" in name:
                    continue
                if not param.requires_grad:
                    param.requires_grad = True
                    anything_unfrozen = True

    # LoRA grad. unfreezing
    lora_cfg = config.get('lora', {}) or {}
    unfreeze_cfg = lora_cfg.get('gradual_unfreeze', {}) or {}
    
    if unfreeze_cfg.get('enabled', False):
        schedule = unfreeze_cfg.get('schedule', {})
        schedule = {int(k): v for k, v in schedule.items()}

        # We freeze all lora layers first and then gradually unfreeze them
        if current_epoch == 0:
            for name, param in model.named_parameters():
                if ".lora_" in name:
                    param.requires_grad = False
            
            if 0 in schedule:
                target_layers = schedule[0]
                for name, param in model.named_parameters():
                    if ".lora_" in name and any(layer in name for layer in target_layers):
                        param.requires_grad = True
                        anything_unfrozen = True

        # Unfreeze more layers after each epoch
        elif current_epoch in schedule:
            target_layers = schedule[current_epoch]
            for name, param in model.named_parameters():
                if ".lora_" in name and any(layer in name for layer in target_layers):
                    if not param.requires_grad:
                        param.requires_grad = True
                        anything_unfrozen = True
                
    return anything_unfrozen

def update_optimizer(optimizer: optim.Optimizer, model: nn.Module, config: dict):
    # Retrieve all existing parameters inside the optimizer
    existing_params = {p for group in optimizer.param_groups for p in group['params']}
            
    optimizer_cfg = config.get('optimizer', {})
    base_lr = optimizer_cfg.get('lr', config.get('training', {}).get('learning_rate', 1e-3))
    lora_lr = optimizer_cfg.get('lora_lr', base_lr)
    
    new_non_lora = []
    new_lora = []
    
    # Find any newly unwrapped params that are not managed by optimizer
    for name, param in model.named_parameters():
        if param.requires_grad and param not in existing_params:
            if ".lora_" in name:
                new_lora.append(param)
            else:
                new_non_lora.append(param)
                
    # Add them as a brand new parameter group
    if new_non_lora:
        optimizer.add_param_group({'params': new_non_lora, 'lr': base_lr})
        print(f"Update: Dynamically added {len(new_non_lora)} un-frozen parameter tensors to the optimizer!")
        
    if new_lora:
        optimizer.add_param_group({'params': new_lora, 'lr': lora_lr})
        print(f"Update: Dynamically added {len(new_lora)} un-frozen LoRA tensors to the optimizer!")
    
    return optimizer

def set_batchnorm_mode(model: nn.Module):
    """
    Keep BatchNorm layers in train mode only when their affine parameters are
    trainable. Frozen BatchNorm layers are switched to eval mode so their
    running statistics stop drifting during head-only or partial fine-tuning.
    """
    for module in model.modules():
        if not isinstance(module, _BatchNorm):
            continue

        bn_params = list(module.parameters(recurse=False))
        has_trainable_params = any(param.requires_grad for param in bn_params)

        if has_trainable_params:
            module.train()
        else:
            module.eval()
