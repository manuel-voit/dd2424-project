import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm


def _chunk_blocks(blocks, chunk_size: int):
    return [blocks[i:i + chunk_size] for i in range(0, len(blocks), chunk_size)]


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


def _get_vit_llrd_info(model_name: str):
    model_name = model_name.lower()
    if model_name == "vit_b_16":
        return {"enabled": True, "num_layers": 14}
    return {"enabled": False, "num_layers": 0}


def _get_vit_layer_id(param_name: str, model_name: str):
    model_name = model_name.lower()

    if model_name == "vit_b_16":
        if param_name.startswith("heads."):
            return 13
        if param_name.startswith("encoder.ln."):
            return 12
        if param_name.startswith("encoder.layers.encoder_layer_"):
            layer_str = param_name.split("encoder.layers.encoder_layer_", 1)[1].split(".", 1)[0]
            return int(layer_str) + 1
        return 0


def _build_non_lora_param_groups(model: nn.Module, config: dict, base_lr: float):
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "").lower()
    model_name = model_cfg.get("name", "")
    llrd_cfg = config.get("optimizer", {}).get("llrd", {}) or {}
    llrd_enabled = bool(llrd_cfg.get("enabled", False)) and model_type == "vit"

    non_lora_named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and ".lora_" not in name
    ]

    if not llrd_enabled:
        return [{"params": [param for _, param in non_lora_named_params], "lr": base_lr}] if non_lora_named_params else []

    vit_llrd_info = _get_vit_llrd_info(model_name)
    if not vit_llrd_info["enabled"]:
        return [{"params": [param for _, param in non_lora_named_params], "lr": base_lr}] if non_lora_named_params else []

    decay = float(llrd_cfg.get("decay", 0.75))
    max_layer_id = vit_llrd_info["num_layers"] - 1

    grouped = {}
    for name, param in non_lora_named_params:
        layer_id = _get_vit_layer_id(name, model_name)
        lr = base_lr * (decay ** (max_layer_id - layer_id))
        if lr not in grouped:
            grouped[lr] = []
        grouped[lr].append(param)

    return [{"params": params, "lr": lr} for lr, params in sorted(grouped.items(), key=lambda item: item[0])]

def build_optimizer(model: nn.Module, config: dict):
    training_cfg = config.get('training', {})
    optimizer_cfg = config.get('optimizer', {})

    optimizer_name = optimizer_cfg.get('name', 'adamw').lower()
    base_lr = optimizer_cfg.get('lr', training_cfg.get('learning_rate', 1e-3))
    lora_lr = optimizer_cfg.get('lora_lr', base_lr)
    weight_decay = optimizer_cfg.get('weight_decay', 0.0)

    _, lora_params = split_trainable_parameters(model)

    param_groups = _build_non_lora_param_groups(model, config, base_lr)
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
    total_epochs = config.get('training', {}).get('epochs', 1)

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
            T_max=scheduler_cfg.get('t_max', total_epochs),
            eta_min=scheduler_cfg.get('eta_min', 0.0),
        )
        return scheduler, False
    elif scheduler_name == 'cosine_annealing_with_linear_warmup':
        warmup_epochs = scheduler_cfg.get('warmup_epochs', 3)
        warmup_start_factor = scheduler_cfg.get('warmup_start_factor', 1e-6)
        eta_min = scheduler_cfg.get('eta_min', 0.0)

        if warmup_epochs <= 0:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_cfg.get('t_max', total_epochs),
                eta_min=eta_min,
            )
            return scheduler, False

        if warmup_epochs >= total_epochs:
            raise ValueError(
                "scheduler.warmup_epochs must be smaller than training.epochs "
                "for cosine_annealing_with_linear_warmup."
            )

        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get('t_max', total_epochs - warmup_epochs),
            eta_min=eta_min,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
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
    model_name = config.get('model', {}).get('name', '').lower()
    
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
            stem_layers = []
            if hasattr(model, 'maxpool'): stem_layers.append(model.maxpool)
            if hasattr(model, 'relu'): stem_layers.append(model.relu)
            if hasattr(model, 'bn1'): stem_layers.append(model.bn1)
            if hasattr(model, 'conv1'): stem_layers.append(model.conv1)
            if stem_layers: blocks.append(stem_layers)
        elif model_type == 'vit':
            if hasattr(model, 'head'):
                blocks.append(model.head)
            elif hasattr(model, 'heads'):
                blocks.append(model.heads)

            if hasattr(model, 'encoder') and hasattr(model.encoder, 'ln'):
                blocks.append(model.encoder.ln)

            if hasattr(model, 'layers'):
                for layer in reversed(model.layers):
                    blocks.append(layer)
            elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
                encoder_layers = list(model.encoder.layers.children())
                if model_name == 'vit_b_16' and len(encoder_layers) == 12:
                    # Treat ViT-B/16 as 4 stages of 3 transformer blocks each
                    stage_blocks = _chunk_blocks(encoder_layers, 3)
                    for stage in reversed(stage_blocks):
                        blocks.append(stage)
                else:
                    for layer in reversed(encoder_layers):
                        blocks.append(layer)
            else:
                blocks = list(model.children())[::-1]
                
            stem_modules = []
            if hasattr(model, 'conv_proj'): stem_modules.append(model.conv_proj)
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'pos_embedding'): 
                stem_modules.append(model.encoder.pos_embedding)
            if hasattr(model, 'class_token'): 
                stem_modules.append(model.class_token)
            if stem_modules: blocks.append(stem_modules)
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
            items = block if isinstance(block, list) else [block]
            for item in items:
                params = item.named_parameters() if isinstance(item, nn.Module) else [("param", item)] if isinstance(item, nn.Parameter) else []
                for name, param in params:
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

    new_non_lora_groups = _build_non_lora_param_groups(model, config, base_lr)
    new_non_lora_count = 0
    for group in new_non_lora_groups:
        new_params = [param for param in group["params"] if param not in existing_params]
        if new_params:
            optimizer.add_param_group({'params': new_params, 'lr': group['lr']})
            new_non_lora_count += len(new_params)

    new_lora = []

    # Find any newly unwrapped LoRA params that are not managed by optimizer
    for name, param in model.named_parameters():
        if param.requires_grad and param not in existing_params and ".lora_" in name:
            new_lora.append(param)

    if new_non_lora_count:
        print(f"Update: Dynamically added {new_non_lora_count} un-frozen parameter tensors to the optimizer!")

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

def get_trainable_parameter_breakdown(model: nn.Module, model_type: str):
    head_prefixes = {
        "resnet": ("fc.",),
        "vit": ("head.", "heads."),
    }.get(model_type, tuple())

    breakdown = {
        "head": 0,
        "backbone": 0,
        "lora": 0,
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        num_params = param.numel()
        if ".lora_" in name:
            breakdown["lora"] += num_params
        elif any(name.startswith(prefix) for prefix in head_prefixes):
            breakdown["head"] += num_params
        else:
            breakdown["backbone"] += num_params

    return breakdown
