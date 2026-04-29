import torch.nn as nn
import torch.optim as optim

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

def build_loss(config: dict):
    loss_cfg = config.get('loss', {})
    loss_name = loss_cfg.get('name', 'cross_entropy').lower()

    if loss_name == 'cross_entropy':
        kwargs = {}
        if 'label_smoothing' in loss_cfg:
            kwargs['label_smoothing'] = loss_cfg['label_smoothing']
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
