import torch
import argparse
import yaml
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Import custom modules
from src.models.cnn_backbone import get_resnet
from src.models.vit_backbone import get_swin
from src.models.lora import inject_lora

from src.data.data_loader import get_dataloaders

from src.engine import train_one_epoch, evaluate

from src.utils.seed import set_seed
from src.utils.mlflow_logger import MLflowLogger
from src.utils.loading import load_model_from_checkpoint
from src.utils.training_setup import (
    build_early_stopping,
    build_loss,
    build_optimizer,
    build_scheduler,
    apply_finetuning_strategy,
    update_optimizer,
    get_trainable_parameter_breakdown,
)


def main():
    parser = argparse.ArgumentParser(description="Train CNN/ViT networks")
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Setup
    seed = config['training'].get('seed', 42)
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {seed}")

    # Extract config variables
    MODEL_TYPE = config['model']['type']
    NUM_CLASSES = config['model']['num_classes']

    BATCH_SIZE = config['training']['batch_size']
    EPOCHS = config['training']['epochs']

    # Initialize tracking components
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logging_cfg = config.get('logging', {})
    checkpoint_dir = logging_cfg.get('checkpoint_dir', 'checkpoints')
    checkpoint_prefix = logging_cfg.get(
        'checkpoint_prefix', f"{MODEL_TYPE}_bs{BATCH_SIZE}_{current_time}"
    )
    checkpoint_path = f"{checkpoint_dir}/{checkpoint_prefix}.pth"

    early_stopping = build_early_stopping(config, checkpoint_path)
    
    logger = None
    if logging_cfg.get('enabled', True):
        logger = MLflowLogger(config=config)
    else:
        print("MLflow logging disabled.")
    measure_compute_time = logging_cfg.get('measure_compute_time', False)

    # Data loading
    loaders = get_dataloaders(config=config)
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    # Model branching
    MODEL_NAME = config['model'].get('name', None)

    if MODEL_TYPE == "resnet":
        model = get_resnet(num_classes=NUM_CLASSES, model_name=MODEL_NAME)
    elif MODEL_TYPE == "vit":
        model = get_swin(num_classes=NUM_CLASSES, model_name=MODEL_NAME)

    # Inject LoRA if defined in the config
    if 'lora' in config and config['lora']:
        print(f"Injecting LoRA (r={config['lora']['r']}) ...")
        model = inject_lora(
            model,
            target_layer_names=config['lora']['targets'],
            r=config['lora']['r'],
            alpha=config['lora']['alpha']
        )

    model = model.to(device)
    
    # Compute class weights if configured
    class_weights = None
    if config.get('data', {}).get('imbalance', {}).get('use_weighted_loss', False):
        
        train_dataset = train_loader.dataset
        # Extract targets for weights (assumes train_dataset is a Subset)
        if hasattr(train_dataset, 'dataset') and hasattr(train_dataset, 'indices'):
            base_ds = train_dataset.dataset
            indices = train_dataset.indices
            if NUM_CLASSES == 2 and hasattr(base_ds, '_bin_labels'):
                targets = [base_ds._bin_labels[i] for i in indices]
            elif hasattr(base_ds, '_labels'):
                targets = [base_ds._labels[i] for i in indices]
            else:
                targets = [train_dataset[i][1] for i in range(len(train_dataset))]
        else:
            targets = [train_dataset[i][1] for i in range(len(train_dataset))]
            
        targets = np.array(targets)
        weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Computed class weights for loss: {class_weights}")

    # Initial setup for fine-tuning before optimizer is built
    apply_finetuning_strategy(model, config, current_epoch=0)
    
    optimizer = build_optimizer(model, config)
    criterion = build_loss(config, class_weights=class_weights)
    scheduler, scheduler_needs_metric = build_scheduler(optimizer, config)

    # Split trainable parameters so LoRA adapters can use their own learning rate
    lora_params = []
    non_lora_trainable_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".lora_" in name:
            lora_params.append(param)
        else:
            non_lora_trainable_params.append(param)

    trainable_params = lora_params + non_lora_trainable_params
    
    # Print a quick sanity check to ensure parameter efficiency
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in trainable_params)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trained_params:,} ({100 * trained_params / total_params:.2f}%)")
    initial_trainable_params = trained_params
    initial_trainable_fraction = trained_params / total_params if total_params > 0 else 0.0
    initial_breakdown = get_trainable_parameter_breakdown(model, MODEL_TYPE)

    optimizer_cfg = config.get('optimizer', {})
    print(f"Optimizer: {optimizer_cfg.get('name', 'adamw')}")
    print(f"Loss: {config.get('loss', {}).get('name', 'cross_entropy')}")
    scheduler_cfg = config.get('scheduler', {})
    print(f"Scheduler: {scheduler_cfg.get('name', 'none')}")
    if non_lora_trainable_params:
        print(f"Head/Base trainable LR: {optimizer_cfg.get('lr', config['training']['learning_rate'])}")
    if lora_params:
        print(f"LoRA trainable LR: {optimizer_cfg.get('lora_lr', optimizer_cfg.get('lr', config['training']['learning_rate']))}")

    # Training Loop
    epoch_compute_times = []
    best_epoch = 1
    best_epoch_accuracy = 1
    best_epoch_f1_macro = 1
    best_val_loss = float('inf')
    best_val_accuracy = float('-inf')
    best_val_f1_macro = float('-inf')

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Unfreeze layers if ft strat is gradual (no effect if simultaneous)
        unfroze_new = apply_finetuning_strategy(model, config, current_epoch=epoch)
        if unfroze_new:
            optimizer = update_optimizer(optimizer, model, config)
            
            # Give scheduler awareness of the newly added parameter group learning rates (prevent bug where scheduler does not update LR of newly unfrozen params)
            if scheduler is not None and hasattr(scheduler, 'base_lrs'):
                new_lrs = [group['lr'] for group in optimizer.param_groups[len(scheduler.base_lrs):]]
                scheduler.base_lrs.extend(new_lrs)
        
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            measure_compute_time=measure_compute_time
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Track the best epoch independently
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_epoch_accuracy = epoch + 1
        if val_metrics['f1_macro'] > best_val_f1_macro:
            best_val_f1_macro = val_metrics['f1_macro']
            best_epoch_f1_macro = epoch + 1
            
        if measure_compute_time:
            epoch_compute_times.append(train_metrics.pop("compute_time_seconds"))
        
        # Check early stopping & save weights
        if scheduler is not None:
            if scheduler_needs_metric:
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        if early_stopping is not None:
            early_stopping(val_metrics['loss'], model)
            if early_stopping.early_stop:
                print("Early stopping triggered! Ending training.")
                break

        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']*100:.2f}% | Train F1: {train_metrics['f1_macro']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1:   {val_metrics['f1_macro']:.4f}")

        # Remove confusion matrices from logs to avoid crashing the logger
        train_metrics.pop("confusion_matrix", None)
        val_metrics.pop("confusion_matrix", None)

        train_log = {f"train_{k}": v for k, v in train_metrics.items()}
        val_log = {f"val_{k}": v for k, v in val_metrics.items()}

        if logger is not None:
            # Log to MLflow using a single dictionary call
            logger.log_scalars({**train_log, **val_log}, step=epoch)

    # Test evaluation should use the best checkpoint
    if early_stopping is not None:
        model, _ = load_model_from_checkpoint(early_stopping.save_path, device)

    # Test evaluation
    print("\nRunning test evaluation ...")
    test_metrics = evaluate(model, test_loader, criterion, device)
    cm = test_metrics.pop("confusion_matrix")
    print(f"Final Test Acc: {test_metrics['accuracy']*100:.2f}% | Final Test F1: {test_metrics['f1_macro']:.4f}")
    
    test_log = {f"test_{k}": v for k, v in test_metrics.items()}
    mean_epoch_compute_time = None
    total_training_compute_time = None
    if measure_compute_time:
        mean_epoch_compute_time = float(np.mean(epoch_compute_times)) if epoch_compute_times else 0.0
        total_training_compute_time = float(np.sum(epoch_compute_times)) if epoch_compute_times else 0.0
        print(f"Mean Training Compute Time / Epoch: {mean_epoch_compute_time:.4f}s")
        print(f"Total Training Compute Time: {total_training_compute_time:.4f}s")

    final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    final_trainable_fraction = final_trainable_params / total_params if total_params > 0 else 0.0
    final_breakdown = get_trainable_parameter_breakdown(model, MODEL_TYPE)

    peak_gpu_memory_allocated_mb = None
    peak_gpu_memory_reserved_mb = None
    if device.type == "cuda":
        peak_gpu_memory_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_gpu_memory_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f"Peak GPU Memory Allocated: {peak_gpu_memory_allocated_mb:.2f} MB")
        print(f"Peak GPU Memory Reserved:  {peak_gpu_memory_reserved_mb:.2f} MB")

    if logger is not None:
        if logging_cfg.get('log_metrics', True):
            final_log = dict(test_log)
            final_log["total_epochs_trained"] = epoch + 1
            final_log["epoch_with_best_val_loss"] = best_epoch
            final_log["epoch_with_best_val_accuracy"] = best_epoch_accuracy
            final_log["epoch_with_best_val_f1_macro"] = best_epoch_f1_macro
            final_log["best_val_accuracy"] = best_val_accuracy
            final_log["best_val_f1_macro"] = best_val_f1_macro
            final_log["model_total_params"] = total_params
            final_log["model_trainable_params_initial"] = initial_trainable_params
            final_log["model_trainable_fraction_initial"] = initial_trainable_fraction
            final_log["model_trainable_head_params"] = initial_breakdown["head"]
            final_log["model_trainable_backbone_params_initial"] = initial_breakdown["backbone"]
            final_log["model_trainable_lora_params_initial"] = initial_breakdown["lora"]
            final_log["model_trainable_params_final"] = final_trainable_params
            final_log["model_trainable_fraction_final"] = final_trainable_fraction
            final_log["model_trainable_backbone_params_final"] = final_breakdown["backbone"]
            final_log["model_trainable_lora_params_final"] = final_breakdown["lora"]
            if mean_epoch_compute_time is not None:
                final_log["train_mean_epoch_compute_time_seconds"] = mean_epoch_compute_time
            if total_training_compute_time is not None:
                final_log["train_total_compute_time_seconds"] = total_training_compute_time
            if peak_gpu_memory_allocated_mb is not None:
                final_log["train_peak_gpu_memory_allocated_mb"] = peak_gpu_memory_allocated_mb
            if peak_gpu_memory_reserved_mb is not None:
                final_log["train_peak_gpu_memory_reserved_mb"] = peak_gpu_memory_reserved_mb
            logger.log_scalars(final_log, step=EPOCHS)
        if logging_cfg.get('log_confusion_matrix', True):
            logger.log_confusion_matrix(cm, step=EPOCHS)

    if logger is not None and logging_cfg.get('log_artifact', True) and early_stopping is not None:
        # Save the best model checkpoint straight into MLflow!
        logger.log_artifact(early_stopping.save_path)
        logger.close()


if __name__ == "__main__":
    main()
