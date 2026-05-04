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
    update_optimizer
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
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Unfreeze layers if ft strat is gradual (no effect if simultaneous)
        unfroze_new = apply_finetuning_strategy(model, config, current_epoch=epoch)
        if unfroze_new:
            optimizer = update_optimizer(optimizer, model, config)
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
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
    if logger is not None:
        if logging_cfg.get('log_metrics', True):
            logger.log_scalars(test_log, step=EPOCHS)
        if logging_cfg.get('log_confusion_matrix', True):
            logger.log_confusion_matrix(cm, step=EPOCHS)

    if logger is not None and logging_cfg.get('log_artifact', True) and early_stopping is not None:
        # Save the best model checkpoint straight into MLflow!
        logger.log_artifact(early_stopping.save_path)
        logger.close()


if __name__ == "__main__":
    main()
