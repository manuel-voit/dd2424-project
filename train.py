import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import yaml
import datetime

# Import custom modules
from src.models.cnn_backbone import get_resnet
from src.models.vit_backbone import get_swin_t
from src.models.lora import inject_lora

from src.data.data_loader import get_dataloaders

from src.engine import train_one_epoch, evaluate

from src.utils.seed import set_seed
from src.utils.saving import EarlyStopping
from src.utils.mlflow_logger import MLflowLogger
from src.utils.loading import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train CNN/ViT networks")
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    parser.add_argument('--disable-mlflow', action='store_true', help="Disable MLflow logging for this run")
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
    LEARNING_RATE = config['training']['learning_rate']
    LORA_LEARNING_RATE = config.get('lora', {}).get('learning_rate', LEARNING_RATE)

    # Initialize tracking components
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    early_stopping = EarlyStopping(
        config=config,
        patience=3, 
        save_path=f"checkpoints/{MODEL_TYPE}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_{current_time}.pth", 
    )
    
    logger = None
    if args.disable_mlflow:
        print("MLflow logging disabled.")
    else:
        logger = MLflowLogger(config=config, experiment_name="Transfer_Learning")

    # Data loading
    loaders = get_dataloaders(config=config)
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    # Model branching
    if MODEL_TYPE == "resnet":
        model = get_resnet(num_classes=NUM_CLASSES)
    elif MODEL_TYPE == "vit":
        model = get_swin_t(num_classes=NUM_CLASSES)

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

    optimizer_param_groups = []
    if non_lora_trainable_params:
        optimizer_param_groups.append({
            'params': non_lora_trainable_params,
            'lr': LEARNING_RATE
        })
    if lora_params:
        optimizer_param_groups.append({
            'params': lora_params,
            'lr': LORA_LEARNING_RATE
        })

    print(f"Head/Base trainable LR: {LEARNING_RATE}")
    if lora_params:
        print(f"LoRA trainable LR: {LORA_LEARNING_RATE}")

    optimizer = optim.AdamW(optimizer_param_groups)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Check early stopping & save weights
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered! Ending training.")
            break

        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']*100:.2f}% | Train F1: {train_metrics['f1_macro']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1:   {val_metrics['f1_macro']:.4f}")

        train_log = {f"train_{k}": v for k, v in train_metrics.items()}
        val_log = {f"val_{k}": v for k, v in val_metrics.items()}

        if logger is not None:
            # Log to MLflow using a single dictionary call
            logger.log_scalars({**train_log, **val_log}, step=epoch)

    # Test evaluation should use the best checkpoint
    model, _ = load_model_from_checkpoint(early_stopping.save_path, device)

    # Test evaluation
    print("\nRunning test evaluation ...")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Acc: {test_metrics['accuracy']*100:.2f}% | Final Test F1: {test_metrics['f1_macro']:.4f}")
    
    test_log = {f"test_{k}": v for k, v in test_metrics.items()}
    if logger is not None:
        logger.log_scalars(test_log, step=EPOCHS)

    if logger is not None:
        # Save the best model checkpoint straight into MLflow!
        logger.log_artifact(early_stopping.save_path)
        logger.close()


if __name__ == "__main__":
    main()
