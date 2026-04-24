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
from src.utils.tensorboard_logger import TensorBoardLogger


def main():
    parser = argparse.ArgumentParser(description="Train CNN/ViT networks")
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Setup
    set_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Extract config variables
    MODEL_TYPE = config['model']['type']
    NUM_CLASSES = config['model']['num_classes']

    BATCH_SIZE = config['training']['batch_size']
    EPOCHS = config['training']['epochs']
    LEARNING_RATE = config['training']['learning_rate']

    # Initialize tracking components
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    early_stopping = EarlyStopping(
        config=config,
        patience=3, 
        save_path=f"checkpoints/{MODEL_TYPE}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_{current_time}.pth", 
    )
    
    logger = TensorBoardLogger(config=config)

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

    # Filter parameters for the Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Print a quick sanity check to ensure parameter efficiency
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in trainable_params)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trained_params:,} ({100 * trained_params / total_params:.2f}%)")

    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Check early stopping & save weights
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered! Ending training.")
            break

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Log to TensorBoard using grouped tags
        logger.log_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        logger.log_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

    # Test valuation
    print("\nRunning test evaluation ...")
    
    # Load the best weights before testing (in case early stopping triggered)
    model.load_state_dict(torch.load(early_stopping.save_path)['model_state_dict'], strict=False)
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Acc: {test_acc*100:.2f}%")
    
    # Log hyperparameters and final metrics
    logger.log_hparams(
        config=config, 
        final_metrics={
            'test_acc': test_acc
        },
        step=EPOCHS
    )

    logger.close()


if __name__ == "__main__":
    main()
