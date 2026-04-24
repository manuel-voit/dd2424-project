import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import datetime

import argparse
import yaml
# UI for Visualization
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from src.models.cnn_backbone import get_binary_resnet
from src.models.vit_backbone import get_binary_swin_t
from src.models.lora import inject_lora

from src.engine import train_one_epoch, evaluate
from src.data.data_loader import get_pet_dataloaders


def set_seed(seed=42):
    """Ensures our ResNet vs ViT comparisons are completely fair and reproducible."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # Setup
    set_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    parser = argparse.ArgumentParser(description="Train CNN/ViT networs")
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Extract config variables
    MODEL_TYPE = config['model']['type']
    NUM_CLASSES = config['model']['num_classes']

    BATCH_SIZE = config['training']['batch_size']
    EPOCHS = config['training']['epochs']
    LEARNING_RATE = config['training']['learning_rate']

    DATA_DIR = config['data']['data_dir']
    IMAGE_SIZE = config['data']['image_size']
    
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"runs/{MODEL_TYPE}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    # Load Data
    loaders = get_pet_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=config['data']['num_workers']
    )

    train_loader, val_loader, test_loader = loaders['binary']['train'], loaders['binary']['val'], loaders['binary']['test']

    # Model branching
    if MODEL_TYPE == "resnet":
        model = get_binary_resnet()
    elif MODEL_TYPE == "vit":
        model = get_binary_swin_t()
    
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
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    writer.close()

if __name__ == "__main__":
    main()