import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
):
    model.train()
    running_loss = 0.0
    correct = 0
    total_images = 0

    # tqdm for progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_images += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'acc': f"{100. * correct / total_images:.2f}%"
        })

    epoch_loss = running_loss / total_images
    epoch_acc = correct / total_images
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    device: torch.device
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_images = 0

    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_images += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total_images
    epoch_acc = correct / total_images
    return epoch_loss, epoch_acc