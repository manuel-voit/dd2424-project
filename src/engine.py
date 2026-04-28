import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.metrics import MetricTracker

def train_one_epoch(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device
):
    model.train()
    tracker = MetricTracker()

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
        _, predicted = outputs.max(1)
        tracker.update(predicted, labels, loss.item())

        # Update progress bar
        batch_acc = tracker.get_batch_accuracy(predicted, labels)
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}", 
            'acc': f"{batch_acc*100:.1f}%"
        })

    return tracker.compute_epoch_metrics()

@torch.no_grad()
def evaluate(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    device: torch.device
):
    model.eval()
    tracker = MetricTracker()

    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, predicted = outputs.max(1)
        tracker.update(predicted, labels, loss.item())

        batch_acc = tracker.get_batch_accuracy(predicted, labels)
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{batch_acc*100:.1f}%"})

    return tracker.compute_epoch_metrics()