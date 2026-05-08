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
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            predicted = (torch.sigmoid(outputs) > 0.5).float()
        else:
            _, predicted = outputs.max(1)
        tracker.update(predicted, labels, loss.item())

        # Update progress bar
        running_acc = tracker.get_running_accuracy()
        running_loss = tracker.running_loss / max(tracker.total_samples, 1)
        progress_bar.set_postfix({
            'loss': f"{running_loss:.4f}", 
            'acc': f"{running_acc*100:.1f}%"
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

        # Check if multi-label or multi-class
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            predicted = (torch.sigmoid(outputs) > 0.5).float()
        else:
            _, predicted = outputs.max(1)
        tracker.update(predicted, labels, loss.item())

        running_acc = tracker.get_running_accuracy()
        running_loss = tracker.running_loss / max(tracker.total_samples, 1)
        progress_bar.set_postfix({'loss': f"{running_loss:.4f}", 'acc': f"{running_acc*100:.1f}%"})

    return tracker.compute_epoch_metrics()