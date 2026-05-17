import torch
import torch.nn as nn
import time
from tqdm import tqdm

from src.utils.metrics import MetricTracker
from src.utils.training_setup import set_batchnorm_mode


def _synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

def train_one_epoch(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    measure_compute_time: bool = False,
):
    model.train()
    set_batchnorm_mode(model)
    tracker = MetricTracker()
    compute_time_seconds = 0.0

    # tqdm for progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if measure_compute_time:
            _synchronize_device(device)
            batch_compute_start = time.perf_counter()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if measure_compute_time:
            _synchronize_device(device)
            compute_time_seconds += time.perf_counter() - batch_compute_start

        # Track metrics
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            tracker.update(preds=predicted, labels=labels, probs=probs, loss_value=loss.item())
        else:
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            tracker.update(preds=predicted, labels=labels, probs=probs, loss_value=loss.item())

        # Update progress bar
        running_acc = tracker.get_running_accuracy()
        running_loss = tracker.running_loss / max(tracker.total_samples, 1)
        progress_bar.set_postfix({
            'loss': f"{running_loss:.4f}", 
            'acc': f"{running_acc*100:.1f}%"
        })

    metrics = tracker.compute_epoch_metrics()
    if measure_compute_time:
        metrics["compute_time_seconds"] = compute_time_seconds
    return metrics

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
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            tracker.update(preds=predicted, labels=labels, probs=probs, loss_value=loss.item())
        else:
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            tracker.update(preds=predicted, labels=labels, probs=probs, loss_value=loss.item())

        running_acc = tracker.get_running_accuracy()
        running_loss = tracker.running_loss / max(tracker.total_samples, 1)
        progress_bar.set_postfix({'loss': f"{running_loss:.4f}", 'acc': f"{running_acc*100:.1f}%"})

    return tracker.compute_epoch_metrics()
