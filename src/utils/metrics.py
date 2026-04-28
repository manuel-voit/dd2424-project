import torch
from sklearn.metrics import accuracy_score, f1_score

class MetricTracker:
    def __init__(self):
        self.reset()

    # Reset tracker at the start of each epoch
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.running_loss = 0.0
        self.total_samples = 0

    # Track predictions and loss batch-wise
    def update(self, preds, labels, loss_value=None):
        self.all_preds.append(preds.detach().cpu())
        self.all_labels.append(labels.detach().cpu())
        
        batch_size = labels.size(0)
        self.total_samples += batch_size
        
        if loss_value is not None:
            self.running_loss += loss_value * batch_size

    def get_batch_accuracy(self, preds, labels):
        correct = preds.eq(labels).sum().item()
        return correct / labels.size(0)

    # Computes final metrics at the end of the epoch
    def compute_epoch_metrics(self):
        preds = torch.cat(self.all_preds).numpy()
        labels = torch.cat(self.all_labels).numpy()

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average='macro', zero_division=0)
        }

        # Add per-class F1 scores
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        for class_idx, score in enumerate(per_class_f1):
            metrics[f"f1_class_{class_idx}"] = float(score)

        if self.total_samples > 0:
            metrics["loss"] = self.running_loss / self.total_samples

        return metrics