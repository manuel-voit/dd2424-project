import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix
import numpy as np

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
        return correct / labels.numel()

    # Computes final metrics at the end of the epoch
    def compute_epoch_metrics(self):
        preds = torch.cat(self.all_preds).numpy()
        labels = torch.cat(self.all_labels).numpy()

        if len(labels.shape) > 1 and labels.shape[1] > 1:
            # Multi-label
            cm = multilabel_confusion_matrix(labels, preds)
        else:
            # Multi-class
            cm = confusion_matrix(labels, preds)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average='macro', zero_division=0),
            "precision_macro": precision_score(labels, preds, average='macro', zero_division=0),
            "recall_macro": recall_score(labels, preds, average='macro', zero_division=0),
            "confusion_matrix": cm
        }

        # Calculate per class metrics
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        per_class_prec = precision_score(labels, preds, average=None, zero_division=0)
        per_class_rec = recall_score(labels, preds, average=None, zero_division=0)

        for class_idx in range(len(per_class_f1)):
            metrics[f"f1_class_{class_idx}"] = float(per_class_f1[class_idx])
            metrics[f"precision_class_{class_idx}"] = float(per_class_prec[class_idx])
            metrics[f"recall_class_{class_idx}"] = float(per_class_rec[class_idx])

        if self.total_samples > 0:
            metrics["loss"] = self.running_loss / self.total_samples

        return metrics