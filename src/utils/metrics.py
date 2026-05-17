import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix
import numpy as np

class MetricTracker:
    def __init__(self):
        self.reset()

    # Reset tracker at the start of each epoch
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.running_loss = 0.0
        self.total_samples = 0
        self.correct_samples = 0

    # Track predictions and loss batch-wise
    def update(self, preds, labels, probs=None, loss_value=None):
        self.all_preds.append(preds.detach().cpu())
        self.all_labels.append(labels.detach().cpu())
            
        if probs is not None:
            self.all_probs.append(probs.detach().cpu())

        batch_size = labels.size(0)
        self.total_samples += batch_size
        
        if loss_value is not None:
            self.running_loss += loss_value * batch_size

        if len(labels.shape) > 1 and labels.shape[1] > 1:
            self.correct_samples += (preds.detach().cpu() == labels.detach().cpu()).all(dim=1).sum().item()
        else:
            self.correct_samples += preds.detach().cpu().eq(labels.detach().cpu()).sum().item()

    def get_running_accuracy(self):
        if self.total_samples == 0:
            return 0.0
        return self.correct_samples / self.total_samples

    def get_batch_accuracy(self, preds, labels):
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            # Multi-label subset accuracy (every label has to match for the prediction to be correct)
            correct_samples = (preds == labels).all(dim=1).sum().item()
            total_samples = labels.size(0)
        else:
            # Multi-class
            correct_samples = preds.eq(labels).sum().item()
            total_samples = labels.numel()    
        return correct_samples / total_samples

    # Computes final metrics at the end of the epoch
    def compute_epoch_metrics(self):
        preds = torch.cat(self.all_preds).numpy()
        labels = torch.cat(self.all_labels).numpy()

        is_multilabel = len(labels.shape) > 1 and labels.shape[1] > 1

        if is_multilabel:
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
        if self.all_probs and is_multilabel:
            probs = torch.cat(self.all_probs).numpy()
            metrics["mAP"] = average_precision_score(labels, probs, average='macro')
            per_class_ap = average_precision_score(labels, probs, average=None)

        # Calculate per class metrics
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        per_class_prec = precision_score(labels, preds, average=None, zero_division=0)
        per_class_rec = recall_score(labels, preds, average=None, zero_division=0)

        for class_idx in range(len(per_class_f1)):
            metrics[f"f1_class_{class_idx}"] = float(per_class_f1[class_idx])
            metrics[f"precision_class_{class_idx}"] = float(per_class_prec[class_idx])
            metrics[f"recall_class_{class_idx}"] = float(per_class_rec[class_idx])

            if self.all_probs and is_multilabel:
                metrics[f"ap_class_{class_idx}"] = float(per_class_ap[class_idx])

        if self.total_samples > 0:
            metrics["loss"] = self.running_loss / self.total_samples

        return metrics