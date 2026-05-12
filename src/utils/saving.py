import os
import torch
import torch.nn as nn

from src.utils.loading import build_model_from_config


def save_trainable_parameters(model: nn.Module, config: dict, save_path: str):
    # Build a fresh base model and store only the state entries that changed
    # relative to that base. This keeps checkpoints compact while still
    # preserving buffers such as BatchNorm running stats.
    base_model = build_model_from_config(config)
    current_state = model.state_dict()
    base_state = base_model.state_dict()

    diff_state_dict = {}
    for name, tensor in current_state.items():
        base_tensor = base_state.get(name)
        if base_tensor is None or not torch.equal(tensor.detach().cpu(), base_tensor.detach().cpu()):
            diff_state_dict[name] = tensor.detach().cpu().clone()

    checkpoint = {
        'config': config,
        'model_state_dict': diff_state_dict,
        'state_dict_type': 'diff',
    }
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Saved compact diff checkpoint with {len(diff_state_dict)} changed state entries to {save_path}")

class EarlyStopping:
    def __init__(self, config: dict, patience: int = 3, min_delta: float = 0.0, save_path: str = "checkpoints/best_model.pth"):
        self.config = config
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def _save_checkpoint(self, model: nn.Module):
        save_trainable_parameters(model, self.config, self.save_path)

    def __call__(self, val_loss: float, model: nn.Module):
        # First epoch: set the baseline and save
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(model)
            
        # If the loss didn't improve
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            # Trigger the stop flag if we ran out of patience
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered!")
                
        # If the loss improved, reset the counter and save the new best weights
        else:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
