import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int = 4, alpha: int = 8):
        super().__init__()
        self.original_layer = original_layer
        
        # Freeze the original pre-trained weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Create the low-rank matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A randomly and B as zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Forward pass: Original path + LoRA path
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return original_output + lora_output
    
def inject_lora(model: nn.Module, target_layer_names: list, r: int = 4, alpha: int = 8):
    pass
