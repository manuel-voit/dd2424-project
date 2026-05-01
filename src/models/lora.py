import torch
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

    @property
    def weight(self):
        delta_weight = torch.matmul(self.lora_B.weight, self.lora_A.weight) * self.scaling
        return self.original_layer.weight + delta_weight

    @property
    def bias(self):
        return self.original_layer.bias

    def reset_parameters(self):
        # Initialize A randomly and B as zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return original_output + lora_output


class LoRAConv2d(nn.Module):
    def __init__(self, original_layer: nn.Conv2d, r: int = 4, alpha: int = 8):
        super().__init__()
        self.original_layer = original_layer

        # Freeze the original pre-trained weights
        for param in self.original_layer.parameters():
            param.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Match the original spatial op in A
        self.lora_A = nn.Conv2d(
            in_channels=original_layer.in_channels,
            out_channels=r,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            dilation=original_layer.dilation,
            groups=original_layer.groups,
            bias=False
        )
        # Project back with a 1x1 conv in B
        self.lora_B = nn.Conv2d(
            in_channels=r,
            out_channels=original_layer.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.reset_parameters()

    @property
    def weight(self):
        delta_weight = torch.einsum(
            "orxy,rihw->oihw",
            self.lora_B.weight,
            self.lora_A.weight
        ) * self.scaling
        return self.original_layer.weight + delta_weight

    @property
    def bias(self):
        return self.original_layer.bias

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return original_output + lora_output

def _inject_lora_recursive(module: nn.Module, target_layer_names: list, r: int, alpha: int, prefix: str = ""):
    replaced_layers = []

    for child_name, child_module in module.named_children():
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        # Check if this is a replaceable leaf layer
        is_linear = isinstance(child_module, nn.Linear)
        is_conv = isinstance(child_module, nn.Conv2d)

        if is_linear or is_conv:
            # Substring matching: Check if ANY of our targets are inside the full name
            if any(target in full_name for target in target_layer_names):
                
                if is_linear:
                    setattr(module, child_name, LoRALinear(child_module, r=r, alpha=alpha))
                elif is_conv:
                    setattr(module, child_name, LoRAConv2d(child_module, r=r, alpha=alpha))
                
                replaced_layers.append(full_name)
        else:
            # Recurse deeper if it's not a leaf layer
            replaced_layers.extend(
                _inject_lora_recursive(child_module, target_layer_names, r, alpha, full_name)
            )

    return replaced_layers


def inject_lora(model: nn.Module, target_layer_names: list, r: int = 4, alpha: int = 8):
    if r <= 0:
        raise ValueError("LoRA rank 'r' must be a positive integer.")

    replaced_layers = _inject_lora_recursive(model, target_layer_names, r, alpha)
    if not replaced_layers:
        raise ValueError(
            f"No layers matched the configured LoRA targets: {target_layer_names}"
        )

    print(f"Injected LoRA into {len(replaced_layers)} layer(s): {', '.join(replaced_layers)}")
    return model
