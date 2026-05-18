import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LoRAMultiheadAttention(nn.Module):
    def __init__(self, original_layer: nn.MultiheadAttention, r: int = 4, alpha: int = 8):
        super().__init__()
        self.original_layer = original_layer

        for param in self.original_layer.parameters():
            param.requires_grad = False

        if original_layer._qkv_same_embed_dim is False:
            raise ValueError("LoRAMultiheadAttention currently supports only same-dim Q/K/V projections.")

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        embed_dim = original_layer.embed_dim
        self.lora_q_A = nn.Linear(embed_dim, r, bias=False)
        self.lora_q_B = nn.Linear(r, embed_dim, bias=False)
        self.lora_v_A = nn.Linear(embed_dim, r, bias=False)
        self.lora_v_B = nn.Linear(r, embed_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B.weight)

    def _delta_weight(self, lora_a: nn.Linear, lora_b: nn.Linear):
        return torch.matmul(lora_b.weight, lora_a.weight) * self.scaling

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if not self.original_layer._qkv_same_embed_dim:
            raise ValueError("LoRAMultiheadAttention requires same-dim Q/K/V.")

        if self.original_layer.batch_first and query.dim() == 3:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            restore_batch_first = True
        else:
            restore_batch_first = False

        in_proj_weight = self.original_layer.in_proj_weight
        q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)

        q_proj_weight = q_weight + self._delta_weight(self.lora_q_A, self.lora_q_B)
        v_proj_weight = v_weight + self._delta_weight(self.lora_v_A, self.lora_v_B)

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.original_layer.embed_dim,
            num_heads=self.original_layer.num_heads,
            in_proj_weight=None,
            in_proj_bias=self.original_layer.in_proj_bias,
            bias_k=self.original_layer.bias_k,
            bias_v=self.original_layer.bias_v,
            add_zero_attn=self.original_layer.add_zero_attn,
            dropout_p=self.original_layer.dropout,
            out_proj_weight=self.original_layer.out_proj.weight,
            out_proj_bias=self.original_layer.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_weight,
            v_proj_weight=v_proj_weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if restore_batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights

def _inject_lora_recursive(module: nn.Module, target_layer_names: list, r: int, alpha: int, prefix: str = ""):
    replaced_layers = []

    for child_name, child_module in module.named_children():
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        # Check if this is a replaceable leaf layer
        is_linear = isinstance(child_module, nn.Linear)
        is_conv = isinstance(child_module, nn.Conv2d)
        is_mha = isinstance(child_module, nn.MultiheadAttention)

        if is_linear or is_conv or is_mha:
            # Substring matching: Check if ANY of our targets are inside the full name
            if any(target in full_name for target in target_layer_names):
                
                if is_linear:
                    setattr(module, child_name, LoRALinear(child_module, r=r, alpha=alpha))
                elif is_conv:
                    setattr(module, child_name, LoRAConv2d(child_module, r=r, alpha=alpha))
                elif is_mha:
                    setattr(module, child_name, LoRAMultiheadAttention(child_module, r=r, alpha=alpha))
                
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
