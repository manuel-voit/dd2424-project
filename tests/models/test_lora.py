import copy
from pathlib import Path

import torch
import torch.nn as nn
import pytest
import yaml

from src.models.cnn_backbone import get_resnet
from src.models.lora import LoRAConv2d, LoRALinear, inject_lora
from src.models.vit_backbone import get_swin
from src.utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_model(model_config: dict):
    model_type = model_config["type"]
    num_classes = model_config["num_classes"]
    model_name = model_config.get("name", None)

    if model_type == "resnet":
        return get_resnet(num_classes=num_classes, model_name=model_name or "resnet50")
    if model_type == "vit":
        return get_swin(num_classes=num_classes, model_name=model_name or "swin_t")

    raise ValueError(f"Unsupported model type: {model_type}")

def collect_replaced_modules(model: nn.Module):
    replaced = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            replaced[name] = type(module).__name__
    return replaced

def snapshot_params(model: nn.Module):
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}

def changed_param_names(before: dict, after_model: nn.Module):
    changed = []
    for name, param in after_model.named_parameters():
        if not torch.equal(before[name], param.detach().cpu()):
            changed.append(name)
    return sorted(changed)

def load_config(config_path: str):
    with open(PROJECT_ROOT / config_path, "r") as file:
        return yaml.safe_load(file)

@pytest.mark.parametrize(
    "config_path",
    [
        "configs/train_resnet_lora_general.yaml",
        "configs/train_resnet_lora_targeted.yaml",
        "configs/train_swin_lora_general.yaml",
        "configs/train_swin_lora_targeted.yaml",
    ],
)
def test_lora_injection_and_update_behavior(config_path: str):
    config = load_config(config_path)
    assert "lora" in config and config["lora"], "Test expects a config with a populated lora section."

    seed = config["training"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cpu")
    image_size = config["data"].get("image_size", 224)
    num_classes = config["model"]["num_classes"]
    target_layer_names = config["lora"]["targets"]
    rank = config["lora"]["r"]
    alpha = config["lora"]["alpha"]

    base_model = build_model(config["model"]).to(device)
    lora_model = copy.deepcopy(base_model)
    lora_model = inject_lora(
        lora_model,
        target_layer_names=target_layer_names,
        r=rank,
        alpha=alpha,
    ).to(device)

    replaced_modules = collect_replaced_modules(lora_model)
    assert replaced_modules, "LoRA injection reported success, but no LoRA modules were found afterwards."

    for module_name in replaced_modules:
        assert any(target in module_name for target in target_layer_names), (
            f"Injected module '{module_name}' does not match configured targets {target_layer_names}."
        )

    trainable_names = [name for name, param in lora_model.named_parameters() if param.requires_grad]
    assert trainable_names, "No trainable parameters remain after LoRA injection."
    assert any(".lora_" in name for name in trainable_names), "No LoRA parameters are marked trainable."

    x = torch.randn(2, 3, image_size, image_size, device=device)
    y = torch.randint(low=0, high=num_classes, size=(2,), device=device)

    base_model.eval()
    lora_model.eval()
    with torch.no_grad():
        base_out = base_model(x)
        lora_out = lora_model(x)

    assert torch.allclose(base_out, lora_out, atol=1e-6, rtol=1e-5), (
        "LoRA-wrapped model changed the forward pass before training."
    )

    params_before = snapshot_params(lora_model)

    optimizer = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    lora_model.train()
    optimizer.zero_grad()
    loss = criterion(lora_model(x), y)
    loss.backward()
    optimizer.step()

    changed_names = changed_param_names(params_before, lora_model)
    changed_frozen = [name for name in changed_names if not lora_model.get_parameter(name).requires_grad]
    changed_lora = [name for name in changed_names if ".lora_" in name]

    assert not changed_frozen, f"Frozen parameters changed unexpectedly: {changed_frozen}"
    assert changed_lora, "No LoRA parameters changed after the optimizer step."
    assert loss.item() > 0.0
