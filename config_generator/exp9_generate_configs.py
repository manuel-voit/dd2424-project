import argparse
import copy
import os
import yaml


MODELS = ["resnet50", "resnet101"]
RANKS = [1, 2, 4, 8, 16, 32]

# Update these once the preferred LoRA placement is finalized.
DEFAULT_TARGET_MODE = "general"  # "targeted" (3x3 convs) or "general" (all convs)
NUM_LAST_STAGES_OPTIONS = [1, 2]  # 1 -> layer4, 2 -> layer3+layer4

DEFAULT_LR = 0.005
DEFAULT_LORA_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_EPOCHS = 10

RESNET_STAGE_BLOCKS = {
    "resnet50": {"layer1": 3, "layer2": 4, "layer3": 6, "layer4": 3},
    "resnet101": {"layer1": 3, "layer2": 4, "layer3": 23, "layer4": 3},
}


def get_last_residual_stages(num_last_stages: int):
    ordered = ["layer4", "layer3", "layer2", "layer1"]
    return ordered[:num_last_stages]


def get_lora_targets(model_name: str, target_mode: str, num_last_stages: int):
    selected_stages = get_last_residual_stages(num_last_stages)
    if target_mode == "general":
        return selected_stages

    if target_mode != "targeted":
        raise ValueError(f"Unsupported target mode: {target_mode}")

    targets = []
    for stage in selected_stages:
        num_blocks = RESNET_STAGE_BLOCKS[model_name][stage]
        for block_idx in range(num_blocks):
            targets.append(f"{stage}.{block_idx}.conv2")
    return targets


def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 9 configurations.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (hardware dependent)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(project_root, "configs", "config_template.yaml")
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return

    with open(template_path, "r") as f:
        template = yaml.safe_load(f)

    active_dir = os.path.join(project_root, "configs", "active")
    os.makedirs(active_dir, exist_ok=True)

    generated_count = 0
    for model in MODELS:
        for num_last_stages in NUM_LAST_STAGES_OPTIONS:
            targets = get_lora_targets(model, DEFAULT_TARGET_MODE, num_last_stages)
            for rank in RANKS:
                config = copy.deepcopy(template)

                config["model"]["name"] = model
                config["model"]["fine_tuning"]["strategy"] = "none"
                config["model"]["fine_tuning"]["num_layers"] = 0
                config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 0

                config["training"]["batch_size"] = args.batch_size
                config["training"]["epochs"] = DEFAULT_EPOCHS
                config["training"]["learning_rate"] = DEFAULT_LR

                config["optimizer"]["lr"] = DEFAULT_LR
                config["optimizer"]["lora_lr"] = DEFAULT_LORA_LR
                config["optimizer"]["weight_decay"] = DEFAULT_WEIGHT_DECAY
                config["scheduler"]["name"] = "cosine_annealing_lr"

                config["lora"]["r"] = rank
                config["lora"]["alpha"] = 2 * rank
                config["lora"]["learning_rate"] = DEFAULT_LORA_LR
                config["lora"]["targets"] = targets
                config["lora"]["gradual_unfreeze"]["enabled"] = False
                config["lora"]["gradual_unfreeze"]["schedule"] = {}

                config["logging"]["experiment_name"] = "Exp9_LoRA_Varying_Rank"
                run_name = (
                    f"exp9_{model}_lora-{DEFAULT_TARGET_MODE}_"
                    f"nl{num_last_stages}_r{rank}"
                )
                config["logging"]["run_name"] = run_name

                file_path = os.path.join(active_dir, f"{run_name}.yaml")
                with open(file_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 9 in {active_dir}")


if __name__ == "__main__":
    main()
