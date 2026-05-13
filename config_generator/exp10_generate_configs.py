import argparse
import copy
import os
import yaml


MODELS = ["resnet50", "resnet101"]
NUM_LAST_STAGES = [1, 2, 3, 4]
TARGET_MODES = ["targeted", "general"]

# Update once Exp9 determines the preferred rank.
DEFAULT_BEST_RANK = 8
DEFAULT_LR = 0.0001
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
    parser = argparse.ArgumentParser(description="Generate Experiment 10 configurations.")
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
        for num_last_stages in NUM_LAST_STAGES:
            for target_mode in TARGET_MODES:
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

                config["lora"]["r"] = DEFAULT_BEST_RANK
                config["lora"]["alpha"] = 2 * DEFAULT_BEST_RANK
                config["lora"]["learning_rate"] = DEFAULT_LORA_LR
                config["lora"]["targets"] = get_lora_targets(model, target_mode, num_last_stages)
                config["lora"]["gradual_unfreeze"]["enabled"] = False
                config["lora"]["gradual_unfreeze"]["schedule"] = {}

                config["logging"]["experiment_name"] = "Exp10_LoRA_Location_Study"
                run_name = (
                    f"exp10_{model}_lora-{target_mode}_"
                    f"nl{num_last_stages}_r{DEFAULT_BEST_RANK}"
                )
                config["logging"]["run_name"] = run_name

                file_path = os.path.join(active_dir, f"{run_name}.yaml")
                with open(file_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 10 in {active_dir}")


if __name__ == "__main__":
    main()
