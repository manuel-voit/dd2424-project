import argparse
import copy
import os
import yaml


MODELS = ["resnet50", "resnet101"]
TRAIN_FRACTIONS = [1.0, 0.1, 0.02]
AUGMENTATIONS = [True, False]
TARGET_MODES = ["targeted", "general"]
RANKS = [1, 8]

NUM_LAST_STAGES_OPTIONS = [1, 2]  # 1 -> layer4, 2 -> layer3+layer4

DEFAULT_LR = 0.005
DEFAULT_LORA_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_EPOCHS = 15

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
    parser = argparse.ArgumentParser(
        description="Generate Experiment 12 configurations for LoRA with limited training data."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (hardware dependent)",
    )
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
            for target_mode in TARGET_MODES:
                targets = get_lora_targets(model, target_mode, num_last_stages)
                for rank in RANKS:
                    for frac in TRAIN_FRACTIONS:
                        for aug in AUGMENTATIONS:
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

                            config["data"]["train_fraction"] = frac
                            config["data"]["augmentation"] = aug

                            config["lora"]["r"] = rank
                            config["lora"]["alpha"] = 2 * rank
                            config["lora"]["learning_rate"] = DEFAULT_LORA_LR
                            config["lora"]["targets"] = targets
                            config["lora"]["gradual_unfreeze"]["enabled"] = False
                            config["lora"]["gradual_unfreeze"]["schedule"] = {}

                            aug_str = "aug-on" if aug else "aug-off"
                            frac_str = f"frac-{frac:.2f}"
                            location_str = f"nl{num_last_stages}"

                            config["logging"]["experiment_name"] = "Exp12_LoRA_Limited_Train_Data"
                            run_name = (
                                f"exp12_{model}_lora-{target_mode}_{location_str}_"
                                f"r{rank}_{frac_str}_{aug_str}"
                            )
                            config["logging"]["run_name"] = run_name

                            file_path = os.path.join(active_dir, f"{run_name}.yaml")
                            with open(file_path, "w") as f:
                                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                            generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 12 in {active_dir}")


if __name__ == "__main__":
    main()
