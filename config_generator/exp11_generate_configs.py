import argparse
import copy
import os
import yaml


MODELS = ["resnet50", "resnet101"]
UNFREEZE_FREQUENCIES = [1, 2, 3]
BLOCKS_TO_UNFREEZE = {
    "resnet50": [1, 2, 3, 4],
    "resnet101": [1, 2, 4, 8],
}

# Update these once Exp9/10 determine the preferred settings.
DEFAULT_BEST_RANK = 8
TARGET_MODES = ["targeted", "general"]
DEFAULT_LR = 0.005
DEFAULT_LORA_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_EPOCHS = 15

RESNET_STAGE_BLOCKS = {
    "resnet50": {"layer3": 6, "layer4": 3},
    "resnet101": {"layer3": 23, "layer4": 3},
}


def get_stage34_targets(model_name: str, target_mode: str):
    if target_mode == "general":
        return ["layer3", "layer4"]

    if target_mode != "targeted":
        raise ValueError(f"Unsupported target mode: {target_mode}")

    targets = []
    for stage in ["layer3", "layer4"]:
        num_blocks = RESNET_STAGE_BLOCKS[model_name][stage]
        for block_idx in range(num_blocks):
            targets.append(f"{stage}.{block_idx}.conv2")
    return targets


def get_reverse_stage34_blocks(model_name: str):
    blocks = []
    for stage in ["layer4", "layer3"]:
        for block_idx in reversed(range(RESNET_STAGE_BLOCKS[model_name][stage])):
            blocks.append(f"{stage}.{block_idx}")
    return blocks


def build_gradual_schedule(
    model_name: str,
    blocks_per_step: int,
    unfreeze_frequency: int,
    max_epochs: int,
):
    ordered_blocks = get_reverse_stage34_blocks(model_name)
    schedule = {}

    # Epoch 0 remains head-only. Starting at `unfreeze_frequency`, add the next
    # `blocks_per_step` residual blocks each time.
    for step_idx, start_idx in enumerate(range(0, len(ordered_blocks), blocks_per_step), start=1):
        epoch = step_idx * unfreeze_frequency
        if epoch >= max_epochs:
            break
        schedule[epoch] = ordered_blocks[start_idx:start_idx + blocks_per_step]

    return schedule


def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 11 configurations.")
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
        for num_blocks in BLOCKS_TO_UNFREEZE[model]:
            for unfreeze_frequency in UNFREEZE_FREQUENCIES:
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
                    config["lora"]["targets"] = get_stage34_targets(model, target_mode)
                    config["lora"]["gradual_unfreeze"]["enabled"] = True
                    config["lora"]["gradual_unfreeze"]["schedule"] = build_gradual_schedule(
                        model,
                        num_blocks,
                        unfreeze_frequency,
                        config["training"]["epochs"],
                    )

                    config["logging"]["experiment_name"] = "Exp11_LoRA_Gradual_Unfreezing"
                    run_name = (
                        f"exp11_{model}_lora-{target_mode}_"
                        f"stage34_r{DEFAULT_BEST_RANK}_nblocks{num_blocks}_ufe{unfreeze_frequency}"
                    )
                    config["logging"]["run_name"] = run_name

                    file_path = os.path.join(active_dir, f"{run_name}.yaml")
                    with open(file_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                    generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 11 in {active_dir}")


if __name__ == "__main__":
    main()
