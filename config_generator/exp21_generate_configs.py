import copy
import os
import yaml


TRAIN_FRACTIONS = [1.0, 0.1, 0.02]
AUGMENTATIONS = [True, False]

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_HEAD_LR = 0.01
DEFAULT_WEIGHT_DECAY = 0.01

# Best-performing Exp20 setups:
# - r=1 with 4 stages
# - r=8 with 1 stage
# - r=8 with 4 stages
MODEL_NAME = "vit_b_16"
LORA_SETUPS = [
    {
        "target_mode": "attention_qv",
        "num_last_stages": 4,
        "rank": 1,
        "lora_lr": 1e-3,
    },
    {
        "target_mode": "attention_qv",
        "num_last_stages": 1,
        "rank": 8,
        "lora_lr": 1e-3,
    },
    {
        "target_mode": "attention_qv",
        "num_last_stages": 4,
        "rank": 8,
        "lora_lr": 1e-3,
    },
]

VIT_B16_STAGE_BLOCKS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
]


def get_vit_b16_attention_targets(num_last_stages: int):
    selected_stages = VIT_B16_STAGE_BLOCKS[-num_last_stages:]
    targets = []
    for stage in selected_stages:
        for block_idx in stage:
            targets.append(f"encoder.layers.encoder_layer_{block_idx}.self_attention")
    return targets


def get_lora_targets(model_name: str, target_mode: str, num_last_stages: int):
    if model_name == "vit_b_16":
        if target_mode != "attention_qv":
            raise ValueError(f"Unsupported ViT-B/16 target mode: {target_mode}")
        return get_vit_b16_attention_targets(num_last_stages)

    raise ValueError(f"Unsupported model for Exp21: {model_name}")


def main():
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
    for setup in LORA_SETUPS:
        targets = get_lora_targets(
            model_name=MODEL_NAME,
            target_mode=setup["target_mode"],
            num_last_stages=setup["num_last_stages"],
        )

        for frac in TRAIN_FRACTIONS:
            for aug in AUGMENTATIONS:
                config = copy.deepcopy(template)

                config["model"]["type"] = "vit"
                config["model"]["name"] = MODEL_NAME
                config["model"]["num_classes"] = 37
                config["model"]["fine_tuning"]["strategy"] = "none"
                config["model"]["fine_tuning"]["num_layers"] = 0
                config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 0

                config["training"]["batch_size"] = DEFAULT_BATCH_SIZE
                config["training"]["epochs"] = DEFAULT_EPOCHS
                config["training"]["learning_rate"] = DEFAULT_HEAD_LR

                config["optimizer"]["name"] = "adamw"
                config["optimizer"]["lr"] = DEFAULT_HEAD_LR
                config["optimizer"]["lora_lr"] = setup["lora_lr"]
                config["optimizer"]["weight_decay"] = DEFAULT_WEIGHT_DECAY
                if "llrd" in config["optimizer"]:
                    del config["optimizer"]["llrd"]

                config["scheduler"]["name"] = "cosine_annealing_lr"
                if "warmup_epochs" in config["scheduler"]:
                    del config["scheduler"]["warmup_epochs"]
                if "warmup_start_factor" in config["scheduler"]:
                    del config["scheduler"]["warmup_start_factor"]
                if "eta_min" in config["scheduler"]:
                    del config["scheduler"]["eta_min"]

                config["loss"]["name"] = "cross_entropy"
                config["data"]["name"] = "oxford_pets"
                config["data"]["data_dir"] = "./data"
                config["data"]["train_fraction"] = frac
                config["data"]["augmentation"] = aug

                config["lora"]["r"] = setup["rank"]
                config["lora"]["alpha"] = 2 * setup["rank"]
                config["lora"]["learning_rate"] = setup["lora_lr"]
                config["lora"]["targets"] = targets
                config["lora"]["gradual_unfreeze"]["enabled"] = False
                config["lora"]["gradual_unfreeze"]["schedule"] = {}

                aug_str = "aug-on" if aug else "aug-off"
                frac_str = f"frac-{frac:.2f}"

                config["logging"]["experiment_name"] = "Exp21_Limited_Data_ViT_LoRA"
                run_name = (
                    f"exp21_{MODEL_NAME}_lora_nl{setup['num_last_stages']}_"
                    f"r{setup['rank']}_{frac_str}_{aug_str}"
                )
                config["logging"]["run_name"] = run_name

                file_path = os.path.join(active_dir, f"{run_name}.yaml")
                with open(file_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 21 in {active_dir}")


if __name__ == "__main__":
    main()
