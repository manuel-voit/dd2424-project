import copy
import os
import yaml


MODELS = ["vit_b_16"]
TRAIN_FRACTIONS = [1.0, 0.1, 0.02]
AUGMENTATIONS = [True, False]

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.0003
DEFAULT_LINEAR_PROBING_LR = 0.01
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_EPOCHS = 1
DEFAULT_LLRD_DECAY = 0.7

STRATEGIES = [
    {
        "name": "linear",
        "ft_strategy": "none",
        "num_layers": 0,
        "unfreeze_every_n_epochs": 0,
    },
    {
        "name": "simultaneous",
        "ft_strategy": "simultaneous",
        "num_layers": 4,
        "unfreeze_every_n_epochs": 0,
    },
    {
        "name": "gradual",
        "ft_strategy": "gradual",
        "num_layers": 3,
        "unfreeze_every_n_epochs": 1,
    },
]


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
    for model in MODELS:
        for frac in TRAIN_FRACTIONS:
            for aug in AUGMENTATIONS:
                for strat in STRATEGIES:
                    config = copy.deepcopy(template)

                    config["model"]["type"] = "vit"
                    config["model"]["name"] = model
                    config["model"]["num_classes"] = 37
                    config["model"]["fine_tuning"]["strategy"] = strat["ft_strategy"]
                    config["model"]["fine_tuning"]["num_layers"] = strat["num_layers"]
                    config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = strat["unfreeze_every_n_epochs"]

                    config["training"]["batch_size"] = DEFAULT_BATCH_SIZE
                    config["training"]["epochs"] = DEFAULT_EPOCHS
                    run_lr = DEFAULT_LINEAR_PROBING_LR if strat["name"] == "linear" else DEFAULT_LR
                    config["training"]["learning_rate"] = run_lr

                    config["optimizer"]["name"] = "adamw"
                    config["optimizer"]["lr"] = run_lr
                    config["optimizer"]["weight_decay"] = DEFAULT_WEIGHT_DECAY
                    if strat["name"] == "linear":
                        if "llrd" in config["optimizer"]:
                            del config["optimizer"]["llrd"]
                    else:
                        config["optimizer"]["llrd"] = {
                            "enabled": True,
                            "decay": DEFAULT_LLRD_DECAY,
                        }
                    if "lora_lr" in config["optimizer"]:
                        del config["optimizer"]["lora_lr"]

                    if strat["name"] == "linear":
                        config["scheduler"]["name"] = "cosine_annealing_lr"
                        if "warmup_epochs" in config["scheduler"]:
                            del config["scheduler"]["warmup_epochs"]
                    else:
                        config["scheduler"]["name"] = "cosine_annealing_with_linear_warmup"
                        config["scheduler"]["warmup_epochs"] = DEFAULT_WARMUP_EPOCHS

                    config["loss"]["name"] = "cross_entropy"
                    config["data"]["name"] = "oxford_pets"
                    config["data"]["data_dir"] = "./data"
                    config["data"]["train_fraction"] = frac
                    config["data"]["augmentation"] = aug

                    if "lora" in config:
                        del config["lora"]

                    aug_str = "aug-on" if aug else "aug-off"
                    frac_str = f"frac-{frac:.2f}"

                    config["logging"]["experiment_name"] = "Exp19_Limited_Data_ViT"
                    run_name = (
                        f"exp19_{model}_{strat['name']}_{frac_str}_{aug_str}_"
                        f"lr{run_lr:.6f}_l2-{DEFAULT_WEIGHT_DECAY}"
                    )
                    config["logging"]["run_name"] = run_name

                    file_path = os.path.join(active_dir, f"{run_name}.yaml")
                    with open(file_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                    generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 19 in {active_dir}")


if __name__ == "__main__":
    main()
