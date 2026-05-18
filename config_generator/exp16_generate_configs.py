import argparse
import copy
import os
import yaml


MODELS = ["vit_b_16"]
LEARNING_RATES = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_WEIGHT_DECAY = 0.01


def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 16 configurations.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
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
        for lr in LEARNING_RATES:
            config = copy.deepcopy(template)

            config["model"]["type"] = "vit"
            config["model"]["name"] = model
            config["model"]["num_classes"] = 37
            config["model"]["fine_tuning"]["strategy"] = "none"
            config["model"]["fine_tuning"]["num_layers"] = 0
            config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 0

            config["training"]["batch_size"] = DEFAULT_BATCH_SIZE
            config["training"]["epochs"] = args.epochs
            config["training"]["learning_rate"] = lr

            config["optimizer"]["name"] = "adamw"
            config["optimizer"]["lr"] = lr
            config["optimizer"]["weight_decay"] = DEFAULT_WEIGHT_DECAY
            if "lora_lr" in config["optimizer"]:
                del config["optimizer"]["lora_lr"]
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

            if "lora" in config:
                del config["lora"]

            config["logging"]["experiment_name"] = "Exp16_Linear_Probing_ViT"
            run_name = f"exp16_{model}_ft-none_lr{lr:.6f}_l2-{DEFAULT_WEIGHT_DECAY}"
            config["logging"]["run_name"] = run_name

            file_path = os.path.join(active_dir, f"{run_name}.yaml")
            with open(file_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 16 in {active_dir}")


if __name__ == "__main__":
    main()
