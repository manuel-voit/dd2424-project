import argparse
import copy
import os
import yaml


MODELS = ["vit_b_16"]
LEARNING_RATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
UNFREEZE_DEPTHS = [1, 2, 3, 4]
WEIGHT_DECAYS = [0.01, 0.05, 0.1]

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_WARMUP_EPOCHS = 1
DEFAULT_LLRD_DECAY = 0.7


def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 17 configurations.")
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
            for depth in UNFREEZE_DEPTHS:
                for wd in WEIGHT_DECAYS:
                    config = copy.deepcopy(template)

                    config["model"]["type"] = "vit"
                    config["model"]["name"] = model
                    config["model"]["num_classes"] = 37
                    config["model"]["fine_tuning"]["strategy"] = "simultaneous"
                    config["model"]["fine_tuning"]["num_layers"] = depth
                    config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 0

                    config["training"]["batch_size"] = DEFAULT_BATCH_SIZE
                    config["training"]["epochs"] = DEFAULT_EPOCHS
                    config["training"]["learning_rate"] = lr

                    config["optimizer"]["name"] = "adamw"
                    config["optimizer"]["lr"] = lr
                    config["optimizer"]["weight_decay"] = wd
                    config["optimizer"]["llrd"] = {
                        "enabled": True,
                        "decay": DEFAULT_LLRD_DECAY,
                    }
                    if "lora_lr" in config["optimizer"]:
                        del config["optimizer"]["lora_lr"]

                    config["scheduler"]["name"] = "cosine_annealing_with_linear_warmup"
                    config["scheduler"]["warmup_epochs"] = DEFAULT_WARMUP_EPOCHS

                    config["loss"]["name"] = "cross_entropy"
                    config["data"]["name"] = "oxford_pets"
                    config["data"]["data_dir"] = "./data"

                    if "lora" in config:
                        del config["lora"]

                    config["logging"]["experiment_name"] = "Exp17_Simultaneous_Unfreezing_ViT"
                    run_name = (
                        f"exp17_{model}_ft-simultaneous_nl{depth}_"
                        f"lr{lr:.6f}_l2-{wd}"
                    )
                    config["logging"]["run_name"] = run_name

                    file_path = os.path.join(active_dir, f"{run_name}.yaml")
                    with open(file_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                    generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 17 in {active_dir}")


if __name__ == "__main__":
    main()
