import copy
import os
import yaml


MODELS = ["vit_b_16"]
UNFREEZE_DEPTHS = [1, 2, 3, 4]
UNFREEZE_FREQUENCIES = [1, 2, 3]

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_WARMUP_EPOCHS = 1
DEFAULT_LLRD_DECAY = 0.7

# Update these once Exp17 identifies the best simultaneous-unfreezing setup.
DEFAULT_BEST_LR = 0.0003
DEFAULT_BEST_WEIGHT_DECAY = 0.01


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
        for depth in UNFREEZE_DEPTHS:
            for ufe in UNFREEZE_FREQUENCIES:
                config = copy.deepcopy(template)

                config["model"]["type"] = "vit"
                config["model"]["name"] = model
                config["model"]["num_classes"] = 37
                config["model"]["fine_tuning"]["strategy"] = "gradual"
                config["model"]["fine_tuning"]["num_layers"] = depth
                config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = ufe

                config["training"]["batch_size"] = DEFAULT_BATCH_SIZE
                config["training"]["epochs"] = DEFAULT_EPOCHS
                config["training"]["learning_rate"] = DEFAULT_BEST_LR

                config["optimizer"]["name"] = "adamw"
                config["optimizer"]["lr"] = DEFAULT_BEST_LR
                config["optimizer"]["weight_decay"] = DEFAULT_BEST_WEIGHT_DECAY
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

                config["logging"]["experiment_name"] = "Exp18_Gradual_Unfreezing_ViT"
                run_name = (
                    f"exp18_{model}_ft-gradual_nl{depth}_ufe{ufe}_"
                    f"lr{DEFAULT_BEST_LR:.6f}_l2-{DEFAULT_BEST_WEIGHT_DECAY}"
                )
                config["logging"]["run_name"] = run_name

                file_path = os.path.join(active_dir, f"{run_name}.yaml")
                with open(file_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 18 in {active_dir}")


if __name__ == "__main__":
    main()
