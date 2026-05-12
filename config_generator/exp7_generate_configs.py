import os
import yaml
import argparse
import copy

def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 7 configurations.")
    parser.add_argument("--batch-size", type=int, default=128, 
                        help="Batch size (hardware dependent)")
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

    models = ["resnet50", "resnet101"]
    
    best_lr = 0.005
    best_l2 = 0.0001
    epochs = 10
    
    combinations = [
        (False, False, False, "balanced_baseline"),
        (True,  False, False, "imbalanced_none"),
        (True,  True,  False, "imbalanced_oversampling"),
        (True,  False, True,  "imbalanced_weightedloss"),
        (True,  True,  True,  "imbalanced_both")
    ]

    generated_count = 0
    for model in models:
        for imb, osamp, wloss, setup_name in combinations:
            config = copy.deepcopy(template)
            
            config["model"]["name"] = model
            
            # Lock in Linear Probing parameters
            config["training"]["learning_rate"] = best_lr
            config["optimizer"]["lr"] = best_lr
            config["optimizer"]["weight_decay"] = best_l2
            config["training"]["epochs"] = epochs
            config["training"]["batch_size"] = args.batch_size
            
            config["model"]["fine_tuning"]["strategy"] = "none"
            config["model"]["fine_tuning"]["num_layers"] = 0
            config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 0

            if "data" not in config:
                config["data"] = {}
                
            config["data"]["imbalance"] = imb
            config["data"]["oversampling"] = osamp
            config["training"]["weighted_loss"] = wloss

            # Strip LoRA configs for baselines
            if "lora" in config:
                del config["lora"]
            if "lora_lr" in config["optimizer"]:
                del config["optimizer"]["lora_lr"]
            
            # Logging & File Naming
            config["logging"]["experiment_name"] = "Exp7_ClassImbalance_and_Countermeasures"
            run_name = f"exp7_{model}_ft-none_{setup_name}"
            config["logging"]["run_name"] = run_name
            
            file_path = os.path.join(active_dir, f"{run_name}.yaml")
            
            with open(file_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
            generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 7 in {active_dir}")

if __name__ == "__main__":
    main()