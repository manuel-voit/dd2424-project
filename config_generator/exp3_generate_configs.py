import os
import yaml
import argparse
import copy
import random

def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 3 configurations.")
    parser.add_argument("--batch-size", type=int, default=128, 
                        help="Batch size (hardware dependent)")
    args = parser.parse_args()

    # Determine project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load template
    template_path = os.path.join(project_root, "configs", "config_template.yaml")
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return

    with open(template_path, "r") as f:
        template = yaml.safe_load(f)

    active_dir = os.path.join(project_root, "configs", "active")
    os.makedirs(active_dir, exist_ok=True)

    models = ["resnet50", "resnet101"]
    
    weight_decay = [0.00005, 0.0001, 0.0005]

    nr_of_epochs = [10, 20]
    
    # LRs
    lrs = [0.01, 0.0075, 0.005]
    
    generated_count = 0
    for model in models:
        for l2 in weight_decay:
            for epochs in nr_of_epochs:
                for lr in lrs:
                    config = copy.deepcopy(template)
                    
                    config["model"]["name"] = model
                    config["training"]["learning_rate"] = lr
                    config["optimizer"]["lr"] = lr
                    config["optimizer"]["weight_decay"] = l2
                    config["training"]["epochs"] = epochs
                    config["training"]["batch_size"] = args.batch_size
                    
                    config["model"]["fine_tuning"]["strategy"] = "none"
                    config["model"]["fine_tuning"]["num_layers"] = 0
                    config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 0

                    # Strip LoRA configs for baselines
                    if "lora" in config:
                        del config["lora"]
                    if "lora_lr" in config["optimizer"]:
                        del config["optimizer"]["lora_lr"]
                    
                    # Logging & File Naming
                    config["logging"]["experiment_name"] = "Exp3_LinearProbing_WeightDecay_and_Epochs"
                    run_name = f"exp3_{model}_ft-none_lr{lr:.6f}_l2-{l2}_epochs{epochs}"
                    config["logging"]["run_name"] = run_name
                    
                    file_path = os.path.join(active_dir, f"{run_name}.yaml")
                    
                    with open(file_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        
                    generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 3 in {active_dir}")

if __name__ == "__main__":
    main()
