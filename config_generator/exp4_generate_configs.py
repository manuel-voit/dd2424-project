import os
import yaml
import argparse
import copy

def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 4 configurations.")
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
    
    # Core of exp4
    num_layers_list = [1, 2, 3, 4]
    
    # Fine-tuning learning rates (smaller than linear probing)
    lrs = [0.0001, 0.00005]
    
    weight_decays = [0.00005, 0.0001, 0.0005]
    epochs = 10
    
    generated_count = 0
    for model in models:
        for num_layers in num_layers_list:
            for lr in lrs:
                for weight_decay in weight_decays:
                    config = copy.deepcopy(template)
                    
                    config["model"]["name"] = model
                    config["training"]["learning_rate"] = lr
                    config["optimizer"]["lr"] = lr
                    config["optimizer"]["weight_decay"] = weight_decay
                    config["training"]["epochs"] = epochs
                    config["training"]["batch_size"] = args.batch_size
                    
                    # Experiment 4 specific overrides
                    config["model"]["fine_tuning"]["strategy"] = "simultaneous"
                    config["model"]["fine_tuning"]["num_layers"] = num_layers
                    config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 0

                    # Strip LoRA configs for baselines
                    if "lora" in config:
                        del config["lora"]
                    if "lora_lr" in config["optimizer"]:
                        del config["optimizer"]["lora_lr"]
                    
                    # Logging & File Naming
                    config["logging"]["experiment_name"] = "Exp4_Simultaneous_Unfreezing_var-layers"
                    run_name = f"exp4_{model}_ft-simultaneous_nl{num_layers}_lr{lr:.6f}_l2-{weight_decay}"
                    config["logging"]["run_name"] = run_name
                    
                    file_path = os.path.join(active_dir, f"{run_name}.yaml")
                    
                    with open(file_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        
                    generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 4 in {active_dir}")

if __name__ == "__main__":
    main()