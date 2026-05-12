import os
import yaml
import argparse
import copy

def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 2 configurations.")
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
    
    baseline_lr = 0.005
    baseline_epochs = 10

    generated_count = 0
    for model in models:
        config = copy.deepcopy(template)
        
        config["model"]["name"] = model
        
        config["model"]["num_classes"] = 2
        
        if "data" not in config:
            config["data"] = {}
        config["data"]["num_classes"] = 2
        config["data"]["task"] = "binary" 
        # ---------------------------------------
        
        config["training"]["learning_rate"] = baseline_lr
        config["optimizer"]["lr"] = baseline_lr
        config["training"]["epochs"] = baseline_epochs
        config["training"]["batch_size"] = args.batch_size
        
        config["model"]["fine_tuning"]["strategy"] = "none"
        config["model"]["fine_tuning"]["num_layers"] = 0
        config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 0

        if "lora" in config:
            del config["lora"]
        if "lora_lr" in config["optimizer"]:
            del config["optimizer"]["lora_lr"]
        
        config["logging"]["experiment_name"] = "Exp2_SanityCheck_Binary"
        run_name = f"exp2_{model}_binary_sanity"
        config["logging"]["run_name"] = run_name
        
        file_path = os.path.join(active_dir, f"{run_name}.yaml")
        
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 2 in {active_dir}")

if __name__ == "__main__":
    main()