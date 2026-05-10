import os
import yaml
import argparse
import itertools
import copy

def main():
    parser = argparse.ArgumentParser(description="Generate experiment configurations.")
    # Batch size to be specified in bash: e.g. python3 generate_configs.py --batch-size 64
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
    completed_dir = os.path.join(project_root, "configs", "completed")
    os.makedirs(active_dir, exist_ok=True)
    os.makedirs(completed_dir, exist_ok=True)

    # Define the hyperparameter grid
    models = ["resnet50", "resnet101"]
    
    # Map 'binary' and 'multi' to their dataset strings and class counts
    datasets = [
        {"id": "binary", "name": "oxford_pets_binary", "num_classes": 2},
        {"id": "multi", "name": "oxford_pets", "num_classes": 37}
    ]
    
    lrs = [1e-3, 5e-4, 1e-4]
    
    schedulers = ["none", "cosine_annealing_lr", "reduce_on_plateau"] 
    
    fine_tuning = ["none", "simultaneous", "gradual"]

    # Calculate total configs
    combinations = list(itertools.product(models, datasets, lrs, schedulers, fine_tuning))
    print(f"Generating {len(combinations)} configuration files...")

    generated_count = 0
    for m, d, lr, sched, ft in combinations:
        config = copy.deepcopy(template)
        
        config["model"]["name"] = m
        config["model"]["num_classes"] = d["num_classes"]
        
        config["data"]["name"] = d["name"]
        
        config["training"]["learning_rate"] = lr
        config["optimizer"]["lr"] = lr
        
        config["training"]["batch_size"] = args.batch_size
        
        config["scheduler"]["name"] = sched
        
        config["model"]["fine_tuning"]["strategy"] = ft
        config["model"]["fine_tuning"]["num_layers"] = 4  # ResNet normally has 4 macro blocks/layers
        config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 2
        
        # Strip LoRA configs for now (baseline runs)
        if "lora" in config:
            del config["lora"]
        if "lora_lr" in config["optimizer"]:
            del config["optimizer"]["lora_lr"]
        
        #Logging & File Naming
        run_name = f"{m}_{d['id']}_lr{lr}_{sched}_ft-{ft}"
        
        config["logging"]["run_name"] = run_name
        
        file_path = os.path.join(active_dir, f"{run_name}.yaml")
        
        # Write to configs/active directory
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        generated_count += 1

    print(f"Generated {generated_count} configurations in {active_dir}")

if __name__ == "__main__":
    main()
