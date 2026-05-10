import os
import yaml
import argparse
import itertools
import copy

def main():
    parser = argparse.ArgumentParser(description="Generate extended baseline experiment configurations.")
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
    
    datasets = [
        {"id": "binary", "name": "oxford_pets_binary", "num_classes": 2},
        {"id": "multi", "name": "oxford_pets", "num_classes": 37}
    ]
    
    lrs = [1e-3, 5e-4]
    schedulers = ["cosine_annealing_lr"] 
    fine_tuning = ["none", "gradual"]
    train_fractions = [1.0, 0.1, 0.02]

    # Imbalance combinations: (enabled, oversample, use_weighted_loss)
    imbalance_configs = [
        (False, False, False),  # Imbalance off
        (True, False, False),   # Imbalance on, no mitigation
        (True, True, False),    # Imbalance on, with oversampling
        (True, False, True),    # Imbalance on, with weighted loss
        (True, True, True)      # Imbalance on, with both oversampling and weighted loss
    ]

    # Calculate total configs
    combinations = list(itertools.product(models, datasets, lrs, schedulers, fine_tuning, train_fractions, imbalance_configs))    
    print(f"Generating {len(combinations)} configuration files...")

    generated_count = 0
    for m, d, lr, sched, ft, frac, (imb_on, osamp, wl) in combinations:
        config = copy.deepcopy(template)
        
        config["model"]["name"] = m
        config["model"]["num_classes"] = d["num_classes"]
        config["data"]["name"] = d["name"]
        
        config["training"]["learning_rate"] = lr
        config["optimizer"]["lr"] = lr
        config["training"]["batch_size"] = args.batch_size
        config["scheduler"]["name"] = sched
        
        config["model"]["fine_tuning"]["strategy"] = ft
        config["model"]["fine_tuning"]["num_layers"] = 4 
        config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 2
        
        # New parameters
        config["data"]["train_fraction"] = frac
        config["data"]["imbalance"]["enabled"] = imb_on
        config["data"]["imbalance"]["oversample"] = osamp
        config["data"]["imbalance"]["use_weighted_loss"] = wl
        
        # Strip LoRA configs for baseline runs
        if "lora" in config:
            del config["lora"]
        if "lora_lr" in config["optimizer"]:
            del config["optimizer"]["lora_lr"]
        
        # Naming logic to keep filenames structured and readable
        imb_str = "imb-on" if imb_on else "imb-off"
        if imb_on:
            mitigations = []
            if osamp: mitigations.append("os")
            if wl: mitigations.append("wl")
            if mitigations:
                imb_str += "-" + "".join(mitigations)
                
        run_name = f"{m}_{d['id']}_lr{lr}_{sched}_ft-{ft}_frac{frac}_{imb_str}"
        config["logging"]["run_name"] = run_name
        
        file_path = os.path.join(active_dir, f"{run_name}.yaml")
        
        # Write to configs/active directory
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        generated_count += 1

    print(f"Generated {generated_count} configurations in {active_dir}")

if __name__ == "__main__":
    main()