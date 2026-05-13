import os
import yaml
import argparse
import copy

def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 6 configurations.")
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
    
    # limiting amount of training data
    train_fractions = [1.0, 0.1, 0.02]
    
    # Data augmentation [on/off]
    augmentations = [True, False]
    
    # L2
    weight_decays = [0.00005, 0.0001, 0.0005]

    # XXXX VALUES TO BE DETERMINED XXXXX
    strategies = [
        {THIS IS TO INTENTIONALLY BREAK, REVIEW STRATEGIES BEFORE RUNNING},
        {
            "name": "linear",
            "ft_strategy": "none",
            "num_layers": 0,
            "unfreeze_every_n_epochs": 0,
            "lr": 0.001 
        },
        {
            "name": "simultaneous",
            "ft_strategy": "simultaneous",
            "num_layers": 4, 
            "unfreeze_every_n_epochs": 0,
            "lr": 0.0001
        },
        {
            "name": "gradual",
            "ft_strategy": "gradual",
            "num_layers": 4,
            "unfreeze_every_n_epochs": 2,
            "lr": 0.0001
        }
    ]

    epochs = 15

    generated_count = 0
    for model in models:
        for frac in train_fractions:
            for aug in augmentations:
                for wd in weight_decays:
                    for strat in strategies:
                        config = copy.deepcopy(template)
                        
                        config["model"]["name"] = model
                        
                        # Set strategy details
                        config["model"]["fine_tuning"]["strategy"] = strat["ft_strategy"]
                        config["model"]["fine_tuning"]["num_layers"] = strat["num_layers"]
                        config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = strat["unfreeze_every_n_epochs"]
                        
                        # Set LR and L2
                        config["training"]["learning_rate"] = strat["lr"]
                        config["optimizer"]["lr"] = strat["lr"]
                        config["optimizer"]["weight_decay"] = wd
                        
                        # Set sample limiting and augmentation
                        config["data"]["train_fraction"] = frac
                        config["data"]["augmentation"] = aug
                        
                        config["training"]["epochs"] = epochs
                        config["training"]["batch_size"] = args.batch_size

                        # Strip LoRA configs for standard finetuning
                        if "lora" in config:
                            del config["lora"]
                        if "lora_lr" in config["optimizer"]:
                            del config["optimizer"]["lora_lr"]
                        
                        # Logging & File Naming
                        aug_str = "aug-on" if aug else "aug-off"
                        frac_str = f"frac-{frac:.2f}"
                        
                        config["logging"]["experiment_name"] = "Exp6_LimitTrainData"
                        run_name = f"exp6_{model}_{strat['name']}_{frac_str}_{aug_str}_l2-{wd}"
                        config["logging"]["run_name"] = run_name
                        
                        file_path = os.path.join(active_dir, f"{run_name}.yaml")
                        
                        with open(file_path, "w") as f:
                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                            
                        generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 6 in {active_dir}")

if __name__ == "__main__":
    main()