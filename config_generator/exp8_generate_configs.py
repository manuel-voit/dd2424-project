import os
import yaml
import argparse
import copy

def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 8 configurations.")
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

    # Set exclusively to resnet101
    models = ["resnet101"] 
    
    # Base locked parameters
    epochs = 10
    
    # Grid search values for LoRA Learning Rate
    lora_lrs = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001]

    generated_count = 0
    for model in models:
        for lora_lr in lora_lrs:
            config = copy.deepcopy(template)
            
            # --- MODEL SETTINGS ---
            if "model" not in config:
                config["model"] = {}
            config["model"]["name"] = model
            
            # Nested fine-tuning settings under 'model'
            if "fine_tuning" not in config["model"]:
                config["model"]["fine_tuning"] = {}
                
            config["model"]["fine_tuning"]["strategy"] = "none"
            
            # Safely pop out the unwanted fine-tuning keys if they exist in the template
            config["model"]["fine_tuning"].pop("num_layers", None)
            config["model"]["fine_tuning"].pop("unfreeze_every_n_epochs", None)

            # --- TRAINING SETTINGS ---
            if "training" not in config:
                config["training"] = {}
            config["training"]["epochs"] = epochs
            config["training"]["batch_size"] = args.batch_size
            
            # --- LORA SETTINGS ---
            if "lora" in config:
                if "gradual_unfreeze" not in config["lora"]:
                    config["lora"]["gradual_unfreeze"] = {}
                    
                # Ensure it targets the 'enabled' sub-key
                config["lora"]["gradual_unfreeze"]["enabled"] = False
                
                # --- STRIP UNUSED SCHEDULE ---
                # Pop from inside gradual_unfreeze if it lives there
                config["lora"]["gradual_unfreeze"].pop("schedule", None)
                # Pop from the main lora block if it lives there instead
                config["lora"].pop("schedule", None) 
                # -----------------------------
                
                # Strip duplicated/useless lora parameters if they exist
                config["lora"].pop("learning_rate", None)
            
            # --- OPTIMIZER SETTINGS ---
            if "optimizer" not in config:
                config["optimizer"] = {}
            
            # Set the grid search target 
            config["optimizer"]["lora_lr"] = lora_lr
            
            # Lock standard learning rate (using 'lr' as per your schema)
            config["optimizer"]["lr"] = 0.005 
            
            # --- LOGGING & FILE NAMING ---
            if "logging" not in config:
                config["logging"] = {}
            config["logging"]["experiment_name"] = "Exp8_LoRA_LR_Search"
            
            # Formatting the LR to avoid awkward decimals in filenames (e.g., 1e-05)
            lr_str = f"{lora_lr:.5f}".rstrip('0').rstrip('.')
            run_name = f"exp8_{model}_lora_lr_{lr_str}"
            
            config["logging"]["run_name"] = run_name
            
            file_path = os.path.join(active_dir, f"{run_name}.yaml")
            
            with open(file_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
            generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 8 in {active_dir}")

if __name__ == "__main__":
    main()