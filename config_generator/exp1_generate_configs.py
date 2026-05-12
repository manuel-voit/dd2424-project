import os
import yaml
import argparse
import copy

def main():
    parser = argparse.ArgumentParser(description="Generate Experiment 1 configurations.")
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
    
    lrs_fine_tuning = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    lrs_linear_probing = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    strategies_and_lrs = [
        ("none", lrs_linear_probing, [0]),
        ("gradual", lrs_fine_tuning, [1, 2]),
        ("simultaneous", lrs_fine_tuning, [1, 2])
    ]

    generated_count = 0
    for model in models:
        for ft, lrs, layers_list in strategies_and_lrs:
            for lr in lrs:
                for nl in layers_list:
                    config = copy.deepcopy(template)
                    
                    config["model"]["name"] = model
                    
                    config["training"]["learning_rate"] = lr
                    config["optimizer"]["lr"] = lr
                    config["training"]["batch_size"] = args.batch_size
                    
                    config["model"]["fine_tuning"]["strategy"] = ft
                    config["model"]["fine_tuning"]["num_layers"] = nl
                    
                    # gradual unfreeze every 3 epochs
                    if ft == "gradual":
                        config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 3
                    else: 
                        # no gradual unfreezing for linear probing or simultaneous fine-tuning
                        config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 0

                    # Strip LoRA configs for baselines
                    if "lora" in config:
                        del config["lora"]
                    if "lora_lr" in config["optimizer"]:
                        del config["optimizer"]["lora_lr"]
                    
                    # Logging & File Naming
                    config["logging"]["experiment_name"] = "Exp1_Learning_Rates_and_Finetuning"
                    run_name = f"exp1_{model}_ft-{ft}_lr{lr}_nl{nl}"
                    config["logging"]["run_name"] = run_name
                    
                    file_path = os.path.join(active_dir, f"{run_name}.yaml")
                    
                    with open(file_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        
                    generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 1 in {active_dir}")

if __name__ == "__main__":
    main()
