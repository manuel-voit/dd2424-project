import copy
import os
import yaml


MODELS = ["vit_b_16"]
LORA_LRS = [1e-4, 3e-4, 1e-3, 3e-3]
NUM_LAST_STAGES = [1, 2, 3, 4]
RANKS = [1, 8]

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_HEAD_LR = 0.01
DEFAULT_WEIGHT_DECAY = 0.01

VIT_B16_STAGE_BLOCKS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
]


def get_vit_b16_attention_targets(num_last_stages: int):
    selected_stages = VIT_B16_STAGE_BLOCKS[-num_last_stages:]
    targets = []
    for stage in selected_stages:
        for block_idx in stage:
            targets.append(f"encoder.layers.encoder_layer_{block_idx}.self_attention")
    return targets


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
        for num_last_stages in NUM_LAST_STAGES:
            targets = get_vit_b16_attention_targets(num_last_stages)
            for rank in RANKS:
                for lora_lr in LORA_LRS:
                    config = copy.deepcopy(template)

                    config["model"]["type"] = "vit"
                    config["model"]["name"] = model
                    config["model"]["num_classes"] = 37
                    config["model"]["fine_tuning"]["strategy"] = "none"
                    config["model"]["fine_tuning"]["num_layers"] = 0
                    config["model"]["fine_tuning"]["unfreeze_every_n_epochs"] = 0

                    config["training"]["batch_size"] = DEFAULT_BATCH_SIZE
                    config["training"]["epochs"] = DEFAULT_EPOCHS
                    config["training"]["learning_rate"] = DEFAULT_HEAD_LR

                    config["optimizer"]["name"] = "adamw"
                    config["optimizer"]["lr"] = DEFAULT_HEAD_LR
                    config["optimizer"]["lora_lr"] = lora_lr
                    config["optimizer"]["weight_decay"] = DEFAULT_WEIGHT_DECAY
                    if "llrd" in config["optimizer"]:
                        del config["optimizer"]["llrd"]

                    config["scheduler"]["name"] = "cosine_annealing_lr"
                    if "warmup_epochs" in config["scheduler"]:
                        del config["scheduler"]["warmup_epochs"]
                    if "warmup_start_factor" in config["scheduler"]:
                        del config["scheduler"]["warmup_start_factor"]
                    if "eta_min" in config["scheduler"]:
                        del config["scheduler"]["eta_min"]

                    config["loss"]["name"] = "cross_entropy"
                    config["data"]["name"] = "oxford_pets"
                    config["data"]["data_dir"] = "./data"

                    config["lora"]["r"] = rank
                    config["lora"]["alpha"] = 2 * rank
                    config["lora"]["learning_rate"] = lora_lr
                    config["lora"]["targets"] = targets
                    config["lora"]["gradual_unfreeze"]["enabled"] = False
                    config["lora"]["gradual_unfreeze"]["schedule"] = {}

                    config["logging"]["experiment_name"] = "Exp20_LoRA_ViT"
                    run_name = (
                        f"exp20_{model}_lora_nl{num_last_stages}_r{rank}_"
                        f"lora-lr{lora_lr:.4g}_head-lr{DEFAULT_HEAD_LR:.2f}_l2-{DEFAULT_WEIGHT_DECAY}"
                    )
                    config["logging"]["run_name"] = run_name

                    file_path = os.path.join(active_dir, f"{run_name}.yaml")
                    with open(file_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                    generated_count += 1

    print(f"Generated {generated_count} configurations for Experiment 20 in {active_dir}")


if __name__ == "__main__":
    main()
