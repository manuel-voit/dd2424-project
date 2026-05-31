# KTH DD2424 Project: Parameter-Efficient Fine-Tuning (PEFT) with LoRA on CNNs and ViTs

> Group members: Manuel Voit, Anton Holderied, Frederic Lorenz

This repository offers a broader experiment framework for transfer learning and parameter-efficient fine-tuning on image classification tasks.

## Current status

The project currently includes:

- End-to-end training and evaluation scripts driven by YAML configs
- Support for ResNet backbones and transformer-based vision models from `torchvision`
- Both classic fine-tuning and LoRA-based adaptation
- Configurable fine-tuning strategies: `none`, `simultaneous`, and `gradual`
- LoRA-specific controls including targeted injection and gradual unfreezing schedules
- Oxford-IIIT Pet and COCO-style dataset loaders, including binary and imbalanced variants
- MLflow logging, local run tracking, and a helper script for exporting runs to DagsHub
- Experiment config generators and an archive of many completed experiment configs
- A small pytest suite covering LoRA injection/update behavior

## Supported models

Backbones currently supported in code:

- ResNet: `resnet50`, `resnet101`
- Vision transformers / hierarchical transformers: `swin_t`, `swin_s`, `swin_b`, `swin_v2_t`, `vit_b_16`

LoRA injection is implemented for the model layers targeted by the experiment configs.

## Supported datasets

Datasets currently supported by `src/data/data_loader.py`:

- `oxford_pets`
- `oxford_pets_binary`
- `coco`
- `coco_binary`

The Oxford-IIIT Pet dataset is automatically downloaded. The MS COCO dataset needs to be downloaded manually.

The data pipeline also supports:

- train/validation/test splits
- configurable image size and dataloader performance settings
- augmentation on/off
- reduced training fractions
- class imbalance simulation
- oversampling and weighted loss for imbalance handling

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Running experiments

Train with a YAML config:

```bash
python train.py --config configs/your_config.yaml
```

Evaluate a saved checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/your_checkpoint.pth
```

Run the local MLflow UI, which can be then be accessed via [http://127.0.0.1:5000/](http://127.0.0.1:5000/):

```bash
mlflow ui
```

By default, MLflow logs are written to `mlruns/`.

## Config workflow

- `configs/config_template.yaml` is the best starting point for new runs.
- `configs/completed/` contains archived experiment configs that have already been run.
- `configs/legacy/` contains older reference configs used earlier in the project.
- `config_generator/` contains scripts that generate large experiment sweeps.

Start a batch of experiment runs with:

```bash
bash run_experiments.sh
```

or on Windows:

```powershell
.\run_experiments.ps1
```

The batch runner scripts expect a `configs/active/` directory. Both scripts train every config in `configs/active/` and move successful runs into `configs/completed/`.

## Testing

Run the test suite with:

```bash
pytest
```

The current tests mainly validate LoRA module injection and ensure only intended trainable parameters update during optimization.

## Repository structure

```text
.
|-- checkpoints/            # Saved model checkpoints
|-- config_generator/       # Scripts for generating experiment sweeps
|-- configs/
|   |-- active/             # Queue of configs to be run by the batch scripts
|   |-- completed/          # Archived configs for finished runs
|   |-- legacy/             # Older reference configs
|   `-- config_template.yaml
|-- data/                   # Local dataset storage
|-- mlruns/                 # Local MLflow tracking data
|-- src/
|   |-- data/               # Dataset loaders and transforms
|   |-- models/             # ResNet, ViT/Swin, and LoRA utilities
|   |-- utils/              # Loading, logging, saving, metrics, training setup
|   `-- engine.py           # Training and evaluation loops
|-- tests/                  # Pytest coverage for LoRA behavior
|-- train.py                # Main training entry point
|-- evaluate.py             # Checkpoint evaluation entry point
|-- run_experiments.ps1     # Windows batch runner for active configs
|-- run_experiments.sh      # Shell batch runner for active configs
`-- push_to_dagshub.py      # Helper for migrating MLflow experiments to DagsHub
```
