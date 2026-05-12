# KTH DD2424: Parameter-Efficient Fine-Tuning (PEFT) with LoRA on CNNs and ViTs

> Group Members: Manuel Voit, Anton Holderied, Frederic Lorenz

## Download dataset and sanity check:

To start training, simply run train.py using a specific config:

```bash
python3 train.py --config configs/your_config.yaml
```

To run the MLflow UI, which can be accessed via [http://127.0.0.1:5000/](http://127.0.0.1:5000/):
```bash
mlflow ui
```

PyTorch will automatically download the Oxford-IIIT Pet dataset for you and load a test batch.

To evaluate a trained network, just run evaluate.py using a specific checkpoint:
```bash
python3 evaluate.py --checkpoint checkpoints/checkpoint.pth"
```

## Folder structure

```
dd2424-project/
├── checkpoints/             # Stores the checkpoints
├── configs/                 # Configuration files for the trainigs
├── data/                    # Directory for the dataset
├── src/                     
│   ├── data/                
│   │   ├── __init__.py
│   │   ├── data_loader.py   # Generic data loader
│   │   ├── pet_dataset.py   # Data loader for the Oxford Pet dataset with stratified Train/Val/Test split
│   │   └── transforms.py    # ImageNet normalization and data augmentations
│   ├── models/              
│   │   ├── __init__.py
│   │   ├── cnn_backbone.py  # Factory for loading and modifying CNN
│   │   ├── vit_backbone.py  # Factory for loading and modifying ViT
│   │   └── lora.py          # Custom LoRALinear, LoRAConv2d classes and the injection function
│   ├── utils/               # Helper functions to keep the main loops clean
│   │   ├── __init__.py
│   │   ├── loading.py       # Load a checkpoint with config
│   │   ├── metrics.py       # Tracking loss/accuracy batch-by-batch
│   │   ├── saving.py        # Early stopping and weight saving logic
│   │   └── seed.py          # Set seeds to ensure determinism
│   └── engine.py            # The pure PyTorch training and evaluation loops
├── train.py                 # The main execution script (parses config, builds model, runs engine)
├── evaluate.py              # The script to test your saved weights against the test set
├── requirements.txt         # Environment dependencies (PyTorch, scikit-learn, pyyaml, etc.)
└── .gitignore               # Prevents uploading large datasets and model checkpoints to GitHub
```
