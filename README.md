# KTH DD2424: Parameter-Efficient Fine-Tuning (PEFT) with LoRA on CNNs

> Group Members: Manuel Voit, Anton Holderied, Frederic Lorenz

## Download dataset and sanity check:

To get started, simply execute:

```bash
python3 code/data_loader.py
```

PyTorch will automatically download the Oxford-IIIT Pet dataset for you and load a test batch.

## Folder structure

dd2424-project/
├── configs/                 
│   ├── train_resnet.yaml    # Hyperparameters for the CNN (ResNet-50)
│   └── train_swin.yaml      # Hyperparameters for the ViT (Swin-T)
├── data/                    # Directory for the dataset
├── src/                     
│   ├── data/                
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset loading and stratified Train/Val/Test splits
│   │   └── transforms.py    # ImageNet normalization and data augmentations
│   ├── models/              
│   │   ├── __init__.py
│   │   ├── cnn_backbone.py  # Factory for loading and modifying CNN
│   │   ├── vit_backbone.py  # Factory for loading and modifying ViT
│   │   └── lora.py          # Custom LoRALinear, LoRAConv2d classes and the injection function
│   ├── utils/               # Helper functions to keep the main loops clean
│   │   ├── __init__.py
│   │   ├── metrics.py       # Tracking loss/accuracy batch-by-batch
│   │   └── saving.py        # Early stopping and LoRA-only weight saving logic
│   └── engine.py            # The pure PyTorch training and evaluation loops
├── train.py                 # The main execution script (parses config, builds model, runs engine)
├── evaluate.py              # The script to test your saved weights against the hold-out set
├── requirements.txt         # Environment dependencies (PyTorch, scikit-learn, pyyaml, etc.)
└── .gitignore               # Prevents uploading large datasets and model checkpoints to GitHub