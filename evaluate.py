import argparse
import torch
import torch.nn as nn

from src.utils.loading import load_model_from_checkpoint
from src.utils.seed import set_seed
from src.data.data_loader import get_dataloaders
from src.engine import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN/ViT network")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the .pth checkpoint file")
    args = parser.parse_args()

    # Setup
    set_seed(42)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load the model and its config
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    # Recreate the dataloaders using the settings from the config
    loaders = get_dataloaders(config=config)
    test_loader = loaders['test']

    # Run the evaluation
    criterion = nn.CrossEntropyLoss()
    print("\nRunning evaluation on the test set ...")

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"Final Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Final Test Loss:     {test_metrics['loss']:.4f}")
    print(f"Final Test F1:       {test_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
