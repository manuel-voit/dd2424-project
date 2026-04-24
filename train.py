import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import random
import numpy as np
import datetime

import argparse
import yaml
# UI for Visualization
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from src.models.cnn_backbone import get_binary_resnet
from src.models.vit_backbone import get_binary_swin_t

from src.engine import train_one_epoch, evaluate
from src.data.data_loader import get_pet_dataloaders
from src.utils.metrics import compute_classification_metrics


def set_seed(seed=42):
    """Ensures our ResNet vs ViT comparisons are completely fair and reproducible."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _extract_labels_from_dataset(dataset):
    """Extract labels from OxfordIIITPet datasets and Subset wrappers."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [int(base._labels[idx]) for idx in dataset.indices]
    return [int(label) for label in dataset._labels]


def _make_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label] for label in labels], dtype=np.float64)
    sample_weights = torch.from_numpy(sample_weights)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def _make_class_weights(labels, num_classes, device):
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    weights = class_counts.sum() / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def main():
    parser = argparse.ArgumentParser(description="Train CNN/ViT networks")
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    seed = config.get('training', {}).get('seed', 42)

    # Setup
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Extract config variables
    MODEL_TYPE = config['model']['type']
    TASK = config.get('data', {}).get('task', 'binary')
    NUM_CLASSES = config['model']['num_classes']

    BATCH_SIZE = config['training']['batch_size']
    EPOCHS = config['training']['epochs']
    LEARNING_RATE = config['training']['learning_rate']

    DATA_DIR = config['data']['data_dir']

    imbalance_cfg = config.get('imbalance', {})
    imbalance_enabled = bool(imbalance_cfg.get('enabled', False))
    class_to_fraction = imbalance_cfg.get('class_to_fraction', {})
    if imbalance_enabled and imbalance_cfg.get('preset') == 'cats_20pct':
        class_to_fraction = {i: 0.2 for i in range(12)}

    mitigation_cfg = config.get('mitigation', {})
    use_weighted_loss = bool(mitigation_cfg.get('weighted_cross_entropy', False))
    use_oversampling = bool(mitigation_cfg.get('oversample_minority', False))
    
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"runs/{MODEL_TYPE}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_{current_time}"
    writer = SummaryWriter(log_dir=log_dir)

    # Load Data
    loaders = get_pet_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=config['data']['num_workers'],
        imbalanced=imbalance_enabled,
        imbalance_config={
            'apply_to': imbalance_cfg.get('apply_to', ['train']),
            'class_to_fraction': class_to_fraction,
            'seed': imbalance_cfg.get('seed', seed),
        },
    )

    train_loader = loaders[TASK]['train']
    val_loader = loaders[TASK]['val']
    test_loader = loaders[TASK]['test']

    if TASK == 'binary' and NUM_CLASSES != 2:
        print(f"Overriding num_classes from {NUM_CLASSES} to 2 for binary task.")
        NUM_CLASSES = 2

    # Model branching
    if MODEL_TYPE == "resnet":
        model = get_binary_resnet()
    elif MODEL_TYPE == "vit":
        model = get_binary_swin_t()
    
    model = model.to(device)

    # Filter parameters for the Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Print a quick sanity check to ensure parameter efficiency
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in trainable_params)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trained_params:,} ({100 * trained_params / total_params:.2f}%)")

    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE)

    train_labels = _extract_labels_from_dataset(train_loader.dataset)

    if use_oversampling:
        sampler = _make_weighted_sampler(train_labels)
        loader_kwargs = {
            'batch_size': train_loader.batch_size,
            'num_workers': train_loader.num_workers,
            'pin_memory': train_loader.pin_memory,
            'sampler': sampler,
        }
        if train_loader.num_workers > 0:
            loader_kwargs['persistent_workers'] = train_loader.persistent_workers
            if train_loader.prefetch_factor is not None:
                loader_kwargs['prefetch_factor'] = train_loader.prefetch_factor
        train_loader = DataLoader(train_loader.dataset, **loader_kwargs)

    if use_weighted_loss:
        class_weights = _make_class_weights(train_labels, NUM_CLASSES, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device, return_outputs=True
        )
        val_metrics = compute_classification_metrics(val_labels, val_preds, NUM_CLASSES)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Weighted-F1: {val_metrics['weighted_f1']:.4f}"
        )
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1_macro/val', val_metrics['macro_f1'], epoch)
        writer.add_scalar('F1_weighted/val', val_metrics['weighted_f1'], epoch)

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, return_outputs=True
    )
    test_metrics = compute_classification_metrics(test_labels, test_preds, NUM_CLASSES)
    print(
        f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
        f"Test Macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"Test Weighted-F1: {test_metrics['weighted_f1']:.4f}"
    )

    worst_classes = sorted(test_metrics['per_class'], key=lambda row: row['f1'])[:5]
    print("Worst classes by F1 on test split:")
    for row in worst_classes:
        print(
            f"  class={row['class_id']}, f1={row['f1']:.4f}, "
            f"recall={row['recall']:.4f}, support={row['support']}"
        )

    writer.close()

if __name__ == "__main__":
    main()
