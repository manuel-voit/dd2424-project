from src.data.pet_dataset import get_pet_dataloaders


def get_dataloaders(config: dict):
    dataset_name = config['data']['name'].lower()
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})
    imbalance_cfg = data_cfg.get('imbalance', {})
    
    if dataset_name == "oxford_pets" or dataset_name =="oxford_pets_binary":

        is_binary = True if dataset_name == "oxford_pets_binary" else False

        return get_pet_dataloaders(
            data_dir=data_cfg['data_dir'],
            image_size=data_cfg['image_size'],
            batch_size=train_cfg['batch_size'],
            num_workers=data_cfg['num_workers'],
            seed=train_cfg.get('seed', 42),
            pin_memory=data_cfg.get('pin_memory', True),
            persistent_workers=data_cfg.get('persistent_workers', True),
            prefetch_factor=data_cfg.get('prefetch_factor', 4),
            binary=is_binary,
            imbalanced=imbalance_cfg.get('enabled', data_cfg.get('imbalanced', False)),
            imbalance_factor=imbalance_cfg.get('imbalance_factor', 0.2),
            augmentation=data_cfg.get('augmentation', True),
            train_fraction=data_cfg.get('train_fraction', 1.0),
            oversample=imbalance_cfg.get('oversample', False)
        )
        
    elif dataset_name == "coco":
        pass
        
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported!")
