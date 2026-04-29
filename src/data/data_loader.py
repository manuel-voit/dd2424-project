from src.data.pet_dataset import get_pet_dataloaders


def get_dataloaders(config: dict):
    dataset_name = config['data']['name'].lower()
    
    if dataset_name == "oxford_pets" or dataset_name =="oxford_pets_binary":

        is_binary = True if dataset_name == "oxford_pets_binary" else False

        return get_pet_dataloaders(
            data_dir=config['data']['data_dir'],
            image_size=config['data']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            seed=config['training'].get('seed', 42),
            binary=is_binary
        )
        
    elif dataset_name == "coco":
        pass
        
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported!")
