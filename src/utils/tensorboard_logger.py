import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, config: dict):
        model_name = config['model']['name']
        
        # Determine the run name based on the config
        if 'lora' in config and config['lora']:
            r = config['lora'].get('r', 0)
            run_name = f"{model_name}_lora_r{r}"
        else:
            run_name = f"{model_name}_full_finetune"
            
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        self.run_dir = os.path.join("runs", f"{run_name}_{timestamp}")
        self.writer = SummaryWriter(self.run_dir)
        print(f"TensorBoard logging initialized at: {self.run_dir}")

    def log_scalars(self, tag: str, value_dict: dict, step: int):
        for key, value in value_dict.items():
            self.writer.add_scalar(f"{tag}/{key}", value, step)
    
    def log_hparams(self, config: dict, final_metrics: dict, step: int = 0):        
        # Base hyperparameters that exist in every run
        hparam_dict = {
            'model': config['model']['name'],
            'batch_size': config['training']['batch_size'],
            'lr': config['training']['learning_rate']
        }
        
        # Read LoRA params if available
        if 'lora' in config and config['lora']:
            hparam_dict['lora_r'] = config['lora'].get('r', 0)
            hparam_dict['lora_alpha'] = config['lora'].get('alpha', 0)
        else:
            hparam_dict['lora_r'] = 0
            hparam_dict['lora_alpha'] = 0
        
        for name, value in final_metrics.items():
            self.writer.add_scalar(name, value, step)

        self.writer.add_hparams(hparam_dict, final_metrics)

    def close(self):
        self.writer.close()
