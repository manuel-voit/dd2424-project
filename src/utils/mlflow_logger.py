import os
from datetime import datetime
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns


class MLflowLogger:
    def __init__(self, config: dict, experiment_name: str = "Transfer_Learning"):
        dagshub.init(
            repo_owner="manuel.voit",  
            repo_name="dd2424-project", 
            mlflow=True
        )
        
        # Group runs under one main experiment dashboard
        mlflow.set_experiment(experiment_name)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Determine the run name based on config
        model_name = config['model']['name']
        if 'lora' in config and config['lora']:
            r = config['lora'].get('r', 0)
            run_name = f"{model_name}_lora_r{r}_{timestamp}"
        else:
            run_name = f"{model_name}_full_finetune_{timestamp}"
        
        # Start the run
        self.run = mlflow.start_run(run_name=run_name)
        print(f"MLflow Run started: {run_name} (ID: {self.run.info.run_id})")

        # Log hyperparameters
        self._log_hparams(config)

    def _log_hparams(self, config: dict):
        params = {
            'model': config['model']['name'],
            'batch_size': config['training']['batch_size'],
            'lr': config['training']['learning_rate'],
            'epochs': config['training']['epochs']
        }
        
        if 'lora' in config and config['lora']:
            params['lora_r'] = config['lora'].get('r', 0)
            params['lora_alpha'] = config['lora'].get('alpha', 0)
            params['lora_lr'] = config['lora'].get('learning_rate', config['training']['learning_rate'])
            
        mlflow.log_params(params)
        
        # Save a copy of the YAML dictionary as a text artifact
        mlflow.log_dict(config, "config.yaml")

    # Plot confusion matrix and upload it to MLflow
    def log_confusion_matrix(self, cm, step, filename="confusion_matrix.png"):
        plt.figure(figsize=(12, 10))
        
        # Heatmap
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Breed')
        plt.ylabel('True Breed')
        plt.title('Test Confusion Matrix')
        
        plt.savefig(filename)
        mlflow.log_artifact(filename, artifact_path="plots")
        plt.close()

    def log_scalars(self, value_dict: dict, step: int):
        mlflow.log_metrics(value_dict, step=step)

    def log_artifact(self, file_path: str):
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path="checkpoints")

    def close(self):
        mlflow.end_run()
        print("MLflow Run finished.")
