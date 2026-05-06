import os
from datetime import datetime
from typing import Optional
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns

# Recursively flatten nested dictionary
def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class MLflowLogger:
    def __init__(self, config: dict, experiment_name: Optional[str] = None):
        logging_cfg = config.get('logging', {})
        dagshub.init(
            repo_owner=logging_cfg.get('dagshub_repo_owner', "manuel.voit"),  
            repo_name=logging_cfg.get('dagshub_repo_name', "dd2424-project"), 
            mlflow=True
        )
        
        # Group runs under one main experiment dashboard
        mlflow.set_experiment(experiment_name or logging_cfg.get('experiment_name', "Transfer_Learning"))
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Determine the run name based on config
        run_name = config['logging']["run_name"] if "run_name" in config['logging'] else f"Run_{timestamp}"
        
        # Start the run
        self.run = mlflow.start_run(run_name=run_name)
        print(f"MLflow Run started: {run_name} (ID: {self.run.info.run_id})")

        # Log hyperparameters
        self._log_hparams(config)

    def _log_hparams(self, config: dict):
        flat_params = flatten_dict(config)

        mlflow.log_params(flat_params)
        
        # Save a copy of the YAML dictionary as a text artifact
        mlflow.log_dict(config, "config.yaml")

    # Plot confusion matrix and upload it to MLflow
    def log_confusion_matrix(self, cm, step, filename="confusion_matrix.png"):
        fig = plt.figure(figsize=(12, 10))
        
        # Heatmap
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Breed')
        plt.ylabel('True Breed')
        plt.title('Test Confusion Matrix')
        
        mlflow.log_figure(fig, artifact_file=filename)
        plt.close()

    def log_scalars(self, value_dict: dict, step: int):
        mlflow.log_metrics(value_dict, step=step)

    def log_artifact(self, file_path: str):
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path="checkpoints")

    def close(self):
        mlflow.end_run()
        print("MLflow Run finished.")
