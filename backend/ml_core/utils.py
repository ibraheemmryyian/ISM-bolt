import mlflow
from typing import Any, Dict, Optional

# --- STUB: ModelRegistry ---
class ModelRegistry:
    def __init__(self):
        try:
            from model_persistence_manager import ModelPersistenceManager
            self._persistence_manager = ModelPersistenceManager()
        except ImportError:
            self._persistence_manager = None
    def get_model(self, *args, **kwargs):
        raise NotImplementedError('ModelRegistry.get_model is a stub. Replace with real implementation.')
    def list_models(self):
        if self._persistence_manager:
            return self._persistence_manager.list_models()
        raise NotImplementedError('ModelRegistry.list_models is not implemented and ModelPersistenceManager is unavailable.')

class ExperimentTracker:
    def __init__(self, experiment_name: str = 'default'):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.run = mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, artifact_path: str):
        mlflow.log_artifact(artifact_path)

    def end_run(self):
        mlflow.end_run() 

# --- STUB: ConfigManager ---
class ConfigManager:
    def __init__(self, *args, **kwargs):
        pass 