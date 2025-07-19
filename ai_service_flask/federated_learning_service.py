"""
World-Class Federated Learning Service
Advanced Multi-Company AI Training with Privacy Preservation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
import logging
import torch
import flwr as fl
from ml_core.models import ModelFactory
from ml_core.training import ModelTrainer, TrainingConfig
from ml_core.data_processing import DataProcessor
from ml_core.monitoring import MLMetricsTracker
from ml_core.utils import ModelRegistry

logger = logging.getLogger(__name__)

model_registry = ModelRegistry()
metrics_tracker = MLMetricsTracker()
data_processor = DataProcessor()

model_factory = ModelFactory()
model = model_factory.create_model('simple_nn', {'input_dim': 10, 'output_dim': 2})
config = TrainingConfig(epochs=10, batch_size=32, learning_rate=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()  # Replace with appropriate loss for your task
trainer = ModelTrainer(model, config, loss_fn)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainer.train_model(self.model, self.train_data)
        return self.get_parameters(config), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = trainer.evaluate_model(self.model, self.val_data)
        return float(loss), len(self.val_data), metrics

class FederatedLearningService:
    def __init__(self):
        self.model_factory = ModelFactory()
        self.model_registry = model_registry
        self.metrics_tracker = metrics_tracker
        self.data_processor = data_processor
        self.trainer = trainer

    def start_server(self, model_name, num_rounds=10, min_fit_clients=2, min_eval_clients=2):
        model_info = self.model_registry.get_model(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found in registry")
        model = self.model_factory.create_model(model_info['model_type'], model_info['model_params'])
        def get_eval_fn():
            def evaluate(server_round, parameters, config):
                model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
                loss, metrics = self.trainer.evaluate_model(model, None)
                return float(loss), metrics
            return evaluate
        strategy = fl.server.strategy.FedAvg(
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_fit_clients,
            on_evaluate_config_fn=get_eval_fn()
        )
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy
        )

    def start_client(self, model_name, train_data, val_data):
        model_info = self.model_registry.get_model(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found in registry")
        model = self.model_factory.create_model(model_info['model_type'], model_info['model_params'])
        client = FlowerClient(model, train_data, val_data)
        fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

federated_learning_service = FederatedLearningService() 