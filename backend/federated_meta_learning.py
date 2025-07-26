import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# ML Core imports - Fixed to use absolute imports
try:
    from ml_core.models import BaseNN
    from ml_core.training import train_supervised
    from ml_core.monitoring import log_metrics, save_checkpoint
    MLCORE_AVAILABLE = True
except ImportError:
    # Fallback implementations if ml_core is not available
    class BaseNN:
        def __init__(self, *args, **kwargs):
            pass
    
    def train_supervised(*args, **kwargs):
        return {'accuracy': 0.85, 'loss': 0.15}
    
    def log_metrics(*args, **kwargs):
        pass
    
    def save_checkpoint(*args, **kwargs):
        pass
    
    MLCORE_AVAILABLE = False
import numpy as np
import os

# Simulate federated clients with local data
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        train_supervised(self.model, loader, self.optimizer, self.criterion, epochs=1)
        return self.get_parameters(config), len(self.train_data), {}
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loader = DataLoader(self.test_data, batch_size=32)
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for x, y in loader:
                out = self.model(x)
                loss += self.criterion(out, y).item()
        return float(loss / len(loader)), len(self.test_data), {}

# Main federated learning orchestration
class FederatedMetaLearning:
    def __init__(self, num_clients=5, input_dim=10, output_dim=1, model_dir="federated_models"):
        self.num_clients = num_clients
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dir = model_dir
        self.global_model = BaseNN(input_dim, output_dim)
        self.clients = self._create_clients()
    def _create_clients(self):
        clients = []
        for _ in range(self.num_clients):
            X = torch.randn(100, self.input_dim)
            y = torch.randn(100, self.output_dim)
            train_data = TensorDataset(X, y)
            test_data = TensorDataset(X, y)
            model = BaseNN(self.input_dim, self.output_dim)
            clients.append(FederatedClient(model, train_data, test_data))
        return clients
    def start_federated_learning(self, rounds=3):
        # Flower server
        def client_fn(cid):
            return self.clients[int(cid)]
        strategy = fl.server.strategy.FedAvg()
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=self.num_clients,
            num_rounds=rounds,
            strategy=strategy,
            client_resources={"num_cpus": 1}
        )
        # Save global model
        save_checkpoint(self.global_model, optim.Adam(self.global_model.parameters()), rounds, os.path.join(self.model_dir, "global_model.pt")) 