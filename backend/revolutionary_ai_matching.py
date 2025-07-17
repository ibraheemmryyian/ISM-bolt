import torch
import numpy as np
from backend.ml_core.models import BaseNN
from backend.ml_core.training import train_supervised
from backend.ml_core.inference import predict_supervised
from backend.ml_core.monitoring import log_metrics, save_checkpoint
from torch.utils.data import DataLoader, TensorDataset
import os

class RevolutionaryMatchingModel:
    def __init__(self, input_dim=12, output_dim=1, model_dir="revolutionary_matching_models"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dir = model_dir
        self.model = BaseNN(input_dim, output_dim)
    def train(self, X, y, epochs=20):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        train_supervised(self.model, loader, optimizer, criterion, epochs=epochs)
        save_checkpoint(self.model, optimizer, epochs, os.path.join(self.model_dir, "revolutionary_matching_model.pt"))
    def predict(self, X):
        logits = predict_supervised(self.model, torch.tensor(X, dtype=torch.float)).detach().cpu().numpy()
        return (logits > 0).astype(int)
