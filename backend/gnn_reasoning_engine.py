import torch
from torch_geometric.data import Data
from backend.ml_core.models import BaseGCN, BaseGAT
from backend.ml_core.training import train_gnn
from backend.ml_core.inference import predict_gnn
from backend.ml_core.monitoring import log_metrics, save_checkpoint
import numpy as np
import os

class GNNReasoningEngine:
    def __init__(self, input_dim=16, output_dim=4, model_type="gcn", model_dir="gnn_models"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_type = model_type
        self.model_dir = model_dir
        if model_type == "gcn":
            self.model = BaseGCN(input_dim, output_dim)
        elif model_type == "gat":
            self.model = BaseGAT(input_dim, output_dim)
        else:
            raise ValueError("Unknown model type")
    def train(self, x, edge_index, y, epochs=20):
        data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor(y, dtype=torch.long))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        train_gnn(self.model, data, optimizer, criterion, epochs=epochs)
        save_checkpoint(self.model, optimizer, epochs, os.path.join(self.model_dir, f"{self.model_type}_model.pt"))
    def infer(self, x, edge_index):
        data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long))
        return predict_gnn(self.model, data).detach().cpu().numpy()
    def export_embeddings(self, x, edge_index):
        self.model.eval()
        with torch.no_grad():
            data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long))
            embeddings = self.model(data.x, data.edge_index)
        return embeddings.cpu().numpy()
