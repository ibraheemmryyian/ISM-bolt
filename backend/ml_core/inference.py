"""
ML Core Inference: Inference utilities for all model types
"""
import torch

def predict_supervised(model, x):
    """Make predictions using a supervised model"""
    model.eval()
    with torch.no_grad():
        return model(x)

def predict_gnn(model, data, device='cpu'):
    model.eval()
    with torch.no_grad():
        return model(data.x.to(device), data.edge_index.to(device))

def predict_graph_embedding(model, head, rel, tail, device='cpu'):
    model.eval()
    with torch.no_grad():
        return model(head.to(device), rel.to(device), tail.to(device)) 