"""
ML Core Training: Training utilities for all model types
"""
import torch
from torch.utils.data import DataLoader

def train_supervised(model, dataloader, optimizer, criterion, epochs=10, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Supervised] Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")
    return model

# Federated training loop (Flower integration stub)
def train_federated(model_fn, client_loaders, server_rounds=5, device='cpu'):
    # Placeholder for Flower-based federated training
    # Each client trains locally, server aggregates
    # Real implementation will use Flower's API
    pass

# GNN training loop
def train_gnn(model, data, optimizer, criterion, epochs=10, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(out, data.y.to(device))
        loss.backward()
        optimizer.step()
        print(f"[GNN] Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
    return model

# RL training loop (stub)
def train_rl(agent, env, episodes=100):
    # Placeholder for RL agent training
    # Real implementation will use gymnasium or custom env
    pass

# Graph embedding training (TransE, DistMult, etc.)
def train_graph_embedding(model, triples, optimizer, epochs=10, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for head, rel, tail, label in triples:
            head = torch.tensor([head]).to(device)
            rel = torch.tensor([rel]).to(device)
            tail = torch.tensor([tail]).to(device)
            label = torch.tensor([label]).float().to(device)
            optimizer.zero_grad()
            score = model(head, rel, tail)
            loss = torch.nn.functional.mse_loss(score, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Graph Embedding] Epoch {epoch+1}/{epochs} Loss: {total_loss/len(triples):.4f}")
    return model 