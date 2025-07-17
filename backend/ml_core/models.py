"""
ML Core Models: Base classes and advanced architectures for ISM AI
"""
import torch
import torch.nn as nn

# Base NN model
class BaseNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)

# GNN base (to be extended by torch-geometric models)
try:
    from torch_geometric.nn import GCNConv, GATConv
    class BaseGCN(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 64)
            self.conv2 = GCNConv(64, out_channels)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x
    class BaseGAT(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.gat1 = GATConv(in_channels, 64, heads=4)
            self.gat2 = GATConv(64*4, out_channels, heads=1)
        def forward(self, x, edge_index):
            x = self.gat1(x, edge_index).relu()
            x = self.gat2(x, edge_index)
            return x
except ImportError:
    pass

# RL agent base
class BaseRLAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, state):
        return self.policy(state)

# Transformer base
try:
    from torch.nn import Transformer
    class BaseTransformer(nn.Module):
        def __init__(self, d_model=128, nhead=8, num_layers=2):
            super().__init__()
            self.transformer = Transformer(d_model, nhead, num_layers)
        def forward(self, src, tgt):
            return self.transformer(src, tgt)
except ImportError:
    pass

# Graph embedding models (TransE, ComplEx, DistMult stubs)
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
    def forward(self, head, relation, tail):
        return torch.norm(self.entity_emb(head) + self.relation_emb(relation) - self.entity_emb(tail), p=1, dim=1)

class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
    def forward(self, head, relation, tail):
        return torch.sum(self.entity_emb(head) * self.relation_emb(relation) * self.entity_emb(tail), dim=1)

# --- STUB: ModelFactory ---
class ModelFactory:
    def __init__(self):
        pass
    def create_model(self, *args, **kwargs):
        raise NotImplementedError('ModelFactory.create_model is a stub. Replace with real implementation.') 