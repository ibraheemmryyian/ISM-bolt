"""
ML Core Models: Base classes and advanced architectures for ISM AI
"""
import torch
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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

class SimpleNN(nn.Module):
    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)

class ModelFactory:
    def create_model(self, model_type: str, params: dict):
        if model_type == 'simple_nn':
            input_dim = params.get('input_dim', 10)
            output_dim = params.get('output_dim', 2)
            return SimpleNN(input_dim, output_dim)
        raise ValueError(f"Unknown model_type: {model_type}")

# --- STUB: ModelArchitecture ---
class ModelArchitecture:
    def __init__(self, *args, **kwargs):
        pass

@dataclass
class ModelConfig:
    """
    Production-grade configuration for all AI models.
    Supports serialization, validation, and dynamic overrides.
    """
    model_type: str
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    hidden_size: Optional[int] = 768
    num_layers: Optional[int] = 6
    num_heads: Optional[int] = 12
    dropout: Optional[float] = 0.1
    num_classes: Optional[int] = None
    task_type: Optional[str] = None
    use_adapters: Optional[bool] = False
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.update(self.additional_params)
        return d

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        base_fields = {f.name for f in cls.__dataclass_fields__.values()}
        base_args = {k: v for k, v in config.items() if k in base_fields}
        additional = {k: v for k, v in config.items() if k not in base_fields}
        return cls(**base_args, additional_params=additional)

    def validate(self):
        if not self.model_type:
            raise ValueError("model_type must be specified in ModelConfig")
        if self.input_dim is not None and self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim is not None and self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.hidden_size is not None and self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers is not None and self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads is not None and self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.dropout is not None and not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be between 0.0 and 1.0")
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        # Add more validation as needed for your use cases 

# --- STUB: PromptGenerationModel ---
class PromptGenerationModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Add extensible initialization here
    def forward(self, *args, **kwargs):
        raise NotImplementedError('PromptGenerationModel.forward must be implemented by subclasses.') 