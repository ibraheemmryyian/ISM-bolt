"""
Fallback implementation for torch_geometric
"""
import torch
import torch.nn as nn
from typing import Any, Optional

class GCNConv(nn.Module):
    """Fallback GCN convolution layer"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        return self.linear(x)

class GATConv(nn.Module):
    """Fallback GAT convolution layer"""
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels * heads)
        self.heads = heads
    
    def forward(self, x, edge_index):
        return self.linear(x)

class HeteroConv(nn.Module):
    """Fallback heterogeneous convolution"""
    def __init__(self, convs):
        super().__init__()
        self.convs = convs
    
    def forward(self, x_dict, edge_index_dict):
        return x_dict

class HeteroData:
    """Fallback heterogeneous data"""
    def __init__(self):
        self.x = None
        self.edge_index = None
