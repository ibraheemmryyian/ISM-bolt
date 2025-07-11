import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, RGCNConv
import networkx as nx
import random
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GNNReasoning:
    """
    Graph Neural Network (GNN) Reasoning for Industrial Symbiosis.
    Supports multiple architectures:
    - GCN: Graph Convolutional Network (classic, general-purpose)
    - GraphSAGE: Inductive, scalable, learns from node neighborhoods
    - GAT: Graph Attention Network, learns which neighbors are most important
    - GIN: Graph Isomorphism Network, highly expressive for graph structure
    - R-GCN: Relational GCN, handles multiple edge types (waste, byproduct, trust)
    """
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.edge_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def nx_to_pyg(self, G: nx.Graph) -> Data:
        """
        Convert a NetworkX graph to a PyTorch Geometric Data object with real node and edge features.
        Node features: one-hot for industry, location, material type.
        Edge features: one-hot for relationship type.
        """
        if not isinstance(G, nx.Graph):
            logger.error("Input 'G' must be a NetworkX Graph.")
            raise ValueError("Input 'G' must be a NetworkX Graph.")
        node_attrs = []
        node_ids = list(G.nodes())
        try:
            for n in node_ids:
                attrs = G.nodes[n]
                node_attrs.append([
                    attrs.get('industry', ''),
                    attrs.get('location', ''),
                    attrs.get('waste_type', '') or attrs.get('material_needed', '')
                ])
            self.node_encoder.fit(node_attrs)
            node_features = self.node_encoder.transform(node_attrs)
            x = torch.tensor(node_features, dtype=torch.float)
            edge_tuples = []
            edge_types = []
            for u, v, d in G.edges(data=True):
                edge_tuples.append((node_ids.index(u), node_ids.index(v)))
                edge_types.append([d.get('key', d.get('type', 'related'))])
            edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
            self.edge_encoder.fit(edge_types)
            edge_features = self.edge_encoder.transform(edge_types)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.nx_mapping = {n: i for i, n in enumerate(node_ids)}
            data.nx_reverse = {i: n for i, n in enumerate(node_ids)}
            logger.info(f"Converted NetworkX graph to PyG Data: {data}")
        except Exception as e:
            logger.error(f"Error converting NetworkX graph to PyG Data: {e}")
            raise
        return data

    def sample_edges(self, data: Data, num_neg: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample positive (existing) and negative (non-existing) edges for link prediction.
        """
        pos_edge_index = data.edge_index
        num_nodes = data.num_nodes
        all_edges = set((int(u), int(v)) for u, v in zip(*pos_edge_index.cpu().numpy()))
        neg_edges = set()
        while len(neg_edges) < (num_neg or pos_edge_index.size(1)):
            u, v = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
            if u != v and (u, v) not in all_edges and (v, u) not in all_edges:
                neg_edges.add((u, v))
        neg_edge_index = torch.tensor(list(neg_edges), dtype=torch.long).t().contiguous()
        return pos_edge_index, neg_edge_index

    def greedy_link_prediction(self, G: nx.Graph, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """
        Greedy algorithm for link prediction: rank all possible pairs by a simple heuristic (e.g., waste match + industry compatibility).
        """
        node_ids = list(G.nodes())
        scored_links = []
        for i, u in enumerate(node_ids):
            for v in node_ids[i+1:]:
                score = 0
                attrs_u = G.nodes[u]
                attrs_v = G.nodes[v]
                if attrs_u.get('industry') == attrs_v.get('industry'):
                    score += 0.5
                if attrs_u.get('waste_type') and attrs_v.get('material_needed') and attrs_u.get('waste_type') in attrs_v.get('material_needed'):
                    score += 0.7
                if score > 0:
                    scored_links.append((u, v, score))
        scored_links.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Greedy algorithm predicted {len(scored_links[:top_n])} links.")
        return scored_links[:top_n]

    def train_gnn(self, data: Data, epochs: int = 100, model_type: str = 'gcn'):
        """
        Train a GNN for link prediction. model_type: 'gcn', 'sage', 'gat', 'gin', 'rgcn'
        """
        pos_edge_index, neg_edge_index = self.sample_edges(data)
        in_channels = data.num_node_features
        edge_dim = data.edge_attr.shape[1] if hasattr(data, 'edge_attr') else 0
        num_relations = data.edge_attr.shape[1] if hasattr(data, 'edge_attr') else 1
        if model_type == 'gcn':
            model = self.SimpleGCN(in_channels, edge_dim).to(self.device)
        elif model_type == 'sage':
            model = self.GraphSAGE(in_channels, edge_dim).to(self.device)
        elif model_type == 'gat':
            model = self.GAT(in_channels, edge_dim).to(self.device)
        elif model_type == 'gin':
            model = self.GIN(in_channels, edge_dim).to(self.device)
        elif model_type == 'rgcn':
            model = self.RGCN(in_channels, num_relations).to(self.device)
        else:
            logger.error(f"Unknown GNN model_type: {model_type}")
            raise ValueError(f"Unknown GNN model_type: {model_type}")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        data = data.to(self.device)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            if model_type == 'rgcn':
                pos_logits = model(data.x, data.edge_index, pos_edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None, data.edge_attr.argmax(dim=1) if hasattr(data, 'edge_attr') else None)
                neg_logits = model(data.x, data.edge_index, neg_edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None, data.edge_attr.argmax(dim=1) if hasattr(data, 'edge_attr') else None)
            else:
                pos_logits = model(data.x, data.edge_index, pos_edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None)
                neg_logits = model(data.x, data.edge_index, neg_edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None)
            pos_labels = torch.ones(pos_logits.size(0), device=self.device)
            neg_labels = torch.zeros(neg_logits.size(0), device=self.device)
            loss = F.binary_cross_entropy_with_logits(torch.cat([pos_logits, neg_logits]), torch.cat([pos_labels, neg_labels]))
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                logger.info(f"[{model_type.upper()}] Epoch {epoch}: Loss {loss.item():.4f}")
        return model

    def predict_links(self, data: Data, model, top_n: int = 5, model_type: str = 'gcn') -> List[Tuple[str, str, float]]:
        """
        Predict top-N new links (non-existing edges) with highest probability.
        Returns: list of (node1, node2, score)
        """
        num_nodes = data.num_nodes
        existing = set((int(u), int(v)) for u, v in zip(*data.edge_index.cpu().numpy()))
        candidates = [(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v and (u, v) not in existing and (v, u) not in existing]
        if not candidates:
            return []
        edge_pairs = torch.tensor(candidates, dtype=torch.long).t().contiguous().to(self.device)
        model.eval()
        with torch.no_grad():
            if model_type == 'rgcn':
                edge_type = data.edge_attr.argmax(dim=1) if hasattr(data, 'edge_attr') else None
                scores = torch.sigmoid(model(data.x.to(self.device), data.edge_index.to(self.device), edge_pairs, data.edge_attr.to(self.device) if hasattr(data, 'edge_attr') else None, edge_type)).cpu().numpy()
            else:
                scores = torch.sigmoid(model(data.x.to(self.device), data.edge_index.to(self.device), edge_pairs, data.edge_attr.to(self.device) if hasattr(data, 'edge_attr') else None)).cpu().numpy()
        top_idx = scores.argsort()[-top_n:][::-1]
        mapping = data.nx_reverse
        results = [(mapping[candidates[i][0]], mapping[candidates[i][1]], float(scores[i])) for i in top_idx]
        return results

    def run_gnn_inference(self, G: nx.Graph, model_type: str = 'gcn', use_greedy: bool = False) -> List[Tuple[str, str, float]]:
        """
        Run GNN inference to predict new links (potential symbiosis) using the selected architecture or greedy algorithm.
        """
        if use_greedy:
            logger.info("Running greedy link prediction.")
            return self.greedy_link_prediction(G)
        data = self.nx_to_pyg(G)
        model = self.train_gnn(data, epochs=30, model_type=model_type)
        return self.predict_links(data, model, top_n=5, model_type=model_type)

    class SimpleGCN(torch.nn.Module):
        def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int = 16) -> None:
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.link_pred = torch.nn.Bilinear(hidden_channels, hidden_channels, 1)
        def forward(self, x, edge_index, edge_pairs, edge_attr=None):
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            src, dst = edge_pairs
            x_src = x[src]
            x_dst = x[dst]
            logits = self.link_pred(x_src, x_dst).squeeze(-1)
            return logits

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int = 16) -> None:
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.link_pred = torch.nn.Bilinear(hidden_channels, hidden_channels, 1)
        def forward(self, x, edge_index, edge_pairs, edge_attr=None):
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            src, dst = edge_pairs
            x_src = x[src]
            x_dst = x[dst]
            logits = self.link_pred(x_src, x_dst).squeeze(-1)
            return logits

    class GAT(torch.nn.Module):
        def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int = 16, heads: int = 2) -> None:
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
            self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=1)
            self.link_pred = torch.nn.Bilinear(hidden_channels, hidden_channels, 1)
        def forward(self, x, edge_index, edge_pairs, edge_attr=None):
            x = F.elu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            src, dst = edge_pairs
            x_src = x[src]
            x_dst = x[dst]
            logits = self.link_pred(x_src, x_dst).squeeze(-1)
            return logits

    class GIN(torch.nn.Module):
        def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int = 16) -> None:
            super().__init__()
            nn1 = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels))
            nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
            self.link_pred = torch.nn.Bilinear(hidden_channels, hidden_channels, 1)
        def forward(self, x, edge_index, edge_pairs, edge_attr=None):
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            src, dst = edge_pairs
            x_src = x[src]
            x_dst = x[dst]
            logits = self.link_pred(x_src, x_dst).squeeze(-1)
            return logits

    class RGCN(torch.nn.Module):
        def __init__(self, in_channels: int, num_relations: int, hidden_channels: int = 16) -> None:
            super().__init__()
            self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
            self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
            self.link_pred = torch.nn.Bilinear(hidden_channels, hidden_channels, 1)
        def forward(self, x, edge_index, edge_pairs, edge_attr=None, edge_type=None):
            x = F.relu(self.conv1(x, edge_index, edge_type))
            x = self.conv2(x, edge_index, edge_type)
            src, dst = edge_pairs
            x_src = x[src]
            x_dst = x[dst]
            logits = self.link_pred(x_src, x_dst).squeeze(-1)
            return logits

if __name__ == "__main__":
    import sys
    import json
    logging.basicConfig(level=logging.INFO)
    # Example graph for testing
    G = nx.Graph()
    G.add_node('A', industry='Steel Manufacturing', location='NY', waste_type='slag', material_needed='', company_name='SteelCo')
    G.add_node('B', industry='Construction', location='NY', waste_type='', material_needed='slag', company_name='BuildCo')
    G.add_node('C', industry='Chemical Manufacturing', location='NJ', waste_type='solvent', material_needed='', company_name='ChemCo')
    G.add_node('D', industry='Pharmaceutical', location='NJ', waste_type='', material_needed='solvent', company_name='PharmaCo')
    G.add_edge('A', 'B', type='potential_symbiosis')
    G.add_edge('C', 'D', type='potential_symbiosis')
    engine = GNNReasoning()
    print("Testing greedy link prediction...")
    greedy_links = engine.greedy_link_prediction(G, top_n=3)
    print(json.dumps(greedy_links, indent=2))
    print("Testing GNN inference (GCN)...")
    gnn_links = engine.run_gnn_inference(G, model_type='gcn', use_greedy=False)
    print(json.dumps(gnn_links, indent=2)) 