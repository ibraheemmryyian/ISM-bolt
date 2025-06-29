import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, RGCNConv
import networkx as nx
import random
from typing import List, Tuple, Dict
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GNNReasoningEngine:
    """
    Enhanced Graph Neural Network Reasoning Engine for Industrial Symbiosis
    with multi-hop detection and advanced link prediction capabilities.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.node_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.edge_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.models = {}
        self.model_performances = {}
        logger.info(f"GNN Reasoning Engine initialized on {self.device}")
    
    def create_industrial_graph(self, participants: List[Dict]) -> nx.Graph:
        """Create a NetworkX graph from industrial participants with rich attributes."""
        G = nx.Graph()
        
        # Add nodes with industrial attributes
        for p in participants:
            G.add_node(p['id'], 
                      industry=p.get('industry', 'Unknown'),
                      location=p.get('location', 'Unknown'),
                      waste_type=p.get('waste_type', p.get('material_needed', 'Unknown')),
                      carbon_footprint=p.get('carbon_footprint', 0),
                      annual_waste=p.get('annual_waste', 0),
                      capabilities=','.join(p.get('capabilities', [])),
                      company_name=p.get('name', p.get('company_name', 'Unknown')))
        
        # Add initial edges based on simple heuristics
        for i, p1 in enumerate(participants):
            for j, p2 in enumerate(participants[i+1:], i+1):
                # Industry compatibility
                if self._check_industry_compatibility(p1, p2):
                    G.add_edge(p1['id'], p2['id'], 
                             type='potential_symbiosis',
                             confidence=0.5,
                             created_at=datetime.now().isoformat())
                
                # Material matching
                if self._check_material_match(p1, p2):
                    G.add_edge(p1['id'], p2['id'], 
                             type='material_match',
                             confidence=0.7,
                             created_at=datetime.now().isoformat())
        
        return G
    
    def _check_industry_compatibility(self, p1: Dict, p2: Dict) -> bool:
        """Check if two industries are compatible for symbiosis."""
        compatible_pairs = [
            ('Steel Manufacturing', 'Construction'),
            ('Chemical Manufacturing', 'Pharmaceutical'),
            ('Food Processing', 'Agriculture'),
            ('Cement Production', 'Construction'),
            ('Paper Manufacturing', 'Packaging'),
            ('Power Generation', 'Manufacturing')
        ]
        
        ind1, ind2 = p1.get('industry', ''), p2.get('industry', '')
        for pair in compatible_pairs:
            if (ind1 in pair[0] and ind2 in pair[1]) or (ind1 in pair[1] and ind2 in pair[0]):
                return True
        return False
    
    def _check_material_match(self, p1: Dict, p2: Dict) -> bool:
        """Check if materials match between participants."""
        p1_waste = p1.get('waste_type', '').lower()
        p1_need = p1.get('material_needed', '').lower()
        p2_waste = p2.get('waste_type', '').lower()
        p2_need = p2.get('material_needed', '').lower()
        
        return (p1_waste and p2_need and p1_waste in p2_need) or \
               (p2_waste and p1_need and p2_waste in p1_need)
    
    def nx_to_pyg_enhanced(self, G: nx.Graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data with enhanced features."""
        node_attrs = []
        node_ids = list(G.nodes())
        
        # Enhanced node features
        for n in node_ids:
            attrs = G.nodes[n]
            # Categorical features
            cat_features = [
                attrs.get('industry', 'Unknown'),
                attrs.get('location', 'Unknown'),
                attrs.get('waste_type', 'Unknown')
            ]
            node_attrs.append(cat_features)
        
        # One-hot encode categorical features
        if len(node_attrs) > 0:
            self.node_encoder.fit(node_attrs)
            cat_features = self.node_encoder.transform(node_attrs)
        else:
            cat_features = np.zeros((len(node_ids), 3))
        
        # Numerical features (normalized)
        num_features = []
        for n in node_ids:
            attrs = G.nodes[n]
            carbon = float(attrs.get('carbon_footprint', 0))
            waste = float(attrs.get('annual_waste', 0))
            # Normalize
            carbon_norm = carbon / 100000.0 if carbon > 0 else 0.0
            waste_norm = waste / 10000.0 if waste > 0 else 0.0
            num_features.append([carbon_norm, waste_norm])
        
        # Combine features
        node_features = np.hstack([cat_features, num_features])
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Edge index and features
        edge_tuples = []
        edge_features = []
        for u, v, d in G.edges(data=True):
            u_idx = node_ids.index(u)
            v_idx = node_ids.index(v)
            edge_tuples.append((u_idx, v_idx))
            edge_tuples.append((v_idx, u_idx))  # Undirected
            
            edge_type = d.get('type', 'potential')
            confidence = d.get('confidence', 0.5)
            edge_features.append([edge_type, confidence])
            edge_features.append([edge_type, confidence])
        
        edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)
        data.nx_mapping = {n: i for i, n in enumerate(node_ids)}
        data.nx_reverse = {i: n for i, n in enumerate(node_ids)}
        
        return data
    
    def train_ensemble_models(self, data: Data, epochs: int = 50):
        """Train ensemble of GNN models for robust predictions."""
        model_configs = {
            'gcn': {'class': self.SimpleGCN, 'name': 'Graph Convolutional Network'},
            'sage': {'class': self.GraphSAGE, 'name': 'GraphSAGE'},
            'gat': {'class': self.GAT, 'name': 'Graph Attention Network'}
        }
        
        in_channels = data.num_node_features
        
        for model_type, config in model_configs.items():
            logger.info(f"Training {config['name']}...")
            model = config['class'](in_channels).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            data = data.to(self.device)
            
            best_loss = float('inf')
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                # Sample positive and negative edges
                pos_edge_index, neg_edge_index = self._sample_edges(data)
                
                # Forward pass
                pos_logits = model(data.x, data.edge_index, pos_edge_index)
                neg_logits = model(data.x, data.edge_index, neg_edge_index)
                
                # Labels
                pos_labels = torch.ones(pos_logits.size(0), device=self.device)
                neg_labels = torch.zeros(neg_logits.size(0), device=self.device)
                
                # Loss
                loss = F.binary_cross_entropy_with_logits(
                    torch.cat([pos_logits, neg_logits]), 
                    torch.cat([pos_labels, neg_labels])
                )
                
                loss.backward()
                optimizer.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                
                if epoch % 10 == 0:
                    logger.info(f"[{model_type.upper()}] Epoch {epoch}: Loss {loss.item():.4f}")
            
            self.models[model_type] = model
            self.model_performances[model_type] = 1.0 / (1.0 + best_loss)
    
    def predict_links(self, participants: List[Dict], model_type: str = 'ensemble', top_n: int = 5) -> List[Dict]:
        """
        Predict top-N symbiosis links with explanations.
        model_type can be 'gcn', 'sage', 'gat', or 'ensemble'
        """
        # Create graph
        G = self.create_industrial_graph(participants)
        data = self.nx_to_pyg_enhanced(G)
        
        # Train models if not already trained
        if not self.models:
            self.train_ensemble_models(data)
        
        # Get predictions
        if model_type == 'ensemble':
            predictions = self._ensemble_predict(data, top_n)
        else:
            if model_type not in self.models:
                # Train specific model
                self.train_ensemble_models(data)
            predictions = self._single_model_predict(data, model_type, top_n)
        
        # Format results with explanations
        results = []
        for node1_id, node2_id, score in predictions:
            p1 = next((p for p in participants if p['id'] == node1_id), None)
            p2 = next((p for p in participants if p['id'] == node2_id), None)
            
            if p1 and p2:
                explanation = self._generate_link_explanation(p1, p2, score)
                results.append({
                    'source': node1_id,
                    'target': node2_id,
                    'score': float(score),
                    'confidence': 'High' if score > 0.8 else 'Medium' if score > 0.6 else 'Low',
                    'explanation': explanation,
                    'source_company': p1.get('name', p1.get('company_name', 'Unknown')),
                    'target_company': p2.get('name', p2.get('company_name', 'Unknown')),
                    'potential_benefits': self._calculate_benefits(p1, p2),
                    'model_type': model_type
                })
        
        return results
    
    def detect_multi_hop_symbiosis(self, participants: List[Dict], max_hops: int = 3) -> List[Dict]:
        """Detect multi-hop symbiosis chains for complex industrial networks."""
        G = self.create_industrial_graph(participants)
        data = self.nx_to_pyg_enhanced(G)
        
        if not self.models:
            self.train_ensemble_models(data)
        
        chains = []
        
        # Find all paths up to max_hops
        for start_node in G.nodes():
            visited = set()
            current_chains = self._find_chains(G, start_node, max_hops, visited, [])
            
            for chain in current_chains:
                if len(chain) >= 3:  # At least 3 nodes for multi-hop
                    score = self._evaluate_chain(chain, participants, data)
                    if score > 0.5:
                        chains.append({
                            'chain': chain,
                            'score': float(score),
                            'hops': len(chain) - 1,
                            'total_waste_reduction': self._calculate_chain_waste_reduction(chain, participants),
                            'total_co2_reduction': self._calculate_chain_co2_reduction(chain, participants),
                            'explanation': self._generate_chain_explanation(chain, participants)
                        })
        
        # Sort by score and remove duplicates
        chains = sorted(chains, key=lambda x: x['score'], reverse=True)
        unique_chains = []
        seen = set()
        
        for chain in chains:
            chain_tuple = tuple(sorted(chain['chain']))
            if chain_tuple not in seen:
                seen.add(chain_tuple)
                unique_chains.append(chain)
        
        return unique_chains[:10]  # Top 10 chains
    
    def _find_chains(self, G: nx.Graph, node: str, max_hops: int, visited: set, current_chain: list) -> List[List[str]]:
        """Recursively find all chains from a starting node."""
        if len(current_chain) >= max_hops + 1:
            return [current_chain]
        
        visited.add(node)
        current_chain.append(node)
        
        chains = []
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                chains.extend(self._find_chains(G, neighbor, max_hops, visited.copy(), current_chain.copy()))
        
        # Also return the current chain if it's valid
        if len(current_chain) >= 3:
            chains.append(current_chain)
        
        return chains
    
    def _evaluate_chain(self, chain: List[str], participants: List[Dict], data: Data) -> float:
        """Evaluate the quality of a symbiosis chain."""
        total_score = 0.0
        
        # Evaluate each link in the chain
        for i in range(len(chain) - 1):
            node1_id = chain[i]
            node2_id = chain[i + 1]
            
            # Get participant data
            p1 = next((p for p in participants if p['id'] == node1_id), None)
            p2 = next((p for p in participants if p['id'] == node2_id), None)
            
            if p1 and p2:
                # Material compatibility
                if self._check_material_match(p1, p2):
                    total_score += 0.3
                
                # Industry compatibility
                if self._check_industry_compatibility(p1, p2):
                    total_score += 0.2
                
                # Geographic proximity (simplified)
                if p1.get('location', '') == p2.get('location', ''):
                    total_score += 0.1
        
        # Normalize by chain length
        return total_score / (len(chain) - 1)
    
    def _calculate_chain_waste_reduction(self, chain: List[str], participants: List[Dict]) -> float:
        """Calculate total waste reduction for a chain."""
        total_reduction = 0.0
        
        for node_id in chain:
            p = next((p for p in participants if p['id'] == node_id), None)
            if p:
                waste = p.get('annual_waste', 0)
                total_reduction += waste * 0.3  # Assume 30% reduction
        
        return total_reduction
    
    def _calculate_chain_co2_reduction(self, chain: List[str], participants: List[Dict]) -> float:
        """Calculate total CO2 reduction for a chain."""
        total_reduction = 0.0
        
        for node_id in chain:
            p = next((p for p in participants if p['id'] == node_id), None)
            if p:
                co2 = p.get('carbon_footprint', 0)
                total_reduction += co2 * 0.25  # Assume 25% reduction
        
        return total_reduction
    
    def _generate_chain_explanation(self, chain: List[str], participants: List[Dict]) -> str:
        """Generate explanation for a multi-hop chain."""
        companies = []
        for node_id in chain:
            p = next((p for p in participants if p['id'] == node_id), None)
            if p:
                companies.append(p.get('name', p.get('company_name', 'Unknown')))
        
        chain_str = " → ".join(companies)
        return f"Multi-hop symbiosis chain: {chain_str}. This {len(chain)-1}-hop network enables cascading resource utilization."
    
    def _sample_edges(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample positive and negative edges for training."""
        pos_edge_index = data.edge_index
        num_nodes = data.num_nodes
        
        # Negative sampling
        neg_edges = []
        num_neg = pos_edge_index.size(1)
        
        while len(neg_edges) < num_neg:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            
            if u != v:
                neg_edges.append([u, v])
        
        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
        
        return pos_edge_index, neg_edge_index
    
    def _ensemble_predict(self, data: Data, top_n: int) -> List[Tuple[str, str, float]]:
        """Ensemble prediction using weighted average of all models."""
        all_predictions = {}
        
        for model_type, model in self.models.items():
            predictions = self._single_model_predict(data, model_type, top_n * 2)
            weight = self.model_performances.get(model_type, 1.0)
            
            for node1, node2, score in predictions:
                key = (node1, node2) if node1 < node2 else (node2, node1)
                if key not in all_predictions:
                    all_predictions[key] = []
                all_predictions[key].append(score * weight)
        
        # Average scores
        final_predictions = []
        for (node1, node2), scores in all_predictions.items():
            avg_score = sum(scores) / len(scores)
            final_predictions.append((node1, node2, avg_score))
        
        # Sort and return top N
        final_predictions.sort(key=lambda x: x[2], reverse=True)
        return final_predictions[:top_n]
    
    def _single_model_predict(self, data: Data, model_type: str, top_n: int) -> List[Tuple[str, str, float]]:
        """Predict using a single model."""
        model = self.models[model_type]
        num_nodes = data.num_nodes
        
        # Generate all possible edges
        candidates = []
        existing_edges = set()
        
        for i in range(data.edge_index.size(1)):
            u = int(data.edge_index[0, i])
            v = int(data.edge_index[1, i])
            existing_edges.add((u, v))
            existing_edges.add((v, u))
        
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if (u, v) not in existing_edges:
                    candidates.append((u, v))
        
        if not candidates:
            return []
        
        # Predict scores
        edge_pairs = torch.tensor(candidates, dtype=torch.long).t().contiguous().to(self.device)
        
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(data.x.to(self.device), data.edge_index.to(self.device), edge_pairs)).cpu().numpy()
        
        # Get top N
        predictions = []
        for i, score in enumerate(scores):
            u, v = candidates[i]
            node1 = data.nx_reverse[u]
            node2 = data.nx_reverse[v]
            predictions.append((node1, node2, float(score)))
        
        predictions.sort(key=lambda x: x[2], reverse=True)
        return predictions[:top_n]
    
    def _generate_link_explanation(self, p1: Dict, p2: Dict, score: float) -> str:
        """Generate detailed explanation for a predicted link."""
        explanations = []
        
        # Industry synergy
        if self._check_industry_compatibility(p1, p2):
            explanations.append(f"Strong industry synergy between {p1.get('industry', 'Unknown')} and {p2.get('industry', 'Unknown')}")
        
        # Material match
        if self._check_material_match(p1, p2):
            explanations.append("Direct material exchange opportunity identified")
        
        # Sustainability impact
        total_co2 = p1.get('carbon_footprint', 0) + p2.get('carbon_footprint', 0)
        if total_co2 > 0:
            reduction = total_co2 * 0.25
            explanations.append(f"Potential CO2 reduction of {reduction:.0f} tons/year")
        
        # Confidence
        confidence = 'high' if score > 0.8 else 'medium' if score > 0.6 else 'moderate'
        explanations.append(f"GNN predicts {confidence} compatibility ({score:.1%})")
        
        return ". ".join(explanations)
    
    def _calculate_benefits(self, p1: Dict, p2: Dict) -> Dict[str, float]:
        """Calculate potential benefits of a symbiosis link."""
        waste_reduction = (p1.get('annual_waste', 0) + p2.get('annual_waste', 0)) * 0.3
        co2_reduction = (p1.get('carbon_footprint', 0) + p2.get('carbon_footprint', 0)) * 0.25
        
        # Economic value estimation
        economic_value = waste_reduction * 100  # $100 per ton saved
        
        return {
            'waste_reduction_tons': round(waste_reduction, 2),
            'co2_reduction_tons': round(co2_reduction, 2),
            'economic_value_usd': round(economic_value, 2),
            'sustainability_score': round((waste_reduction + co2_reduction) / 1000, 3)
        }
    
    # Model classes
    class SimpleGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels=64):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.link_pred = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels * 2, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(hidden_channels, 1)
            )
        
        def forward(self, x, edge_index, edge_pairs):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
            
            src, dst = edge_pairs
            x_src = x[src]
            x_dst = x[dst]
            x_combined = torch.cat([x_src, x_dst], dim=-1)
            
            return self.link_pred(x_combined).squeeze(-1)
    
    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels=64):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
            self.link_pred = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels * 2, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(hidden_channels, 1)
            )
        
        def forward(self, x, edge_index, edge_pairs):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
            
            src, dst = edge_pairs
            x_src = x[src]
            x_dst = x[dst]
            x_combined = torch.cat([x_src, x_dst], dim=-1)
            
            return self.link_pred(x_combined).squeeze(-1)
    
    class GAT(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels=64, heads=4):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
            self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
            self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
            self.link_pred = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels * 2, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(hidden_channels, 1)
            )
        
        def forward(self, x, edge_index, edge_pairs):
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.elu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
            
            src, dst = edge_pairs
            x_src = x[src]
            x_dst = x[dst]
            x_combined = torch.cat([x_src, x_dst], dim=-1)
            
            return self.link_pred(x_combined).squeeze(-1)
