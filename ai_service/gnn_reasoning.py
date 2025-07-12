import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.data import Data, Batch
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import pickle
import threading
import time
from collections import defaultdict
import hashlib
import os

logger = logging.getLogger(__name__)

@dataclass
class GNNState:
    """GNN state for persistent reasoning"""
    graph_id: str
    node_embeddings: Dict[str, np.ndarray]
    edge_weights: Dict[Tuple[str, str], float]
    attention_weights: Dict[Tuple[str, str], float]
    reasoning_paths: List[List[str]]
    confidence_scores: Dict[str, float]
    last_updated: datetime
    metadata: Dict[str, Any]

class GNNReasoningEngine:
    """Advanced GNN reasoning engine with persistent state and warm starts"""
    
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Initialize GNN models
        self.gcn_model = self._create_gcn_model()
        self.gat_model = self._create_gat_model()
        self.graph_conv_model = self._create_graph_conv_model()
        
        # State management
        self.graph_states = {}
        self.reasoning_cache = {}
        self.attention_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            'inference_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'reasoning_paths_generated': 0
        }
        
        # Thread safety
        self.state_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        
        # Load persistent state
        self._load_persistent_state()
        
        # Warm start critical components
        self._warm_start_models()
    
    def _create_gcn_model(self) -> nn.Module:
        """Create Graph Convolutional Network model"""
        class GCNModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.conv3 = GCNConv(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.3)
                self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
                self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
            
            def forward(self, x, edge_index, edge_weight=None):
                x = self.conv1(x, edge_index, edge_weight)
                x = self.batch_norm1(x)
                x = F.relu(x)
                x = self.dropout(x)
                
                x = self.conv2(x, edge_index, edge_weight)
                x = self.batch_norm2(x)
                x = F.relu(x)
                x = self.dropout(x)
                
                x = self.conv3(x, edge_index, edge_weight)
                return x
        
        return GCNModel(self.embedding_dim, self.hidden_dim, self.hidden_dim)
    
    def _create_gat_model(self) -> nn.Module:
        """Create Graph Attention Network model"""
        class GATModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, heads=8):
                super().__init__()
                self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.3)
                self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.3)
                self.gat3 = GATConv(hidden_dim, output_dim, heads=1, dropout=0.3)
                self.batch_norm1 = nn.BatchNorm1d(hidden_dim * heads)
                self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
            
            def forward(self, x, edge_index, edge_weight=None):
                x = self.gat1(x, edge_index, edge_weight)
                x = self.batch_norm1(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
                
                x = self.gat2(x, edge_index, edge_weight)
                x = self.batch_norm2(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
                
                x = self.gat3(x, edge_index, edge_weight)
                return x
        
        return GATModel(self.embedding_dim, self.hidden_dim, self.hidden_dim)
    
    def _create_graph_conv_model(self) -> nn.Module:
        """Create Graph Convolution model"""
        class GraphConvModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.conv1 = GraphConv(input_dim, hidden_dim)
                self.conv2 = GraphConv(hidden_dim, hidden_dim)
                self.conv3 = GraphConv(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.3)
                self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
                self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
            
            def forward(self, x, edge_index, edge_weight=None):
                x = self.conv1(x, edge_index, edge_weight)
                x = self.batch_norm1(x)
                x = F.relu(x)
                x = self.dropout(x)
                
                x = self.conv2(x, edge_index, edge_weight)
                x = self.batch_norm2(x)
                x = F.relu(x)
                x = self.dropout(x)
                
                x = self.conv3(x, edge_index, edge_weight)
                return x
        
        return GraphConvModel(self.embedding_dim, self.hidden_dim, self.hidden_dim)
    
    def _warm_start_models(self):
        """Warm start models for optimal performance"""
        try:
            # Create dummy data for warm start
            dummy_x = torch.randn(10, self.embedding_dim)
            dummy_edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
            
            # Warm start each model
            with torch.no_grad():
                self.gcn_model.eval()
                self.gat_model.eval()
                self.graph_conv_model.eval()
                
                _ = self.gcn_model(dummy_x, dummy_edge_index)
                _ = self.gat_model(dummy_x, dummy_edge_index)
                _ = self.graph_conv_model(dummy_x, dummy_edge_index)
            
            logger.info("GNN models warm started successfully")
            
        except Exception as e:
            logger.warning(f"Failed to warm start GNN models: {e}")
    
    def _load_persistent_state(self):
        """Load persistent state from disk"""
        try:
            state_file = "gnn_persistent_state.pkl"
            if os.path.exists(state_file):
                with open(state_file, 'rb') as f:
                    saved_state = pickle.load(f)
                    self.graph_states = saved_state.get('graph_states', {})
                    self.reasoning_cache = saved_state.get('reasoning_cache', {})
                    self.attention_cache = saved_state.get('attention_cache', {})
                logger.info(f"Loaded persistent GNN state with {len(self.graph_states)} graphs")
        except Exception as e:
            logger.warning(f"Failed to load persistent GNN state: {e}")
    
    def _save_persistent_state(self):
        """Save persistent state to disk"""
        try:
            state_file = "gnn_persistent_state.pkl"
            state_data = {
                'graph_states': self.graph_states,
                'reasoning_cache': self.reasoning_cache,
                'attention_cache': self.attention_cache,
                'timestamp': datetime.now()
            }
            with open(state_file, 'wb') as f:
                pickle.dump(state_data, f)
        except Exception as e:
            logger.warning(f"Failed to save persistent GNN state: {e}")
    
    def create_symbiosis_graph(self, companies: List[Dict], matches: List[Dict]) -> str:
        """Create symbiosis graph and return graph ID"""
        graph_id = self._generate_graph_id(companies, matches)
        
        with self.state_lock:
            if graph_id in self.graph_states:
                logger.info(f"Graph {graph_id} already exists, updating...")
                return graph_id
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add company nodes
            for company in companies:
                G.add_node(company['id'], **company)
            
            # Add match edges
            for match in matches:
                G.add_edge(
                    match['company_id'], 
                    match['partner_id'],
                    weight=match['match_score'],
                    **match
                )
            
            # Create node embeddings
            node_embeddings = self._create_node_embeddings(companies)
            
            # Create edge weights
            edge_weights = {}
            for match in matches:
                edge_key = (match['company_id'], match['partner_id'])
                edge_weights[edge_key] = match['match_score']
            
            # Initialize attention weights
            attention_weights = self._initialize_attention_weights(G)
            
            # Create GNN state
            gnn_state = GNNState(
                graph_id=graph_id,
                node_embeddings=node_embeddings,
                edge_weights=edge_weights,
                attention_weights=attention_weights,
                reasoning_paths=[],
                confidence_scores={},
                last_updated=datetime.now(),
                metadata={'num_nodes': len(companies), 'num_edges': len(matches)}
            )
            
            self.graph_states[graph_id] = gnn_state
            self._save_persistent_state()
            
            logger.info(f"Created symbiosis graph {graph_id} with {len(companies)} nodes and {len(matches)} edges")
            return graph_id
    
    def _generate_graph_id(self, companies: List[Dict], matches: List[Dict]) -> str:
        """Generate unique graph ID"""
        data_str = json.dumps({
            'companies': sorted([c['id'] for c in companies]),
            'matches': sorted([(m['company_id'], m['partner_id']) for m in matches])
        }, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def _create_node_embeddings(self, companies: List[Dict]) -> Dict[str, np.ndarray]:
        """Create node embeddings for companies"""
        embeddings = {}
        for company in companies:
            # Create feature vector
            features = [
                company.get('sustainability_score', 0) / 100,
                company.get('employee_count', 0) / 1000,
                company.get('annual_revenue', 0) / 1000000,
                len(company.get('materials_inventory', [])),
                len(company.get('production_processes', []))
            ]
            
            # Pad to embedding dimension
            while len(features) < self.embedding_dim:
                features.append(0.0)
            features = features[:self.embedding_dim]
            
            embeddings[company['id']] = np.array(features, dtype=np.float32)
        
        return embeddings
    
    def _initialize_attention_weights(self, G: nx.Graph) -> Dict[Tuple[str, str], float]:
        """Initialize attention weights for graph edges"""
        attention_weights = {}
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 0.5)
            attention_weights[(u, v)] = weight
            attention_weights[(v, u)] = weight  # Undirected graph
        return attention_weights
    
    def reason_about_symbiosis(self, graph_id: str, query_node: str, 
                             reasoning_type: str = 'path') -> Dict[str, Any]:
        """Perform reasoning about symbiosis relationships"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"{graph_id}:{query_node}:{reasoning_type}"
        with self.cache_lock:
            if cache_key in self.reasoning_cache:
                self.performance_metrics['cache_hits'] += 1
                result = self.reasoning_cache[cache_key]
                result['cached'] = True
                return result
        
        with self.state_lock:
            if graph_id not in self.graph_states:
                raise ValueError(f"Graph {graph_id} not found")
            
            gnn_state = self.graph_states[graph_id]
            
            # Perform reasoning based on type
            if reasoning_type == 'path':
                result = self._path_reasoning(gnn_state, query_node)
            elif reasoning_type == 'influence':
                result = self._influence_reasoning(gnn_state, query_node)
            elif reasoning_type == 'community':
                result = self._community_reasoning(gnn_state, query_node)
            elif reasoning_type == 'optimization':
                result = self._optimization_reasoning(gnn_state, query_node)
            else:
                raise ValueError(f"Unknown reasoning type: {reasoning_type}")
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.performance_metrics['inference_times'].append(inference_time)
            self.performance_metrics['reasoning_paths_generated'] += 1
            
            # Cache result
            with self.cache_lock:
                self.reasoning_cache[cache_key] = result
            
            # Update graph state
            gnn_state.reasoning_paths.append(result.get('reasoning_path', []))
            gnn_state.confidence_scores[query_node] = result.get('confidence', 0.0)
            gnn_state.last_updated = datetime.now()
            
            self._save_persistent_state()
            
            return result
    
    def _path_reasoning(self, gnn_state: GNNState, query_node: str) -> Dict[str, Any]:
        """Perform path-based reasoning"""
        # Create PyTorch Geometric data
        node_ids = list(gnn_state.node_embeddings.keys())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Create node features tensor
        x = torch.tensor([gnn_state.node_embeddings[node_id] for node_id in node_ids], dtype=torch.float)
        
        # Create edge index and weights
        edge_list = []
        edge_weights = []
        for (u, v), weight in gnn_state.edge_weights.items():
            if u in node_to_idx and v in node_to_idx:
                edge_list.append([node_to_idx[u], node_to_idx[v]])
                edge_weights.append(weight)
        
        if not edge_list:
            return {'reasoning_path': [], 'confidence': 0.0, 'explanation': 'No connections found'}
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # Perform GNN inference
        with torch.no_grad():
            gcn_output = self.gcn_model(x, edge_index, edge_weight)
            gat_output = self.gat_model(x, edge_index, edge_weight)
            graph_conv_output = self.graph_conv_model(x, edge_index, edge_weight)
            
            # Ensemble the outputs
            ensemble_output = (gcn_output + gat_output + graph_conv_output) / 3
        
        # Find reasoning paths
        reasoning_paths = self._find_reasoning_paths(gnn_state, query_node, ensemble_output, node_to_idx)
        
        # Calculate confidence
        confidence = self._calculate_path_confidence(reasoning_paths, ensemble_output)
        
        return {
            'reasoning_path': reasoning_paths,
            'confidence': confidence,
            'explanation': self._generate_path_explanation(reasoning_paths),
            'node_embeddings': ensemble_output.numpy().tolist(),
            'attention_weights': gnn_state.attention_weights
        }
    
    def _influence_reasoning(self, gnn_state: GNNState, query_node: str) -> Dict[str, Any]:
        """Perform influence-based reasoning"""
        # Calculate node influence using centrality measures
        G = self._create_networkx_graph(gnn_state)
        
        if query_node not in G.nodes():
            return {'influence_score': 0.0, 'influenced_nodes': [], 'confidence': 0.0}
        
        # Calculate various centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # Calculate influence score
        influence_score = (
            degree_centrality.get(query_node, 0) * 0.3 +
            betweenness_centrality.get(query_node, 0) * 0.3 +
            closeness_centrality.get(query_node, 0) * 0.2 +
            eigenvector_centrality.get(query_node, 0) * 0.2
        )
        
        # Find influenced nodes (neighbors and 2-hop neighbors)
        influenced_nodes = list(G.neighbors(query_node))
        for neighbor in list(G.neighbors(query_node)):
            influenced_nodes.extend(list(G.neighbors(neighbor)))
        influenced_nodes = list(set(influenced_nodes))
        
        return {
            'influence_score': influence_score,
            'influenced_nodes': influenced_nodes,
            'centrality_measures': {
                'degree': degree_centrality.get(query_node, 0),
                'betweenness': betweenness_centrality.get(query_node, 0),
                'closeness': closeness_centrality.get(query_node, 0),
                'eigenvector': eigenvector_centrality.get(query_node, 0)
            },
            'confidence': min(influence_score * 2, 1.0)
        }
    
    def _community_reasoning(self, gnn_state: GNNState, query_node: str) -> Dict[str, Any]:
        """Perform community-based reasoning"""
        G = self._create_networkx_graph(gnn_state)
        
        if query_node not in G.nodes():
            return {'community': [], 'community_score': 0.0, 'confidence': 0.0}
        
        # Detect communities
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # Find community containing query node
        query_community = None
        for community in communities:
            if query_node in community:
                query_community = list(community)
                break
        
        if query_community is None:
            return {'community': [], 'community_score': 0.0, 'confidence': 0.0}
        
        # Calculate community score
        community_score = len(query_community) / len(G.nodes())
        
        # Calculate modularity
        modularity = nx.community.modularity(G, communities)
        
        return {
            'community': query_community,
            'community_score': community_score,
            'modularity': modularity,
            'total_communities': len(communities),
            'confidence': min(community_score * 2, 1.0)
        }
    
    def _optimization_reasoning(self, gnn_state: GNNState, query_node: str) -> Dict[str, Any]:
        """Perform optimization-based reasoning"""
        G = self._create_networkx_graph(gnn_state)
        
        if query_node not in G.nodes():
            return {'optimization_score': 0.0, 'recommendations': [], 'confidence': 0.0}
        
        # Calculate optimization opportunities
        node_degree = G.degree(query_node)
        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
        
        # Optimization score based on degree and edge weights
        optimization_score = 0.0
        recommendations = []
        
        if node_degree < avg_degree:
            optimization_score += 0.3
            recommendations.append("Increase connections to improve network integration")
        
        # Analyze edge weights
        edge_weights = [G[u][v]['weight'] for u, v in G.edges(query_node)]
        if edge_weights:
            avg_weight = sum(edge_weights) / len(edge_weights)
            if avg_weight < 0.7:
                optimization_score += 0.4
                recommendations.append("Strengthen existing connections")
        
        # Check for potential new connections
        potential_connections = []
        for node in G.nodes():
            if node != query_node and not G.has_edge(query_node, node):
                potential_connections.append(node)
        
        if potential_connections:
            optimization_score += 0.3
            recommendations.append(f"Explore connections with {len(potential_connections)} potential partners")
        
        return {
            'optimization_score': min(optimization_score, 1.0),
            'recommendations': recommendations,
            'current_degree': node_degree,
            'average_degree': avg_degree,
            'potential_connections': len(potential_connections),
            'confidence': min(optimization_score * 1.5, 1.0)
        }
    
    def _find_reasoning_paths(self, gnn_state: GNNState, query_node: str, 
                            embeddings: torch.Tensor, node_to_idx: Dict[str, int]) -> List[List[str]]:
        """Find reasoning paths from query node"""
        G = self._create_networkx_graph(gnn_state)
        
        if query_node not in G.nodes():
            return []
        
        # Find all simple paths from query node
        paths = []
        for target in G.nodes():
            if target != query_node:
                try:
                    simple_paths = list(nx.all_simple_paths(G, query_node, target, cutoff=3))
                    paths.extend(simple_paths)
                except nx.NetworkXNoPath:
                    continue
        
        # Sort paths by length and weight
        def path_score(path):
            if len(path) < 2:
                return 0.0
            total_weight = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            return total_weight / len(path)
        
        paths.sort(key=path_score, reverse=True)
        return paths[:10]  # Return top 10 paths
    
    def _calculate_path_confidence(self, paths: List[List[str]], embeddings: torch.Tensor) -> float:
        """Calculate confidence score for reasoning paths"""
        if not paths:
            return 0.0
        
        # Calculate confidence based on path quality
        path_scores = []
        for path in paths:
            if len(path) >= 2:
                # Simple scoring based on path length and embedding similarity
                score = 1.0 / len(path)  # Shorter paths are better
                path_scores.append(score)
        
        if not path_scores:
            return 0.0
        
        return min(sum(path_scores) / len(path_scores), 1.0)
    
    def _generate_path_explanation(self, paths: List[List[str]]) -> str:
        """Generate human-readable explanation for reasoning paths"""
        if not paths:
            return "No reasoning paths found"
        
        path = paths[0]  # Use the best path
        if len(path) < 2:
            return "Single node analysis"
        
        explanation = f"Reasoning path: {' -> '.join(path)}"
        explanation += f" (Length: {len(path)}, Strength: {len(paths)} paths found)"
        
        return explanation
    
    def _create_networkx_graph(self, gnn_state: GNNState) -> nx.Graph:
        """Create NetworkX graph from GNN state"""
        G = nx.Graph()
        
        # Add nodes
        for node_id in gnn_state.node_embeddings.keys():
            G.add_node(node_id)
        
        # Add edges
        for (u, v), weight in gnn_state.edge_weights.items():
            G.add_edge(u, v, weight=weight)
        
        return G
    
    def get_graph_state(self, graph_id: str) -> Optional[GNNState]:
        """Get graph state"""
        return self.graph_states.get(graph_id)
    
    def update_graph_state(self, graph_id: str, updates: Dict[str, Any]):
        """Update graph state"""
        with self.state_lock:
            if graph_id in self.graph_states:
                gnn_state = self.graph_states[graph_id]
                
                if 'node_embeddings' in updates:
                    gnn_state.node_embeddings.update(updates['node_embeddings'])
                
                if 'edge_weights' in updates:
                    gnn_state.edge_weights.update(updates['edge_weights'])
                
                if 'attention_weights' in updates:
                    gnn_state.attention_weights.update(updates['attention_weights'])
                
                gnn_state.last_updated = datetime.now()
                self._save_persistent_state()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            'average_inference_time': np.mean(self.performance_metrics['inference_times']) if self.performance_metrics['inference_times'] else 0.0,
            'total_graphs': len(self.graph_states),
            'cache_hit_rate': self.performance_metrics['cache_hits'] / max(self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'], 1)
        }
    
    def clear_cache(self):
        """Clear reasoning cache"""
        with self.cache_lock:
            self.reasoning_cache.clear()
            self.attention_cache.clear()
        logger.info("GNN reasoning cache cleared")

# Global instance
gnn_reasoning_engine = GNNReasoningEngine()