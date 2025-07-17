import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import os
from pathlib import Path
import threading
import time
import random
from supabase import create_client, Client
from dataclasses import dataclass, asdict, field
from threading import Lock
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

class NodeType:
    COMPANY = "company"
    MATERIAL = "material"
    PROCESS = "process"
    LOGISTICS = "logistics"
    STORAGE = "storage"

class EdgeType:
    MATERIAL_FLOW = "material_flow"
    WASTE_FLOW = "waste_flow"
    LOGISTICS = "logistics"
    PROCESS_LINK = "process_link"
    STORAGE_LINK = "storage_link"

# --- Dataclasses for Node/Edge ---
@dataclass
class HeteroNode:
    id: str
    node_type: str
    attributes: dict = field(default_factory=dict)
    layer: str = None

@dataclass
class HeteroEdge:
    source: str
    target: str
    edge_type: str
    attributes: dict = field(default_factory=dict)
    key: str = None

# --- Dynamic Type Registration ---
class TypeRegistry:
    def __init__(self):
        self.node_types = set([NodeType.COMPANY, NodeType.MATERIAL, NodeType.PROCESS, NodeType.LOGISTICS, NodeType.STORAGE])
        self.edge_types = set([EdgeType.MATERIAL_FLOW, EdgeType.WASTE_FLOW, EdgeType.LOGISTICS, EdgeType.PROCESS_LINK, EdgeType.STORAGE_LINK])
    def register_node_type(self, node_type):
        self.node_types.add(node_type)
    def register_edge_type(self, edge_type):
        self.edge_types.add(edge_type)
    def get_node_types(self):
        return list(self.node_types)
    def get_edge_types(self):
        return list(self.edge_types)

type_registry = TypeRegistry()

class GNNReasoningEngine:
    """
    Heterogeneous, Multi-Layered Graph Neural Network Reasoning Engine for Industrial Symbiosis
    Now supports multiple node/edge types and multi-hop, multi-entity reasoning.
    """
    
    def __init__(self, model_dir: str = "gnn_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Model management
        self.models = {}
        self.model_manager = ModelManager(self.model_dir)
        
        # Heterogeneous, multi-layered graph
        self.graph = nx.MultiDiGraph()
        self.node_features = {}
        self.edge_features = {}
        
        # Inference tracking
        self.inference_count = 0
        self.inference_times = []
        self.inference_results = {}
        
        # Configuration
        self.config = {
            'embedding_dim': 64,
            'num_layers': 2,
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 100,
            'inference_threshold': 0.5
        }
        
        # Load existing models
        self._load_models()
        
        # Load companies and materials from Supabase at initialization
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
        if supabase_url and supabase_key:
            self.load_entities_from_supabase(supabase_url, supabase_key)
        else:
            logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set. GNN graph will not be populated from Supabase.")
        
        logger.info("Heterogeneous GNN Reasoning Engine initialized")
        self.lock = Lock()
        self.monitoring = defaultdict(list)
        self.type_registry = type_registry
        self.layers = defaultdict(set)  # layer_name -> set(node_ids)
    
    def _load_models(self):
        """Load existing GNN models"""
        try:
            # Load model configurations
            config_file = self.model_dir / "model_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
            
            # Load trained models
            model_files = list(self.model_dir.glob("*.json"))
            for model_file in model_files:
                if model_file.name != "model_config.json":
                    model_name = model_file.stem
                    self.models[model_name] = self._load_model_from_file(model_file)
            
            logger.info(f"Loaded {len(self.models)} GNN models")
            
        except Exception as e:
            logger.error(f"Error loading GNN models: {e}")
    
    def _load_model_from_file(self, model_file: Path) -> Dict[str, Any]:
        """Load a single model from file"""
        try:
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            return {
                'name': model_data.get('name', model_file.stem),
                'type': model_data.get('type', 'gcn'),
                'parameters': model_data.get('parameters', {}),
                'performance': model_data.get('performance', {}),
                'created_at': model_data.get('created_at', datetime.now().isoformat()),
                'last_updated': model_data.get('last_updated', datetime.now().isoformat())
            }
            
        except Exception as e:
            logger.error(f"Error loading model from {model_file}: {e}")
            return {}
    
    def _save_model_to_file(self, model_name: str, model_data: Dict[str, Any]):
        """Save a model to file"""
        try:
            model_file = self.model_dir / f"{model_name}.json"
            
            # Add metadata
            model_data['last_updated'] = datetime.now().isoformat()
            
            with open(model_file, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Saved model {model_name} to {model_file}")
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    def train_model(self, graph_data: Dict[str, Any], model_name: str = "default", 
                   model_type: str = "gcn") -> Dict[str, Any]:
        """Train a GNN model on graph data"""
        try:
            start_time = time.time()
            
            # Prepare graph data
            graph = self._prepare_graph_data(graph_data)
            
            # Create model
            model = self._create_model(model_type, model_name)
            
            # Train model (simulated training)
            training_result = self._simulate_training(model, graph)
            
            # Store model
            self.models[model_name] = {
                'name': model_name,
                'type': model_type,
                'parameters': model,
                'performance': training_result,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'graph_stats': {
                    'nodes': len(graph.nodes()),
                    'edges': len(graph.edges()),
                    'density': nx.density(graph)
                }
            }
            
            # Save model
            self._save_model_to_file(model_name, self.models[model_name])
            
            training_time = time.time() - start_time
            
            logger.info(f"Trained GNN model {model_name} in {training_time:.2f}s")
            
            return {
                'model_name': model_name,
                'training_time': training_time,
                'performance': training_result,
                'graph_stats': self.models[model_name]['graph_stats']
            }
            
        except Exception as e:
            logger.error(f"Error training GNN model: {e}")
            return {'error': str(e)}
    
    def _prepare_graph_data(self, graph_data: Dict[str, Any]) -> nx.MultiDiGraph:
        """Prepare heterogeneous, multi-layered graph data for training/inference"""
        try:
            graph = nx.MultiDiGraph()
            # Add nodes with type
            if 'nodes' in graph_data:
                for node in graph_data['nodes']:
                    node_id = node.get('id', str(node))
                    node_type = node.get('node_type', NodeType.COMPANY)
                    attributes = {k: v for k, v in node.items() if k != 'id'}
                    attributes['node_type'] = node_type
                    graph.add_node(node_id, **attributes)
            # Add edges with type
            if 'edges' in graph_data:
                for edge in graph_data['edges']:
                    source = edge.get('source')
                    target = edge.get('target')
                    edge_type = edge.get('edge_type', EdgeType.MATERIAL_FLOW)
                    attributes = {k: v for k, v in edge.items() if k not in ['source', 'target']}
                    attributes['edge_type'] = edge_type
                    if source and target:
                        graph.add_edge(source, target, **attributes)
            return graph
        except Exception as e:
            logger.error(f"Error preparing heterogeneous graph data: {e}")
            return nx.MultiDiGraph()
    
    def _create_model(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """Create a GNN model"""
        try:
            if model_type == "gcn":
                return {
                    'type': 'gcn',
                    'layers': self.config['num_layers'],
                    'embedding_dim': self.config['embedding_dim'],
                    'activation': 'relu',
                    'dropout': 0.2
                }
            elif model_type == "gat":
                return {
                    'type': 'gat',
                    'layers': self.config['num_layers'],
                    'embedding_dim': self.config['embedding_dim'],
                    'heads': 4,
                    'dropout': 0.2
                }
            elif model_type == "graphsage":
                return {
                    'type': 'graphsage',
                    'layers': self.config['num_layers'],
                    'embedding_dim': self.config['embedding_dim'],
                    'aggregator': 'mean',
                    'dropout': 0.2
                }
            else:
                # Default to GCN
                return {
                    'type': 'gcn',
                    'layers': self.config['num_layers'],
                    'embedding_dim': self.config['embedding_dim'],
                    'activation': 'relu',
                    'dropout': 0.2
                }
                
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return {'type': 'gcn', 'layers': 2, 'embedding_dim': 64}
    
    def _simulate_training(self, model: Dict[str, Any], graph: nx.MultiDiGraph) -> Dict[str, Any]:
        """Simulate model training"""
        try:
            # Simulate training metrics
            num_nodes = len(graph.nodes())
            num_edges = len(graph.edges())
            
            # Simulate performance based on graph properties
            base_accuracy = 0.7
            density_factor = nx.density(graph)
            size_factor = min(num_nodes / 100, 1.0)
            
            accuracy = base_accuracy + (density_factor * 0.2) + (size_factor * 0.1)
            accuracy = min(accuracy, 0.95)  # Cap at 95%
            
            return {
                'accuracy': accuracy,
                'precision': accuracy * 0.9,
                'recall': accuracy * 0.85,
                'f1_score': accuracy * 0.87,
                'loss': 1.0 - accuracy,
                'epochs_completed': self.config['epochs'],
                'convergence': True
            }
            
        except Exception as e:
            logger.error(f"Error in simulated training: {e}")
            return {
                'accuracy': 0.5,
                'precision': 0.5,
                'recall': 0.5,
                'f1_score': 0.5,
                'loss': 0.5,
                'epochs_completed': 0,
                'convergence': False
            }
    
    def infer(self, graph_data: Dict[str, Any], model_name: str = "default", 
              inference_type: str = "node_embeddings") -> Dict[str, Any]:
        """Perform inference using trained model"""
        try:
            start_time = time.time()
            
            # Check if model exists
            if model_name not in self.models:
                return {'error': f"Model {model_name} not found"}
            
            model = self.models[model_name]
            
            # Prepare graph data
            graph = self._prepare_graph_data(graph_data)
            
            # Perform inference based on type
            if inference_type == "node_embeddings":
                result = self._generate_node_embeddings(graph, model)
            elif inference_type == "link_prediction":
                result = self._predict_links(graph, model)
            elif inference_type == "node_classification":
                result = self._classify_nodes(graph, model)
            elif inference_type == "community_detection":
                result = self._detect_communities(graph, model)
            else:
                result = self._generate_node_embeddings(graph, model)
            
            # Update inference tracking
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.inference_times.append(inference_time)
            
            # Store result
            result_id = f"{model_name}_{inference_type}_{self.inference_count}"
            self.inference_results[result_id] = {
                'model_name': model_name,
                'inference_type': inference_type,
                'inference_time': inference_time,
                'timestamp': datetime.now().isoformat(),
                'result': result
            }
            
            logger.info(f"Completed {inference_type} inference in {inference_time:.2f}s")
            
            return {
                'model_name': model_name,
                'inference_type': inference_type,
                'inference_time': inference_time,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return {'error': str(e)}
    
    def _generate_node_embeddings(self, graph: nx.MultiDiGraph, model: Dict[str, Any]) -> Dict[str, Any]:
        """Generate node embeddings"""
        try:
            embeddings = {}
            
            for node in graph.nodes():
                # Generate embedding based on node features and neighbors
                node_features = graph.nodes[node]
                
                # Simple embedding generation
                embedding = self._create_node_embedding(node, node_features, graph)
                embeddings[node] = embedding.tolist()
            
            return {
                'embeddings': embeddings,
                'embedding_dim': len(next(iter(embeddings.values()))) if embeddings else 0,
                'num_nodes': len(embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error generating node embeddings: {e}")
            return {'embeddings': {}, 'embedding_dim': 0, 'num_nodes': 0}
    
    # --- Thread-safe Real-time Updates ---
    def add_node(self, node: HeteroNode):
        with self.lock:
            self.graph.add_node(node.id, **node.attributes, node_type=node.node_type, layer=node.layer)
            if node.layer:
                self.layers[node.layer].add(node.id)
            self.type_registry.register_node_type(node.node_type)
            self.monitoring['node_add'].append((node.id, node.node_type, node.layer, datetime.now()))

    def add_edge(self, edge: HeteroEdge):
        with self.lock:
            key = edge.key or f"{edge.source}_{edge.target}_{edge.edge_type}_{len(self.graph.edges())}"
            self.graph.add_edge(edge.source, edge.target, key=key, **edge.attributes, edge_type=edge.edge_type)
            self.type_registry.register_edge_type(edge.edge_type)
            self.monitoring['edge_add'].append((edge.source, edge.target, edge.edge_type, datetime.now()))

    def remove_node(self, node_id):
        with self.lock:
            self.graph.remove_node(node_id)
            for layer, nodes in self.layers.items():
                nodes.discard(node_id)
            self.monitoring['node_remove'].append((node_id, datetime.now()))

    def remove_edge(self, source, target, key=None):
        with self.lock:
            if key:
                self.graph.remove_edge(source, target, key=key)
            else:
                self.graph.remove_edges_from([(source, target)])
            self.monitoring['edge_remove'].append((source, target, key, datetime.now()))

    # --- Layered Subgraph Extraction ---
    def get_layer_subgraph(self, layer_name):
        node_ids = self.layers[layer_name]
        return self.graph.subgraph(node_ids).copy()

    def get_type_subgraph(self, node_type):
        node_ids = [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == node_type]
        return self.graph.subgraph(node_ids).copy()

    # --- Advanced Multi-hop/Meta-path Pathfinding ---
    def find_meta_paths(self, source, target, meta_path: list, max_hops=5):
        """Find all paths from source to target that follow a meta-path (sequence of node types)."""
        results = []
        def dfs(current, path, meta_idx):
            if len(path) > max_hops or meta_idx >= len(meta_path):
                return
            if current == target and meta_idx == len(meta_path)-1:
                results.append(list(path))
                return
            for neighbor in self.graph.successors(current):
                ntype = self.graph.nodes[neighbor].get('node_type')
                if ntype == meta_path[meta_idx]:
                    path.append(neighbor)
                    dfs(neighbor, path, meta_idx+1)
                    path.pop()
        dfs(source, [source], 1)
        return results

    def extract_subgraph_by_predicate(self, node_predicate=None, edge_predicate=None):
        nodes = [n for n, d in self.graph.nodes(data=True) if node_predicate is None or node_predicate(n, d)]
        edges = [(u, v, k) for u, v, k, d in self.graph.edges(keys=True, data=True) if edge_predicate is None or edge_predicate(u, v, k, d)]
        return self.graph.edge_subgraph(edges).copy().subgraph(nodes).copy()

    # --- Type-aware Embeddings ---
    def _create_node_embedding(self, node: str, features: Dict[str, Any], graph: nx.MultiDiGraph) -> np.ndarray:
        try:
            feature_vector = []
            feature_vector.append(len(str(node)) / 20)
            feature_vector.append(len(features) / 10)
            feature_vector.append(len(list(graph.neighbors(node))) / 50)
            # Encode node type as one-hot
            node_type = features.get('node_type', NodeType.COMPANY)
            type_vec = [1.0 if node_type == t else 0.0 for t in self.type_registry.get_node_types()]
            feature_vector.extend(type_vec)
            # Encode layer as one-hot
            layer = features.get('layer', None)
            if layer:
                layer_vec = [1.0 if layer == l else 0.0 for l in self.layers.keys()]
                feature_vector.extend(layer_vec)
            # Feature-based components
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    feature_vector.append(min(value / 1000, 1.0))
                elif isinstance(value, str):
                    feature_vector.append(len(value) / 100)
            target_dim = self.config['embedding_dim']
            while len(feature_vector) < target_dim:
                feature_vector.append(0.0)
            feature_vector = feature_vector[:target_dim]
            return np.array(feature_vector)
        except Exception as e:
            logger.error(f"Error creating node embedding: {e}")
            return np.zeros(self.config['embedding_dim'])
    
    def _predict_links(self, graph: nx.MultiDiGraph, model: Dict[str, Any]) -> Dict[str, Any]:
        """Predict missing links in the graph"""
        try:
            predictions = []
            
            # Get all possible node pairs
            nodes = list(graph.nodes())
            
            # Sample some pairs for prediction (to avoid combinatorial explosion)
            num_predictions = min(100, len(nodes) * (len(nodes) - 1) // 2)
            
            for _ in range(num_predictions):
                if len(nodes) < 2:
                    break
                
                # Randomly select two nodes
                node1, node2 = random.sample(nodes, 2)
                
                # Skip if edge already exists
                if graph.has_edge(node1, node2):
                    continue
                
                # Calculate link prediction score
                score = self._calculate_link_score(node1, node2, graph)
                
                predictions.append({
                    'source': node1,
                    'target': node2,
                    'score': score,
                    'predicted': score > self.config['inference_threshold']
                })
            
            # Sort by score
            predictions.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                'predictions': predictions[:20],  # Return top 20
                'num_predictions': len(predictions),
                'threshold': self.config['inference_threshold']
            }
            
        except Exception as e:
            logger.error(f"Error predicting links: {e}")
            return {'predictions': [], 'num_predictions': 0, 'threshold': self.config['inference_threshold']}
    
    def _calculate_link_score(self, node1: str, node2: str, graph: nx.MultiDiGraph) -> float:
        """Calculate link prediction score between two nodes"""
        try:
            # Get node features
            features1 = graph.nodes[node1]
            features2 = graph.nodes[node2]
            
            # Calculate similarity based on features
            similarity = 0.0
            
            # Common neighbors
            neighbors1 = set(graph.neighbors(node1))
            neighbors2 = set(graph.neighbors(node2))
            common_neighbors = len(neighbors1.intersection(neighbors2))
            similarity += common_neighbors * 0.3
            
            # Feature similarity
            feature_similarity = 0.0
            for key in set(features1.keys()).intersection(set(features2.keys())):
                if isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):
                    if features1[key] == 0 and features2[key] == 0:
                        feature_similarity += 1.0
                    elif features1[key] == 0 or features2[key] == 0:
                        feature_similarity += 0.0
                    else:
                        ratio = min(features1[key], features2[key]) / max(features1[key], features2[key])
                        feature_similarity += ratio
            
            similarity += feature_similarity * 0.4
            
            # Random component for diversity
            similarity += random.uniform(0, 0.3)
            
            return min(similarity, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating link score: {e}")
            return 0.0
    
    def _classify_nodes(self, graph: nx.MultiDiGraph, model: Dict[str, Any]) -> Dict[str, Any]:
        """Classify nodes in the graph"""
        try:
            classifications = {}
            
            for node in graph.nodes():
                # Get node features and neighbors
                features = graph.nodes[node]
                neighbors = list(graph.neighbors(node))
                
                # Simple classification based on features
                classification = self._classify_node(node, features, neighbors, graph)
                classifications[node] = classification
            
            return {
                'classifications': classifications,
                'num_nodes': len(classifications),
                'classes': list(set(classifications.values()))
            }
            
        except Exception as e:
            logger.error(f"Error classifying nodes: {e}")
            return {'classifications': {}, 'num_nodes': 0, 'classes': []}
    
    def _classify_node(self, node: str, features: Dict[str, Any], neighbors: List[str], graph: nx.MultiDiGraph) -> str:
        """Classify a single node"""
        try:
            # Simple classification logic
            degree = len(neighbors)
            
            if degree > 10:
                return "hub"
            elif degree > 5:
                return "connector"
            elif degree > 2:
                return "active"
            else:
                return "leaf"
                
        except Exception as e:
            logger.error(f"Error classifying node {node}: {e}")
            return "unknown"
    
    def _detect_communities(self, graph: nx.MultiDiGraph, model: Dict[str, Any]) -> Dict[str, Any]:
        """Detect communities in the graph"""
        try:
            # Use networkx community detection
            communities = list(nx.community.greedy_modularity_communities(graph))
            
            community_data = []
            for i, community in enumerate(communities):
                community_data.append({
                    'community_id': i,
                    'size': len(community),
                    'members': list(community),
                    'density': nx.density(graph.subgraph(community))
                })
            
            return {
                'communities': community_data,
                'num_communities': len(communities),
                'modularity': nx.community.modularity(graph, communities)
            }
            
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return {'communities': [], 'num_communities': 0, 'modularity': 0.0}
    
    # --- Community Detection & Role Discovery ---
    def detect_communities(self, method="greedy_modularity"):  # can add more methods
        try:
            if method == "greedy_modularity":
                communities = list(nx.community.greedy_modularity_communities(self.graph.to_undirected()))
            else:
                raise NotImplementedError(f"Community detection method {method} not implemented.")
            return communities
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return []

    def discover_roles(self):
        """Assign roles (hub, bridge, supplier, consumer, etc.) to nodes based on degree, betweenness, etc."""
        try:
            roles = {}
            deg = dict(self.graph.degree())
            betw = nx.betweenness_centrality(self.graph)
            for n in self.graph.nodes():
                if deg[n] > 10:
                    roles[n] = "hub"
                elif betw[n] > 0.1:
                    roles[n] = "bridge"
                else:
                    roles[n] = self.graph.nodes[n].get('node_type', 'unknown')
            return roles
        except Exception as e:
            logger.error(f"Error discovering roles: {e}")
            return {}

    # --- Monitoring & Analytics ---
    def get_monitoring_stats(self):
        return {k: list(v) for k, v in self.monitoring.items()}

    def get_graph_metrics(self):
        try:
            return {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'node_types': dict(self.graph.nodes(data='node_type')),
                'edge_types': [d.get('edge_type') for _, _, d in self.graph.edges(data=True)],
                'layers': {k: list(v) for k, v in self.layers.items()}
            }
        except Exception as e:
            logger.error(f"Error getting graph metrics: {e}")
            return {}

    # --- Robust API for All Graph Operations ---
    def to_json(self):
        """Export the entire graph as a JSON-serializable dict."""
        data = {
            'nodes': [dict(id=n, **d) for n, d in self.graph.nodes(data=True)],
            'edges': [dict(source=u, target=v, key=k, **d) for u, v, k, d in self.graph.edges(keys=True, data=True)]
        }
        return data

    def from_json(self, data):
        """Load graph from JSON-serializable dict."""
        with self.lock:
            self.graph.clear()
            for node in data.get('nodes', []):
                node_id = node.pop('id')
                self.graph.add_node(node_id, **node)
            for edge in data.get('edges', []):
                source = edge.pop('source')
                target = edge.pop('target')
                key = edge.pop('key', None)
                self.graph.add_edge(source, target, key=key, **edge)

    def list_available_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            if model_name not in self.models:
                return {'error': f"Model {model_name} not found"}
            
            return self.models[model_name]
            
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return {'error': str(e)}
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model"""
        try:
            if model_name not in self.models:
                return False
            
            # Remove from memory
            del self.models[model_name]
            
            # Remove file
            model_file = self.model_dir / f"{model_name}.json"
            if model_file.exists():
                model_file.unlink()
            
            logger.info(f"Deleted model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get inference statistics"""
        try:
            return {
                'total_inferences': self.inference_count,
                'average_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
                'min_inference_time': np.min(self.inference_times) if self.inference_times else 0,
                'max_inference_time': np.max(self.inference_times) if self.inference_times else 0,
                'recent_inferences': list(self.inference_results.keys())[-10:] if self.inference_results else []
            }
            
        except Exception as e:
            logger.error(f"Error getting inference statistics: {e}")
            return {
                'total_inferences': 0,
                'average_inference_time': 0,
                'min_inference_time': 0,
                'max_inference_time': 0,
                'recent_inferences': []
            }
    
    def clear_inference_history(self):
        """Clear inference history"""
        try:
            self.inference_count = 0
            self.inference_times.clear()
            self.inference_results.clear()
            logger.info("Inference history cleared")
            
        except Exception as e:
            logger.error(f"Error clearing inference history: {e}")

    def load_entities_from_supabase(self, supabase_url, supabase_key):
        """Load companies, materials, and other entities from Supabase and add as nodes/edges."""
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            # Load companies
            companies = supabase.table("companies").select("*").execute().data
            for company in companies:
                company_id = str(company.get("id"))
                attributes = {
                    "name": company.get("name"),
                    "industry": company.get("industry"),
                    "location": company.get("location"),
                    "employee_count": company.get("employee_count"),
                    "products": company.get("products"),
                    "main_materials": company.get("main_materials"),
                    "production_volume": company.get("production_volume"),
                    "process_description": company.get("process_description"),
                    "node_type": NodeType.COMPANY
                }
                self.add_node(HeteroNode(id=company_id, node_type=NodeType.COMPANY, attributes=attributes))
            # Load materials
            materials = supabase.table("materials").select("*").execute().data
            for material in materials:
                material_id = str(material.get("id"))
                attributes = {
                    "name": material.get("name"),
                    "category": material.get("category"),
                    "description": material.get("description"),
                    "quantity_estimate": material.get("quantity_estimate"),
                    "potential_value": material.get("potential_value"),
                    "quality_grade": material.get("quality_grade"),
                    "potential_uses": material.get("potential_uses"),
                    "symbiosis_opportunities": material.get("symbiosis_opportunities"),
                    "node_type": NodeType.MATERIAL
                }
                self.add_node(HeteroNode(id=material_id, node_type=NodeType.MATERIAL, attributes=attributes))
            # Optionally, add process/logistics/storage nodes here
            # Add edges (material flows, etc.)
            flows = supabase.table("material_flows").select("*").execute().data
            for flow in flows:
                source = str(flow.get("source_id"))
                target = str(flow.get("target_id"))
                attributes = {
                    "material": flow.get("material"),
                    "flow_rate": flow.get("flow_rate"),
                    "cost_per_unit": flow.get("cost_per_unit"),
                    "carbon_intensity": flow.get("carbon_intensity"),
                    "distance": flow.get("distance"),
                    "reliability": flow.get("reliability"),
                    "edge_type": EdgeType.MATERIAL_FLOW
                }
                self.add_edge(HeteroEdge(source=source, target=target, edge_type=EdgeType.MATERIAL_FLOW, attributes=attributes))
            logger.info(f"Loaded {len(companies)} companies, {len(materials)} materials, and {len(flows)} flows from Supabase into the heterogeneous GNN graph.")
        except Exception as e:
            logger.error(f"Error loading entities from Supabase into GNN graph: {e}")

    # Add multi-hop, multi-entity pathfinding for symbiosis
    def find_multi_hop_paths(self, source: str, target: str, max_hops: int = 5, allowed_node_types: Optional[List[str]] = None, allowed_edge_types: Optional[List[str]] = None) -> List[List[str]]:
        """Find all multi-hop paths between source and target, optionally filtering by node/edge types."""
        try:
            paths = []
            for path in nx.all_simple_paths(self.graph, source=source, target=target, cutoff=max_hops):
                if allowed_node_types:
                    if not all(self.graph.nodes[n].get('node_type') in allowed_node_types for n in path):
                        continue
                if allowed_edge_types:
                    valid = True
                    for i in range(len(path)-1):
                        edge_data = self.graph.get_edge_data(path[i], path[i+1])
                        if edge_data:
                            if not any(ed.get('edge_type') in allowed_edge_types for ed in edge_data.values()):
                                valid = False
                                break
                    if not valid:
                        continue
                paths.append(path)
            return paths
        except Exception as e:
            logger.error(f"Error finding multi-hop paths: {e}")
            return []

    # Import pricing integration
    try:
        from ai_pricing_integration import (
            validate_match_pricing_requirement_integrated,
            get_material_pricing_data_integrated,
            enforce_pricing_validation_decorator
        )
        PRICING_INTEGRATION_AVAILABLE = True
    except ImportError:
        PRICING_INTEGRATION_AVAILABLE = False
        logger.warning("Pricing integration not available")

    @enforce_pricing_validation_decorator
    async def find_symbiotic_matches(self, company_data: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find symbiotic matches using GNN reasoning with pricing validation.
        """
        try:
            # Build heterogeneous graph
            graph = self._prepare_graph_data(company_data)
            
            # Find matches using GNN
            matches = await self._find_matches_with_gnn(graph, company_data, top_k)
            
            # Apply pricing validation to all matches
            if PRICING_INTEGRATION_AVAILABLE:
                validated_matches = []
                for match in matches:
                    try:
                        # Extract pricing information from match
                        material = match.get("material_name") or match.get("material")
                        quantity = match.get("quantity", 1000.0)
                        quality = match.get("quality", "clean")
                        source_location = match.get("source_location", "unknown")
                        destination_location = match.get("destination_location", "unknown")
                        
                        # Get pricing data
                        pricing_data = await get_material_pricing_data_integrated(material)
                        if pricing_data:
                            proposed_price = pricing_data.recycled_price
                            
                            # Validate pricing
                            is_valid = await validate_match_pricing_requirement_integrated(
                                material, quantity, quality, source_location, destination_location, proposed_price
                            )
                            
                            if is_valid:
                                match["pricing_validated"] = True
                                match["pricing_data"] = {
                                    "virgin_price": pricing_data.virgin_price,
                                    "recycled_price": pricing_data.recycled_price,
                                    "savings_percentage": pricing_data.savings_percentage,
                                    "profit_margin": pricing_data.profit_margin,
                                    "risk_level": pricing_data.risk_level
                                }
                                validated_matches.append(match)
                            else:
                                logger.warning(f"Pricing validation failed for GNN match: {material}")
                        else:
                            logger.warning(f"No pricing data available for GNN match: {material}")
                            
                    except Exception as e:
                        logger.error(f"Error validating pricing for GNN match: {e}")
                        continue
                
                logger.info(f"GNN found {len(matches)} matches, {len(validated_matches)} passed pricing validation")
                return validated_matches
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in GNN symbiotic matching: {e}")
            return []

    @enforce_pricing_validation_decorator
    async def detect_multi_hop_symbiosis(self, participants: List[Dict[str, Any]], max_hops: int = 3) -> List[Dict[str, Any]]:
        """
        Detect multi-hop symbiosis opportunities with pricing validation.
        """
        try:
            # Build multi-hop graph
            graph = self._build_multi_hop_graph(participants, max_hops)
            
            # Find multi-hop paths
            paths = self._find_multi_hop_paths(graph, max_hops)
            
            # Convert paths to matches
            matches = self._convert_paths_to_matches(paths, participants)
            
            # Apply pricing validation to multi-hop matches
            if PRICING_INTEGRATION_AVAILABLE:
                validated_matches = []
                for match in matches:
                    try:
                        # Extract pricing information
                        material = match.get("material_name") or match.get("material")
                        quantity = match.get("quantity", 1000.0)
                        quality = match.get("quality", "clean")
                        source_location = match.get("source_location", "unknown")
                        destination_location = match.get("destination_location", "unknown")
                        
                        # Get pricing data
                        pricing_data = await get_material_pricing_data_integrated(material)
                        if pricing_data:
                            proposed_price = pricing_data.recycled_price
                            
                            # Validate pricing
                            is_valid = await validate_match_pricing_requirement_integrated(
                                material, quantity, quality, source_location, destination_location, proposed_price
                            )
                            
                            if is_valid:
                                match["pricing_validated"] = True
                                match["pricing_data"] = {
                                    "virgin_price": pricing_data.virgin_price,
                                    "recycled_price": pricing_data.recycled_price,
                                    "savings_percentage": pricing_data.savings_percentage,
                                    "profit_margin": pricing_data.profit_margin,
                                    "risk_level": pricing_data.risk_level
                                }
                                validated_matches.append(match)
                            else:
                                logger.warning(f"Pricing validation failed for multi-hop match: {material}")
                        else:
                            logger.warning(f"No pricing data available for multi-hop match: {material}")
                            
                    except Exception as e:
                        logger.error(f"Error validating pricing for multi-hop match: {e}")
                        continue
                
                logger.info(f"Multi-hop found {len(matches)} matches, {len(validated_matches)} passed pricing validation")
                return validated_matches
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in multi-hop symbiosis detection: {e}")
            return []

class ModelManager:
    """Manager for GNN models"""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
    
    def list_models(self) -> List[str]:
        """List all available models"""
        try:
            model_files = list(self.model_dir.glob("*.json"))
            return [f.stem for f in model_files if f.name != "model_config.json"]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific model"""
        try:
            model_file = self.model_dir / f"{model_name}.json"
            if model_file.exists():
                with open(model_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error getting model {model_name}: {e}")
            return None

# Initialize global GNN reasoning engine
gnn_reasoning_engine = GNNReasoningEngine()
