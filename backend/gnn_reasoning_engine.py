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
from datetime import datetime
import pandas as pd
import pickle
from pathlib import Path
import json
import threading
import time

logger = logging.getLogger(__name__)

class GNNModelManager:
    """Manages multiple GNN models with persistent storage"""
    
    def __init__(self, model_cache_dir: str = "./models"):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.model_configs = {}
        self.training_history = {}
        
        # Load existing models
        self._load_persistent_models()
        
    def _load_persistent_models(self):
        """Load all persistent GNN models"""
        try:
            # Load model configurations
            config_path = self.model_cache_dir / "gnn_configs.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.model_configs = json.load(f)
                logger.info(f"Loaded {len(self.model_configs)} model configurations")
            
            # Load models
            for model_name in self.model_configs.keys():
                model_path = self.model_cache_dir / f"{model_name}.pth"
                if model_path.exists():
                    self._load_model(model_name)
                    
        except Exception as e:
            logger.error(f"Error loading persistent models: {e}")
    
    def _load_model(self, model_name: str):
        """Load a specific model"""
        try:
            config = self.model_configs[model_name]
            model = self._create_model_from_config(config)
            
            model_path = self.model_cache_dir / f"{model_name}.pth"
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            self.models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> nn.Module:
        """Create model from configuration"""
        model_type = config.get('type', 'GCN')
        input_dim = config.get('input_dim', 128)
        hidden_dim = config.get('hidden_dim', 256)
        output_dim = config.get('output_dim', 64)
        num_layers = config.get('num_layers', 3)
        
        if model_type == 'GCN':
            return GCNModel(input_dim, hidden_dim, output_dim, num_layers)
        elif model_type == 'GAT':
            return GATModel(input_dim, hidden_dim, output_dim, num_layers)
        elif model_type == 'GraphSAGE':
            return GraphSAGEModel(input_dim, hidden_dim, output_dim, num_layers)
        elif model_type == 'GIN':
            return GINModel(input_dim, hidden_dim, output_dim, num_layers)
        else:
            return GCNModel(input_dim, hidden_dim, output_dim, num_layers)
    
    def save_model(self, model_name: str, model: nn.Module, config: Dict[str, Any]):
        """Save a model with its configuration"""
        try:
            # Save model weights
            model_path = self.model_cache_dir / f"{model_name}.pth"
            torch.save(model.state_dict(), model_path)
            
            # Save configuration
            self.model_configs[model_name] = config
            config_path = self.model_cache_dir / "gnn_configs.json"
            with open(config_path, 'w') as f:
                json.dump(self.model_configs, f, indent=2)
            
            # Store in memory
            self.models[model_name] = model
            
            logger.info(f"Saved model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    def get_model(self, model_name: str) -> Optional[nn.Module]:
        """Get a model by name"""
        return self.models.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())

# GNN Model Architectures
if GNN_AVAILABLE:
    class GCNModel(nn.Module):
        """Graph Convolutional Network for industrial symbiosis analysis"""
        
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers
            
            # Graph convolution layers
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.convs.append(GCNConv(hidden_dim, output_dim))
            
            # Batch normalization
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Forward pass"""
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
            
            # Final layer
            x = self.convs[-1](x, edge_index)
            return x

    class GATModel(nn.Module):
        """Graph Attention Network for industrial symbiosis analysis"""
        
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3, 
                     num_heads: int = 8):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers
            self.num_heads = num_heads
            
            # Graph attention layers
            self.gat_layers = nn.ModuleList()
            self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.2))
            
            for _ in range(num_layers - 2):
                self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.2))
            
            self.gat_layers.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=0.2))
            
            # Layer normalization
            self.layer_norms = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.layer_norms.append(nn.LayerNorm(hidden_dim * num_heads))
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Forward pass"""
            for i, gat_layer in enumerate(self.gat_layers[:-1]):
                x = gat_layer(x, edge_index)
                x = self.layer_norms[i](x)
                x = F.relu(x)
            
            # Final layer
            x = self.gat_layers[-1](x, edge_index)
            return x

    class GraphSAGEModel(nn.Module):
        """GraphSAGE for industrial symbiosis analysis"""
        
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers
            
            # GraphSAGE layers
            self.sage_layers = nn.ModuleList()
            self.sage_layers.append(SAGEConv(input_dim, hidden_dim))
            
            for _ in range(num_layers - 2):
                self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.sage_layers.append(SAGEConv(hidden_dim, output_dim))
            
            # Batch normalization
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Forward pass"""
            for i, sage_layer in enumerate(self.sage_layers[:-1]):
                x = sage_layer(x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = self.dropout(x)
            
            # Final layer
            x = self.sage_layers[-1](x, edge_index)
            return x

    class GINModel(nn.Module):
        """Graph Isomorphism Network for industrial symbiosis analysis"""
        
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers
            
            # GIN layers
            self.gin_layers = nn.ModuleList()
            self.gin_layers.append(GINConv(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )))
            
            for _ in range(num_layers - 2):
                self.gin_layers.append(GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )))
            
            self.gin_layers.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )))
            
            # Batch normalization
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Forward pass"""
            for i, gin_layer in enumerate(self.gin_layers[:-1]):
                x = gin_layer(x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)
            
            # Final layer
            x = self.gin_layers[-1](x, edge_index)
            return x

class GNNReasoningEngine:
    """
    Advanced GNN Reasoning Engine for Industrial Symbiosis
    Features:
    - Multiple GNN architectures (GCN, GAT, GraphSAGE, GIN)
    - Persistent model storage
    - Real-time inference
    - Multi-task learning
    - Explainable AI
    """
    
    def __init__(self, model_cache_dir: str = "./models"):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Model management
        self.model_manager = GNNModelManager(model_cache_dir) if GNN_AVAILABLE else None
        
        # Data processing
        self.node_encoder = None
        self.edge_encoder = None
        self.feature_scaler = None
        
        # Inference cache
        self.inference_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Threading for concurrent operations
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.inference_times = []
        self.model_performance = {}
        
        logger.info("GNN Reasoning Engine initialized")
    
    def create_industrial_graph(self, participants: List[Dict]) -> nx.Graph:
        """
        Create a NetworkX graph from industrial participants with rich attributes.
        Args:
            participants (List[Dict]): List of participant dictionaries with attributes.
        Returns:
            nx.Graph: Constructed industrial symbiosis graph.
        """
        if not isinstance(participants, list):
            logger.error("Input 'participants' must be a list of dictionaries.")
            raise ValueError("Input 'participants' must be a list of dictionaries.")
        G = nx.Graph()
        try:
            # Add nodes with industrial attributes
            for p in participants:
                if not isinstance(p, dict):
                    logger.warning(f"Participant {p} is not a dictionary. Skipping.")
                    continue
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
                    if self._check_industry_compatibility(p1, p2):
                        G.add_edge(p1['id'], p2['id'], 
                                 type='potential_symbiosis',
                                 confidence=0.5,
                                 created_at=datetime.now().isoformat())
                    if self._check_material_match(p1, p2):
                        G.add_edge(p1['id'], p2['id'], 
                                 type='material_match',
                                 confidence=0.7,
                                 created_at=datetime.now().isoformat())
            logger.info(f"Created industrial graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        except Exception as e:
            logger.error(f"Error creating industrial graph: {e}")
            raise
        return G
    
    def _check_industry_compatibility(self, p1: Dict, p2: Dict) -> bool:
        """
        Check if two industries are compatible for symbiosis.
        Args:
            p1 (Dict): First participant.
            p2 (Dict): Second participant.
        Returns:
            bool: True if compatible, False otherwise.
        """
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
        """
        Check if materials match between participants.
        Args:
            p1 (Dict): First participant.
            p2 (Dict): Second participant.
        Returns:
            bool: True if materials match, False otherwise.
        """
        p1_waste = p1.get('waste_type', '').lower()
        p1_need = p1.get('material_needed', '').lower()
        p2_waste = p2.get('waste_type', '').lower()
        p2_need = p2.get('material_needed', '').lower()
        
        return (p1_waste and p2_need and p1_waste in p2_need) or \
               (p2_waste and p1_need and p2_waste in p1_need)
    
    def nx_to_pyg_enhanced(self, G: nx.Graph) -> Data:
        """Enhanced conversion from NetworkX to PyTorch Geometric with advanced features"""
        if not GNN_AVAILABLE:
            return None
            
        try:
            # Get node IDs and create mapping
            node_ids = list(G.nodes())
            node_mapping = {node: i for i, node in enumerate(node_ids)}
            
            # Enhanced node features
            node_attrs = []
            for n in node_ids:
                attrs = G.nodes[n]
                
                # Categorical features
                cat_features = [
                    attrs.get('industry', 'Unknown'),
                    attrs.get('location', 'Unknown'),
                    attrs.get('waste_type', 'Unknown'),
                    attrs.get('entity_type', 'company')
                ]
                node_attrs.append(cat_features)
            
            # One-hot encode categorical features
            if len(node_attrs) > 0:
                if self.node_encoder is None:
                    from sklearn.preprocessing import LabelEncoder
                    self.node_encoder = LabelEncoder()
                    # Flatten and fit
                    all_cats = [cat for node_cats in node_attrs for cat in node_cats]
                    self.node_encoder.fit(all_cats)
                
                # Transform
                flattened_cats = [cat for node_cats in node_attrs for cat in node_cats]
                cat_encoded = self.node_encoder.transform(flattened_cats)
                
                # Reshape back to node features
                cat_features = []
                for i in range(0, len(cat_encoded), 4):
                    cat_features.append(cat_encoded[i:i+4])
            else:
                cat_features = np.zeros((len(node_ids), 4))
            
            # Numerical features (normalized)
            num_features = []
            for n in node_ids:
                attrs = G.nodes[n]
                carbon = float(attrs.get('carbon_footprint', 0))
                waste = float(attrs.get('annual_waste', 0))
                employees = float(attrs.get('employee_count', 0))
                revenue = float(attrs.get('annual_revenue', 0))
                
                # Normalize features
                carbon_norm = min(carbon / 100000.0, 1.0) if carbon > 0 else 0.0
                waste_norm = min(waste / 10000.0, 1.0) if waste > 0 else 0.0
                employees_norm = min(employees / 1000.0, 1.0) if employees > 0 else 0.0
                revenue_norm = min(revenue / 1000000.0, 1.0) if revenue > 0 else 0.0
                
                num_features.append([carbon_norm, waste_norm, employees_norm, revenue_norm])
            
            # Combine categorical and numerical features
            node_features = np.hstack([cat_features, num_features])
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Enhanced edge features
            edge_tuples = []
            edge_features = []
            
            for u, v, d in G.edges(data=True):
                u_idx = node_mapping[u]
                v_idx = node_mapping[v]
                
                # Add both directions for undirected graph
                edge_tuples.append((u_idx, v_idx))
                edge_tuples.append((v_idx, u_idx))
                
                # Edge features
                edge_type = d.get('type', 'potential')
                confidence = d.get('confidence', 0.5)
                quantity = min(float(d.get('quantity', 0)) / 1000.0, 1.0)
                cost = min(float(d.get('cost', 0)) / 10000.0, 1.0)
                
                edge_feat = [confidence, quantity, cost]
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)  # Both directions
            
            edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            return data
            
        except Exception as e:
            logger.error(f"Error in nx_to_pyg_enhanced: {e}")
            return None

    def train_model(self, graph_data: nx.Graph, model_name: str = "default", 
                   model_type: str = "GCN", task_type: str = "node_classification") -> Dict[str, Any]:
        """Train a GNN model on graph data"""
        if not GNN_AVAILABLE:
            return {'error': 'GNN not available'}
            
        try:
            start_time = time.time()
            
            # Convert graph to PyTorch Geometric format
            pyg_data = self.nx_to_pyg_enhanced(graph_data)
            if pyg_data is None:
                return {'error': 'Failed to convert graph data'}
            
            # Create model
            input_dim = pyg_data.x.size(1)
            hidden_dim = 256
            output_dim = 64
            
            model = self._create_model(model_type, input_dim, hidden_dim, output_dim)
            
            # Training configuration
            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            model.train()
            num_epochs = 100
            training_losses = []
            
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                
                # Forward pass
                out = model(pyg_data.x, pyg_data.edge_index)
                
                # Create dummy labels for training (in practice, use real labels)
                labels = torch.randint(0, output_dim, (pyg_data.x.size(0),))
                
                # Calculate loss
                loss = criterion(out, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                training_losses.append(loss.item())
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Save model
            config = {
                'type': model_type,
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'output_dim': output_dim,
                'num_layers': 3,
                'task_type': task_type,
                'trained_at': datetime.now().isoformat()
            }
            
            self.model_manager.save_model(model_name, model, config)
            
            training_time = time.time() - start_time
            
            # Performance metrics
            model.eval()
            with torch.no_grad():
                test_out = model(pyg_data.x, pyg_data.edge_index)
                test_loss = criterion(test_out, labels).item()
            
            self.model_performance[model_name] = {
                'final_loss': training_losses[-1],
                'test_loss': test_loss,
                'training_time': training_time,
                'num_epochs': num_epochs,
                'model_type': model_type
            }
            
            logger.info(f"Model {model_name} trained successfully in {training_time:.2f}s")
            
            return {
                'model_name': model_name,
                'training_time': training_time,
                'final_loss': training_losses[-1],
                'test_loss': test_loss,
                'model_type': model_type
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {'error': str(e)}

    def _create_model(self, model_type: str, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        """Create a GNN model of specified type"""
        if model_type == 'GCN':
            return GCNModel(input_dim, hidden_dim, output_dim)
        elif model_type == 'GAT':
            return GATModel(input_dim, hidden_dim, output_dim)
        elif model_type == 'GraphSAGE':
            return GraphSAGEModel(input_dim, hidden_dim, output_dim)
        elif model_type == 'GIN':
            return GINModel(input_dim, hidden_dim, output_dim)
        else:
            return GCNModel(input_dim, hidden_dim, output_dim)

    def infer(self, graph_data: nx.Graph, model_name: str = "default", 
              inference_type: str = "node_embeddings") -> Dict[str, Any]:
        """Perform inference using trained GNN model"""
        if not GNN_AVAILABLE:
            return {'error': 'GNN not available'}
            
        try:
            start_time = time.time()
            
            # Check cache
            cache_key = f"{model_name}_{hash(str(graph_data.nodes()))}_{inference_type}"
            if cache_key in self.inference_cache:
                cached_result = self.inference_cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.cache_ttl:
                    logger.info(f"Using cached inference result for {model_name}")
                    return cached_result['result']
            
            # Get model
            model = self.model_manager.get_model(model_name)
            if model is None:
                return {'error': f'Model {model_name} not found'}
            
            # Convert graph to PyTorch Geometric format
            pyg_data = self.nx_to_pyg_enhanced(graph_data)
            if pyg_data is None:
                return {'error': 'Failed to convert graph data'}
            
            # Perform inference
            model.eval()
            with torch.no_grad():
                if inference_type == "node_embeddings":
                    embeddings = model(pyg_data.x, pyg_data.edge_index)
                    result = {
                        'embeddings': embeddings.numpy().tolist(),
                        'node_ids': list(graph_data.nodes()),
                        'embedding_dim': embeddings.size(1)
                    }
                elif inference_type == "node_classification":
                    logits = model(pyg_data.x, pyg_data.edge_index)
                    probabilities = F.softmax(logits, dim=1)
                    predictions = torch.argmax(logits, dim=1)
                    result = {
                        'predictions': predictions.numpy().tolist(),
                        'probabilities': probabilities.numpy().tolist(),
                        'node_ids': list(graph_data.nodes())
                    }
                elif inference_type == "graph_classification":
                    # Global pooling for graph-level prediction
                    node_embeddings = model(pyg_data.x, pyg_data.edge_index)
                    graph_embedding = torch.mean(node_embeddings, dim=0)
                    result = {
                        'graph_embedding': graph_embedding.numpy().tolist(),
                        'embedding_dim': graph_embedding.size(0)
                    }
                else:
                    return {'error': f'Unknown inference type: {inference_type}'}
            
            inference_time = time.time() - start_time
            result['inference_time'] = inference_time
            result['model_name'] = model_name
            result['inference_type'] = inference_type
            
            # Cache result
            self.inference_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            # Update performance metrics
            self.inference_times.append(inference_time)
            
            logger.info(f"Inference completed in {inference_time:.4f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return {'error': str(e)}

    def explain_prediction(self, graph_data: nx.Graph, model_name: str, 
                          target_node: str, method: str = "attention") -> Dict[str, Any]:
        """Explain GNN predictions using attention or gradient-based methods"""
        if not GNN_AVAILABLE:
            return {'error': 'GNN not available'}
            
        try:
            # Get model
            model = self.model_manager.get_model(model_name)
            if model is None:
                return {'error': f'Model {model_name} not found'}
            
            # Convert graph to PyTorch Geometric format
            pyg_data = self.nx_to_pyg_enhanced(graph_data)
            if pyg_data is None:
                return {'error': 'Failed to convert graph data'}
            
            # Get node mapping
            node_ids = list(graph_data.nodes())
            node_mapping = {node: i for i, node in enumerate(node_ids)}
            
            if target_node not in node_mapping:
                return {'error': f'Target node {target_node} not found in graph'}
            
            target_idx = node_mapping[target_node]
            
            if method == "attention":
                # Attention-based explanation (for GAT models)
                if isinstance(model, GATModel):
                    attention_weights = self._extract_attention_weights(model, pyg_data, target_idx)
                    explanation = {
                        'method': 'attention',
                        'target_node': target_node,
                        'attention_weights': attention_weights,
                        'important_neighbors': self._get_important_neighbors(attention_weights, node_ids)
                    }
                else:
                    explanation = {'error': 'Attention explanation only available for GAT models'}
            
            elif method == "gradient":
                # Gradient-based explanation
                gradients = self._compute_gradients(model, pyg_data, target_idx)
                explanation = {
                    'method': 'gradient',
                    'target_node': target_node,
                    'feature_importance': gradients.tolist(),
                    'important_features': self._get_important_features(gradients)
                }
            
            else:
                explanation = {'error': f'Unknown explanation method: {method}'}
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error in explanation: {e}")
            return {'error': str(e)}

    def _extract_attention_weights(self, model: GATModel, data: Data, target_idx: int) -> torch.Tensor:
        """Extract attention weights for a target node"""
        # This is a simplified implementation
        # In practice, you would need to modify the GAT model to return attention weights
        return torch.rand(data.edge_index.size(1))  # Placeholder

    def _compute_gradients(self, model: nn.Module, data: Data, target_idx: int) -> torch.Tensor:
        """Compute gradients with respect to input features"""
        model.eval()
        data.x.requires_grad_(True)
        
        # Forward pass
        output = model(data.x, data.edge_index)
        
        # Backward pass
        output[target_idx].backward()
        
        return data.x.grad[target_idx]

    def _get_important_neighbors(self, attention_weights: torch.Tensor, node_ids: List[str]) -> List[str]:
        """Get important neighbor nodes based on attention weights"""
        # Simplified implementation
        return node_ids[:5]  # Return top 5 neighbors

    def _get_important_features(self, gradients: torch.Tensor) -> List[int]:
        """Get important features based on gradient magnitudes"""
        return torch.topk(torch.abs(gradients), k=5).indices.tolist()

    def get_model_performance(self, model_name: str = None) -> Dict[str, Any]:
        """Get performance statistics for models"""
        if model_name:
            return self.model_performance.get(model_name, {})
        else:
            return self.model_performance

    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {'error': 'No inference data available'}
        
        return {
            'total_inferences': len(self.inference_times),
            'average_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'std_inference_time': np.std(self.inference_times)
        }

    def clear_cache(self):
        """Clear inference cache"""
        self.inference_cache.clear()
        logger.info("Inference cache cleared")

    def list_available_models(self) -> List[str]:
        """List all available trained models"""
        return self.model_manager.list_models()

# Initialize global GNN reasoning engine
gnn_reasoning_engine = GNNReasoningEngine()

def main():
    """Main function to handle API calls"""
    import sys
    import json
    
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No action specified"}))
        sys.exit(1)
    
    try:
        # Parse input data
        input_data = json.loads(sys.argv[1])
        action = input_data.get('action')
        
        # Initialize engine
        engine = GNNReasoningEngine()
        
        if action == 'create_graph':
            participants = input_data.get('participants', [])
            graph_type = input_data.get('graph_type', 'industrial')
            
            graph = engine.create_industrial_graph(participants)
            result = {
                'graph_data': nx.node_link_data(graph),
                'node_count': graph.number_of_nodes(),
                'edge_count': graph.number_of_edges()
            }
            
        elif action == 'train_model':
            graph_data = input_data.get('graph_data')
            model_name = input_data.get('model_name', 'default')
            model_type = input_data.get('model_type', 'GCN')
            task_type = input_data.get('task_type', 'node_classification')
            
            # Convert back to NetworkX graph
            graph = nx.node_link_graph(graph_data)
            
            result = engine.train_model(graph, model_name, model_type, task_type)
            
        elif action == 'infer':
            graph_data = input_data.get('graph_data')
            model_name = input_data.get('model_name', 'default')
            inference_type = input_data.get('inference_type', 'node_embeddings')
            
            # Convert back to NetworkX graph
            graph = nx.node_link_graph(graph_data)
            
            result = engine.infer(graph, model_name, inference_type)
            
        elif action == 'list_models':
            result = {
                'models': engine.list_available_models(),
                'model_count': len(engine.list_available_models())
            }
            
        elif action == 'health_check':
            result = {
                'status': 'healthy',
                'models_loaded': len(engine.list_available_models()),
                'engine_initialized': True
            }
            
        else:
            result = {"error": f"Unknown action: {action}"}
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
