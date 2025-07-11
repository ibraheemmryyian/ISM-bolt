import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from datetime import datetime
import pickle
import os
from pathlib import Path

# GNN imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, GraphConv
    from torch_geometric.utils import to_networkx, from_networkx
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    logging.warning("PyTorch Geometric not available. GNN features will be disabled.")

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Advanced Dynamic Knowledge Graph for Industrial Symbiosis
    Features:
    - Graph building and updating
    - Advanced querying and reasoning
    - GNN-based pattern mining
    - Persistent model storage
    - Real-time graph evolution
    """
    
    def __init__(self, model_cache_dir: str = "./models"):
        self.graph = nx.MultiDiGraph()
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # GNN components
        self.gnn_model = None
        self.node_embeddings = {}
        self.edge_embeddings = {}
        self.feature_dim = 128
        
        # Graph statistics
        self.stats = {
            'nodes': 0,
            'edges': 0,
            'last_updated': None,
            'embeddings_generated': False
        }
        
        # Load existing model if available
        self._load_persistent_model()
        
        # Initialize GNN if available
        if GNN_AVAILABLE:
            self._initialize_gnn()
        
        logger.info(f"Knowledge Graph initialized with {self.stats['nodes']} nodes and {self.stats['edges']} edges")

    def _load_persistent_model(self):
        """Load persistent graph and embeddings"""
        try:
            # Load graph structure
            graph_path = self.model_cache_dir / "knowledge_graph.pkl"
            if graph_path.exists():
                with open(graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
                self.stats['nodes'] = self.graph.number_of_nodes()
                self.stats['edges'] = self.graph.number_of_edges()
                logger.info(f"Loaded existing graph with {self.stats['nodes']} nodes")
            
            # Load embeddings
            embeddings_path = self.model_cache_dir / "node_embeddings.pkl"
            if embeddings_path.exists():
                with open(embeddings_path, 'rb') as f:
                    self.node_embeddings = pickle.load(f)
                self.stats['embeddings_generated'] = True
                logger.info(f"Loaded {len(self.node_embeddings)} node embeddings")
                
        except Exception as e:
            logger.error(f"Error loading persistent model: {e}")

    def _save_persistent_model(self):
        """Save graph and embeddings to disk"""
        try:
            # Save graph structure
            graph_path = self.model_cache_dir / "knowledge_graph.pkl"
            with open(graph_path, 'wb') as f:
                pickle.dump(self.graph, f)
            
            # Save embeddings
            if self.node_embeddings:
                embeddings_path = self.model_cache_dir / "node_embeddings.pkl"
                with open(embeddings_path, 'wb') as f:
                    pickle.dump(self.node_embeddings, f)
            
            self.stats['last_updated'] = datetime.now()
            logger.info("Persistent model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving persistent model: {e}")

    def _initialize_gnn(self):
        """Initialize GNN model for graph reasoning"""
        if not GNN_AVAILABLE:
            return
            
        try:
            self.gnn_model = IndustrialSymbiosisGNN(
                input_dim=self.feature_dim,
                hidden_dim=256,
                output_dim=64,
                num_layers=3
            )
            
            # Load pre-trained weights if available
            model_path = self.model_cache_dir / "gnn_model.pth"
            if model_path.exists():
                self.gnn_model.load_state_dict(torch.load(model_path))
                self.gnn_model.eval()
                logger.info("Loaded pre-trained GNN model")
            else:
                logger.info("Initialized new GNN model")
                
        except Exception as e:
            logger.error(f"Error initializing GNN: {e}")

    def add_entity(self, entity_id: str, attributes: Dict[str, Any], entity_type: str = "company"):
        """Add or update an entity node in the graph with enhanced attributes"""
        # Prepare node features
        node_features = self._prepare_node_features(attributes, entity_type)
        
        # Add node with features
        self.graph.add_node(entity_id, 
                           **attributes,
                           features=node_features,
                           entity_type=entity_type,
                           created_at=datetime.now(),
                           updated_at=datetime.now())
        
        self.stats['nodes'] = self.graph.number_of_nodes()
        logger.debug(f"Added entity: {entity_id} ({entity_type})")

    def add_relationship(self, source_id: str, target_id: str, rel_type: str, 
                        attributes: Dict[str, Any] = None, confidence: float = 1.0):
        """Add a relationship (edge) between two entities with confidence scoring"""
        if attributes is None:
            attributes = {}
        
        # Prepare edge features
        edge_features = self._prepare_edge_features(attributes, rel_type)
        
        # Add edge with features
        self.graph.add_edge(source_id, target_id, 
                           key=rel_type,
                           **attributes,
                           features=edge_features,
                           confidence=confidence,
                           created_at=datetime.now())
        
        self.stats['edges'] = self.graph.number_of_edges()
        logger.debug(f"Added relationship: {source_id} --{rel_type}--> {target_id}")

    def _prepare_node_features(self, attributes: Dict[str, Any], entity_type: str) -> np.ndarray:
        """Prepare node features for GNN"""
        features = np.zeros(self.feature_dim)
        
        # Entity type encoding
        type_encodings = {
            'company': 0.1,
            'material': 0.2,
            'location': 0.3,
            'industry': 0.4,
            'regulation': 0.5
        }
        features[0] = type_encodings.get(entity_type, 0.0)
        
        # Numerical features
        if 'annual_waste' in attributes:
            features[1] = min(float(attributes['annual_waste']) / 10000, 1.0)
        if 'carbon_footprint' in attributes:
            features[2] = min(float(attributes['carbon_footprint']) / 100000, 1.0)
        if 'employee_count' in attributes:
            features[3] = min(float(attributes['employee_count']) / 1000, 1.0)
        
        # Categorical features (one-hot encoding)
        if 'industry' in attributes:
            industry_hash = hash(attributes['industry']) % 20
            features[4 + industry_hash] = 1.0
        
        if 'location' in attributes:
            location_hash = hash(attributes['location']) % 20
            features[24 + location_hash] = 1.0
        
        return features

    def _prepare_edge_features(self, attributes: Dict[str, Any], rel_type: str) -> np.ndarray:
        """Prepare edge features for GNN"""
        features = np.zeros(32)  # Smaller edge feature dimension
        
        # Relationship type encoding
        rel_encodings = {
            'supplies': 0.1,
            'consumes': 0.2,
            'located_in': 0.3,
            'regulates': 0.4,
            'partners_with': 0.5,
            'similar_to': 0.6
        }
        features[0] = rel_encodings.get(rel_type, 0.0)
        
        # Numerical edge features
        if 'quantity' in attributes:
            features[1] = min(float(attributes['quantity']) / 1000, 1.0)
        if 'cost' in attributes:
            features[2] = min(float(attributes['cost']) / 10000, 1.0)
        if 'distance' in attributes:
            features[3] = min(float(attributes['distance']) / 1000, 1.0)
        
        return features

    def query(self, query_params: Dict[str, Any]) -> List[Dict]:
        """
        Advanced graph querying with multiple query types
        """
        query_type = query_params.get('type', 'path')
        
        if query_type == 'path':
            return self._path_query(query_params)
        elif query_type == 'pattern':
            return self._pattern_query(query_params)
        elif query_type == 'similarity':
            return self._similarity_query(query_params)
        elif query_type == 'recommendation':
            return self._recommendation_query(query_params)
        else:
            return self._general_query(query_params)

    def _path_query(self, params: Dict[str, Any]) -> List[Dict]:
        """Find paths between entities"""
        source = params.get('source')
        target = params.get('target')
        max_length = params.get('max_length', 3)
        
        if not source or not target:
            return []
        
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            return [{'path': path, 'length': len(path)} for path in paths]
        except nx.NetworkXNoPath:
            return []

    def _pattern_query(self, params: Dict[str, Any]) -> List[Dict]:
        """Find patterns in the graph"""
        pattern_type = params.get('pattern_type', 'triangle')
        
        if pattern_type == 'triangle':
            triangles = list(nx.triangles(self.graph.to_undirected()))
            return [{'pattern': 'triangle', 'nodes': list(triangle)} for triangle in triangles]
        elif pattern_type == 'cycle':
            cycles = list(nx.simple_cycles(self.graph))
            return [{'pattern': 'cycle', 'nodes': cycle} for cycle in cycles]
        
        return []

    def _similarity_query(self, params: Dict[str, Any]) -> List[Dict]:
        """Find similar entities using embeddings"""
        entity_id = params.get('entity_id')
        top_k = params.get('top_k', 5)
        
        if not entity_id or entity_id not in self.node_embeddings:
            return []
        
        target_embedding = self.node_embeddings[entity_id]
        similarities = []
        
        for node_id, embedding in self.node_embeddings.items():
            if node_id != entity_id:
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((node_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [{'entity_id': node_id, 'similarity': float(sim)} 
                for node_id, sim in similarities[:top_k]]

    def _recommendation_query(self, params: Dict[str, Any]) -> List[Dict]:
        """Generate recommendations using GNN"""
        entity_id = params.get('entity_id')
        recommendation_type = params.get('recommendation_type', 'partnership')
        
        if not entity_id or not GNN_AVAILABLE:
            return []
        
        try:
            # Generate embeddings if not available
            if not self.node_embeddings:
                self._generate_embeddings()
            
            # Use GNN for recommendations
            recommendations = self._gnn_recommendations(entity_id, recommendation_type)
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    def _general_query(self, params: Dict[str, Any]) -> List[Dict]:
        """General graph querying"""
        filters = params.get('filters', {})
        limit = params.get('limit', 100)
        
        results = []
        for node, attrs in self.graph.nodes(data=True):
            if self._matches_filters(attrs, filters):
                results.append({'id': node, **attrs})
                if len(results) >= limit:
                    break
        
        return results

    def _matches_filters(self, attrs: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if node attributes match filters"""
        for key, value in filters.items():
            if key not in attrs or attrs[key] != value:
                return False
        return True

    def run_gnn_reasoning(self, reasoning_type: str = "opportunity_discovery") -> Dict[str, Any]:
        """
        Run advanced GNN-based reasoning for pattern mining and opportunity discovery
        """
        if not GNN_AVAILABLE:
            return {'error': 'GNN not available'}
        
        try:
            # Generate embeddings if not available
            if not self.node_embeddings:
                self._generate_embeddings()
            
            # Run specific reasoning tasks
            if reasoning_type == "opportunity_discovery":
                return self._discover_opportunities()
            elif reasoning_type == "network_analysis":
                return self._analyze_network()
            elif reasoning_type == "anomaly_detection":
                return self._detect_anomalies()
            elif reasoning_type == "trend_analysis":
                return self._analyze_trends()
            else:
                return self._general_reasoning()
                
        except Exception as e:
            logger.error(f"Error in GNN reasoning: {e}")
            return {'error': str(e)}

    def _generate_embeddings(self):
        """Generate node embeddings using GNN"""
        if not GNN_AVAILABLE or not self.gnn_model:
            return
        
        try:
            # Convert graph to PyTorch Geometric format
            pyg_data = self._convert_to_pyg()
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.gnn_model.encode(pyg_data.x, pyg_data.edge_index)
                
            # Store embeddings
            for i, node_id in enumerate(self.graph.nodes()):
                self.node_embeddings[node_id] = embeddings[i].numpy()
            
            self.stats['embeddings_generated'] = True
            logger.info(f"Generated embeddings for {len(self.node_embeddings)} nodes")
            
            # Save embeddings
            self._save_persistent_model()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")

    def _convert_to_pyg(self) -> Data:
        """Convert NetworkX graph to PyTorch Geometric format"""
        # Create node feature matrix
        node_features = []
        node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        
        for node in self.graph.nodes():
            features = self.graph.nodes[node].get('features', np.zeros(self.feature_dim))
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edge index
        edge_index = []
        for u, v, k in self.graph.edges(keys=True):
            edge_index.append([node_mapping[u], node_mapping[v]])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)

    def _discover_opportunities(self) -> Dict[str, Any]:
        """Discover industrial symbiosis opportunities using GNN"""
        opportunities = []
        
        # Find potential partnerships
        for node_id, embedding in self.node_embeddings.items():
            node_attrs = self.graph.nodes[node_id]
            if node_attrs.get('entity_type') != 'company':
                continue
            
            # Find similar companies
            similar_companies = self._similarity_query({
                'entity_id': node_id,
                'top_k': 10
            })
            
            for similar in similar_companies:
                similar_id = similar['entity_id']
                similar_attrs = self.graph.nodes[similar_id]
                
                if similar_attrs.get('entity_type') == 'company':
                    # Check for complementary waste/resource patterns
                    opportunity = self._analyze_complementarity(node_id, similar_id)
                    if opportunity['score'] > 0.7:
                        opportunities.append(opportunity)
        
        return {
            'opportunities': opportunities[:20],  # Top 20 opportunities
            'total_found': len(opportunities),
            'reasoning_type': 'opportunity_discovery'
        }

    def _analyze_complementarity(self, company1: str, company2: str) -> Dict[str, Any]:
        """Analyze complementarity between two companies"""
        attrs1 = self.graph.nodes[company1]
        attrs2 = self.graph.nodes[company2]
        
        # Calculate complementarity score
        score = 0.0
        factors = []
        
        # Industry complementarity
        if attrs1.get('industry') != attrs2.get('industry'):
            score += 0.3
            factors.append('different_industries')
        
        # Location proximity
        if attrs1.get('location') == attrs2.get('location'):
            score += 0.2
            factors.append('same_location')
        
        # Waste/resource complementarity (simplified)
        if 'waste_type' in attrs1 and 'resource_needs' in attrs2:
            score += 0.3
            factors.append('waste_resource_match')
        
        # Size complementarity
        size1 = attrs1.get('employee_count', 0)
        size2 = attrs2.get('employee_count', 0)
        if 0.5 <= size1/size2 <= 2.0:
            score += 0.2
            factors.append('compatible_size')
        
        return {
            'company1': company1,
            'company2': company2,
            'score': min(score, 1.0),
            'factors': factors,
            'potential_savings': score * 50000,  # Estimated savings
            'carbon_reduction': score * 100  # Estimated CO2 reduction
        }

    def _analyze_network(self) -> Dict[str, Any]:
        """Analyze network structure and properties"""
        analysis = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph.to_undirected()),
            'connected_components': nx.number_connected_components(self.graph.to_undirected()),
            'centrality_measures': {},
            'community_structure': {}
        }
        
        # Calculate centrality measures
        if self.graph.number_of_nodes() > 0:
            analysis['centrality_measures'] = {
                'degree_centrality': dict(nx.degree_centrality(self.graph)),
                'betweenness_centrality': dict(nx.betweenness_centrality(self.graph)),
                'closeness_centrality': dict(nx.closeness_centrality(self.graph))
            }
        
        # Detect communities
        try:
            communities = list(nx.community.greedy_modularity_communities(self.graph.to_undirected()))
            analysis['community_structure'] = {
                'num_communities': len(communities),
                'community_sizes': [len(c) for c in communities],
                'modularity': nx.community.modularity(self.graph.to_undirected(), communities)
            }
        except:
            analysis['community_structure'] = {'error': 'Could not compute communities'}
        
        return analysis

    def _detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in the network"""
        anomalies = []
        
        # Detect isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            anomalies.append({
                'type': 'isolated_nodes',
                'nodes': isolated_nodes,
                'severity': 'medium',
                'description': 'Nodes with no connections'
            })
        
        # Detect high-degree nodes (potential hubs)
        degree_centrality = nx.degree_centrality(self.graph)
        high_degree_nodes = [node for node, centrality in degree_centrality.items() 
                           if centrality > 0.8]
        if high_degree_nodes:
            anomalies.append({
                'type': 'high_degree_nodes',
                'nodes': high_degree_nodes,
                'severity': 'low',
                'description': 'Nodes with very high connectivity'
            })
        
        return {
            'anomalies': anomalies,
            'total_anomalies': len(anomalies)
        }

    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in the network over time"""
        # This would require temporal data
        # For now, return basic statistics
        return {
            'growth_rate': self.stats['nodes'] / max(1, self.stats['nodes'] - 10),
            'connection_density': self.stats['edges'] / max(1, self.stats['nodes']),
            'average_degree': 2 * self.stats['edges'] / max(1, self.stats['nodes'])
        }

    def _general_reasoning(self) -> Dict[str, Any]:
        """General reasoning about the graph"""
        return {
            'graph_stats': self.stats,
            'embedding_status': self.stats['embeddings_generated'],
            'gnn_available': GNN_AVAILABLE,
            'model_persistent': True
        }

    def _gnn_recommendations(self, entity_id: str, recommendation_type: str) -> List[Dict]:
        """Generate recommendations using GNN embeddings"""
        if entity_id not in self.node_embeddings:
            return [] 
        
        target_embedding = self.node_embeddings[entity_id]
        recommendations = []
        
        for node_id, embedding in self.node_embeddings.items():
            if node_id == entity_id:
                continue
            
            node_attrs = self.graph.nodes[node_id]
            
            # Filter by recommendation type
            if recommendation_type == 'partnership' and node_attrs.get('entity_type') == 'company':
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                )
                recommendations.append({
                    'entity_id': node_id,
                    'similarity': float(similarity),
                    'type': 'partnership',
                    'reasoning': 'High embedding similarity'
                })
        
        # Sort by similarity
        recommendations.sort(key=lambda x: x['similarity'], reverse=True)
        return recommendations[:10]

    def update_graph(self, updates: List[Dict[str, Any]]):
        """Batch update the graph with new data"""
        for update in updates:
            if update['type'] == 'add_entity':
                self.add_entity(update['entity_id'], update['attributes'], update.get('entity_type'))
            elif update['type'] == 'add_relationship':
                self.add_relationship(update['source'], update['target'], 
                                    update['rel_type'], update.get('attributes'))
        
        # Regenerate embeddings after updates
        if updates and GNN_AVAILABLE:
            self._generate_embeddings()
        
        # Save persistent model
        self._save_persistent_model()

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        return {
            **self.stats,
            'node_types': dict(self.graph.nodes(data='entity_type')),
            'edge_types': [edge[2] for edge in self.graph.edges(keys=True)],
            'gnn_available': GNN_AVAILABLE,
            'embeddings_available': bool(self.node_embeddings)
        }

# GNN Model Definition
if GNN_AVAILABLE:
    class IndustrialSymbiosisGNN(nn.Module):
        """Graph Neural Network for Industrial Symbiosis Analysis"""
        
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
            
            # Attention mechanism
            self.attention = GATConv(output_dim, output_dim)
            
            # Output projection
            self.output_proj = nn.Linear(output_dim, output_dim)
            
        def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Encode node features"""
            # Graph convolutions
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=0.2, training=self.training)
            
            # Final convolution
            x = self.convs[-1](x, edge_index)
            
            # Attention
            x = self.attention(x, edge_index)
            
            # Output projection
            x = self.output_proj(x)
            
            return x
        
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Forward pass"""
            return self.encode(x, edge_index)

# Initialize global knowledge graph instance
knowledge_graph = KnowledgeGraph() 