import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime
import os
from pathlib import Path
try:
    import psycopg2
except ImportError:
    psycopg2 = None
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Advanced Knowledge Graph for Industrial Symbiosis
    Features:
    - Entity-relationship modeling
    - Graph-based reasoning
    - Embedding generation
    - Path finding and recommendations
    - Real-time updates
    """
    
    def __init__(self, graph_file: str = "knowledge_graph.json"):
        self.graph_file = graph_file
        self.graph = nx.Graph()
        self.node_embeddings = {}
        self.edge_weights = {}
        self.entity_types = {}
        self.relationship_types = {}
        
        # Statistics
        self.stats = {
            'nodes': 0,
            'edges': 0,
            'embeddings_available': False,
            'last_updated': datetime.now().isoformat()
        }
        
        # Load existing graph or create new one
        self._load_or_create_graph()
        
        logger.info(f"Knowledge Graph initialized with {self.stats['nodes']} nodes and {self.stats['edges']} edges")
    
    def _load_or_create_graph(self):
        """Load existing graph or create new one"""
        try:
            if os.path.exists(self.graph_file):
                with open(self.graph_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct graph from data
                self.graph = nx.node_link_graph(data['graph'])
                self.node_embeddings = data.get('embeddings', {})
                self.edge_weights = data.get('edge_weights', {})
                self.entity_types = data.get('entity_types', {})
                self.relationship_types = data.get('relationship_types', {})
                
                # Update statistics
                self.stats['nodes'] = len(self.graph.nodes())
                self.stats['edges'] = len(self.graph.edges())
                self.stats['embeddings_available'] = len(self.node_embeddings) > 0
                
                logger.info(f"Loaded existing knowledge graph from {self.graph_file}")
            else:
                logger.info("Creating new knowledge graph")
                
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            # Create empty graph
            self.graph = nx.Graph()
    
    def _save_graph(self):
        """Save graph to file"""
        try:
            data = {
                'graph': nx.node_link_data(self.graph),
                'embeddings': self.node_embeddings,
                'edge_weights': self.edge_weights,
                'entity_types': self.entity_types,
                'relationship_types': self.relationship_types,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.graph_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Knowledge graph saved to {self.graph_file}")
            
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
    
    def add_entity(self, entity_id: str, attributes: Dict[str, Any], entity_type: str = "company") -> bool:
        """Add entity to knowledge graph"""
        try:
            # Add node to graph
            self.graph.add_node(entity_id, **attributes)
            
            # Store entity type
            self.entity_types[entity_id] = entity_type
            
            # Generate embedding if possible
            if self._can_generate_embedding(attributes):
                embedding = self._generate_embedding(attributes)
                self.node_embeddings[entity_id] = embedding.tolist()
                self.stats['embeddings_available'] = True
            
            # Update statistics
            self.stats['nodes'] = len(self.graph.nodes())
            self.stats['last_updated'] = datetime.now().isoformat()
            
            # Save graph periodically
            if self.stats['nodes'] % 10 == 0:  # Save every 10 entities
                self._save_graph()
            
            logger.info(f"Added entity {entity_id} of type {entity_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding entity {entity_id}: {e}")
            return False
    
    def add_relationship(self, source_id: str, target_id: str, relationship_type: str, attributes: Dict[str, Any] = None) -> bool:
        """Add relationship between entities"""
        try:
            if attributes is None:
                attributes = {}
            
            # Add edge to graph
            self.graph.add_edge(source_id, target_id, **attributes)
            
            # Store relationship type
            edge_key = (source_id, target_id)
            self.relationship_types[edge_key] = relationship_type
            
            # Calculate edge weight based on relationship type
            weight = self._calculate_edge_weight(relationship_type, attributes)
            self.edge_weights[edge_key] = weight
            
            # Update statistics
            self.stats['edges'] = len(self.graph.edges())
            self.stats['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Added relationship {relationship_type} between {source_id} and {target_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding relationship between {source_id} and {target_id}: {e}")
            return False
    
    def _can_generate_embedding(self, attributes: Dict[str, Any]) -> bool:
        """Check if we can generate embedding for attributes"""
        # Check if we have text data for embedding
        text_fields = ['name', 'description', 'industry', 'location', 'products']
        return any(field in attributes and attributes[field] for field in text_fields)
    
    def _generate_embedding(self, attributes: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for entity attributes"""
        try:
            # Simple embedding generation based on text attributes
            text_parts = []
            
            # Collect text from various fields
            for field in ['name', 'description', 'industry', 'location', 'products']:
                if field in attributes and attributes[field]:
                    text_parts.append(str(attributes[field]))
            
            # Combine all text
            combined_text = " ".join(text_parts)
            
            # Simple feature-based embedding
            embedding = self._text_to_features(combined_text)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros(8)
    
    def _text_to_features(self, text: str) -> np.ndarray:
        """Convert text to feature vector"""
        # Simple feature extraction
        features = []
        
        # Text length (normalized)
        features.append(min(len(text) / 1000, 1.0))
        
        # Word count (normalized)
        word_count = len(text.split())
        features.append(min(word_count / 100, 1.0))
        
        # Unique words (normalized)
        unique_words = len(set(text.lower().split()))
        features.append(min(unique_words / 50, 1.0))
        
        # Industry keywords
        industry_keywords = ['manufacturing', 'chemical', 'food', 'textile', 'construction', 'recycling']
        industry_score = sum(1 for keyword in industry_keywords if keyword in text.lower()) / len(industry_keywords)
        features.append(industry_score)
        
        # Location keywords
        location_keywords = ['dubai', 'abudhabi', 'riyadh', 'doha', 'kuwait', 'oman']
        location_score = sum(1 for keyword in location_keywords if keyword in text.lower()) / len(location_keywords)
        features.append(location_score)
        
        # Material keywords
        material_keywords = ['plastic', 'metal', 'paper', 'organic', 'chemical', 'waste']
        material_score = sum(1 for keyword in material_keywords if keyword in text.lower()) / len(material_keywords)
        features.append(material_score)
        
        # Sustainability keywords
        sustainability_keywords = ['green', 'sustainable', 'renewable', 'eco', 'environmental']
        sustainability_score = sum(1 for keyword in sustainability_keywords if keyword in text.lower()) / len(sustainability_keywords)
        features.append(sustainability_score)
        
        # Overall complexity
        complexity = (word_count * unique_words) / max(1, len(text))
        features.append(min(complexity / 100, 1.0))
        
        return np.array(features)
    
    def _calculate_edge_weight(self, relationship_type: str, attributes: Dict[str, Any]) -> float:
        """Calculate weight for relationship edge"""
        base_weights = {
            'supplies_to': 0.8,
            'buys_from': 0.8,
            'partners_with': 0.9,
            'competes_with': 0.3,
            'located_near': 0.6,
            'same_industry': 0.7,
            'complementary': 0.8
        }
        
        base_weight = base_weights.get(relationship_type, 0.5)
        
        # Adjust weight based on attributes
        if 'strength' in attributes:
            base_weight *= attributes['strength']
        
        if 'frequency' in attributes:
            base_weight *= min(attributes['frequency'] / 10, 1.0)
        
        return min(max(base_weight, 0.0), 1.0)
    
    def find_similar_entities(self, entity_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar entities using graph structure and embeddings"""
        try:
            if entity_id not in self.graph.nodes():
                return []
            
            similarities = []
            
            # Get entity attributes
            entity_attrs = self.graph.nodes[entity_id]
            
            # Compare with all other entities
            for other_id in self.graph.nodes():
                if other_id == entity_id:
                    continue
                
                other_attrs = self.graph.nodes[other_id]
                
                # Calculate similarity using multiple methods
                structural_similarity = self._calculate_structural_similarity(entity_id, other_id)
                attribute_similarity = self._calculate_attribute_similarity(entity_attrs, other_attrs)
                embedding_similarity = self._calculate_embedding_similarity(entity_id, other_id)
                
                # Combine similarities
                combined_similarity = (
                    0.4 * structural_similarity +
                    0.4 * attribute_similarity +
                    0.2 * embedding_similarity
                )
                
                similarities.append({
                    'entity_id': other_id,
                    'similarity': combined_similarity,
                    'structural_similarity': structural_similarity,
                    'attribute_similarity': attribute_similarity,
                    'embedding_similarity': embedding_similarity,
                    'attributes': other_attrs
                })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar entities for {entity_id}: {e}")
            return []
    
    def _calculate_structural_similarity(self, entity1_id: str, entity2_id: str) -> float:
        """Calculate structural similarity between entities"""
        try:
            # Get neighbors
            neighbors1 = set(self.graph.neighbors(entity1_id))
            neighbors2 = set(self.graph.neighbors(entity2_id))
            
            # Jaccard similarity of neighbors
            if neighbors1 or neighbors2:
                intersection = len(neighbors1.intersection(neighbors2))
                union = len(neighbors1.union(neighbors2))
                return intersection / union
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating structural similarity: {e}")
            return 0.0
    
    def _calculate_attribute_similarity(self, attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> float:
        """Calculate attribute similarity between entities"""
        try:
            # Compare common attributes
            common_attrs = set(attrs1.keys()).intersection(set(attrs2.keys()))
            
            if not common_attrs:
                return 0.0
            
            similarities = []
            for attr in common_attrs:
                val1 = attrs1[attr]
                val2 = attrs2[attr]
                
                if isinstance(val1, str) and isinstance(val2, str):
                    # String similarity
                    if val1.lower() == val2.lower():
                        similarities.append(1.0)
                    elif val1.lower() in val2.lower() or val2.lower() in val1.lower():
                        similarities.append(0.7)
                    else:
                        similarities.append(0.0)
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numerical similarity
                    if val1 == 0 and val2 == 0:
                        similarities.append(1.0)
                    elif val1 == 0 or val2 == 0:
                        similarities.append(0.0)
                    else:
                        ratio = min(val1, val2) / max(val1, val2)
                        similarities.append(ratio)
                else:
                    similarities.append(0.0)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating attribute similarity: {e}")
            return 0.0
    
    def _calculate_embedding_similarity(self, entity1_id: str, entity2_id: str) -> float:
        """Calculate embedding similarity between entities"""
        try:
            if entity1_id in self.node_embeddings and entity2_id in self.node_embeddings:
                emb1 = np.array(self.node_embeddings[entity1_id])
                emb2 = np.array(self.node_embeddings[entity2_id])
                
                # Cosine similarity
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 > 0 and norm2 > 0:
                    return dot_product / (norm1 * norm2)
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}")
            return 0.0
    
    def find_paths(self, source_id: str, target_id: str, max_paths: int = 5) -> List[List[str]]:
        """Find paths between two entities"""
        try:
            if source_id not in self.graph.nodes() or target_id not in self.graph.nodes():
                return []
            
            # Find all simple paths
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=5))
            
            # Sort by path length and return top paths
            paths.sort(key=len)
            return paths[:max_paths]
            
        except Exception as e:
            logger.error(f"Error finding paths between {source_id} and {target_id}: {e}")
            return []
    
    def get_entity_neighbors(self, entity_id: str, max_neighbors: int = 10) -> List[Dict[str, Any]]:
        """Get neighbors of an entity"""
        try:
            if entity_id not in self.graph.nodes():
                return []
            
            neighbors = []
            for neighbor_id in self.graph.neighbors(entity_id):
                neighbor_attrs = self.graph.nodes[neighbor_id]
                edge_attrs = self.graph.edges[entity_id, neighbor_id]
                
                neighbors.append({
                    'entity_id': neighbor_id,
                    'attributes': neighbor_attrs,
                    'relationship': edge_attrs,
                    'weight': self.edge_weights.get((entity_id, neighbor_id), 0.5)
                })
            
            # Sort by weight and return top neighbors
            neighbors.sort(key=lambda x: x['weight'], reverse=True)
            return neighbors[:max_neighbors]
            
        except Exception as e:
            logger.error(f"Error getting neighbors for {entity_id}: {e}")
            return []
    
    def run_gnn_reasoning(self, reasoning_type: str = "opportunity_discovery") -> Dict[str, Any]:
        """Run GNN-based reasoning on the graph"""
        try:
            if reasoning_type == "opportunity_discovery":
                return self._discover_opportunities()
            elif reasoning_type == "community_detection":
                return self._detect_communities()
            elif reasoning_type == "centrality_analysis":
                return self._analyze_centrality()
            else:
                return {"error": f"Unknown reasoning type: {reasoning_type}"}
                
        except Exception as e:
            logger.error(f"Error in GNN reasoning: {e}")
            return {"error": str(e)}
    
    def _discover_opportunities(self) -> Dict[str, Any]:
        """Discover business opportunities using graph analysis"""
        try:
            opportunities = []
            
            # Find disconnected components that could be connected
            components = list(nx.connected_components(self.graph))
            
            if len(components) > 1:
                # Look for potential connections between components
                for i, comp1 in enumerate(components):
                    for j, comp2 in enumerate(components[i+1:], i+1):
                        # Find entities that could potentially connect
                        for entity1 in list(comp1)[:3]:  # Limit to first 3 entities
                            for entity2 in list(comp2)[:3]:
                                # Check if they could be compatible
                                if self._could_be_compatible(entity1, entity2):
                                    opportunities.append({
                                        'entity1': entity1,
                                        'entity2': entity2,
                                        'opportunity_type': 'cross_component_connection',
                                        'potential_impact': 'high'
                                    })
            
            # Find high-degree nodes that could be hubs
            high_degree_nodes = [node for node, degree in self.graph.degree() if degree >= 3]
            for node in high_degree_nodes:
                opportunities.append({
                    'entity': node,
                    'opportunity_type': 'hub_expansion',
                    'potential_impact': 'medium'
                })
            
            return {
                'opportunities_found': len(opportunities),
                'opportunities': opportunities[:10],  # Limit to top 10
                'reasoning_type': 'opportunity_discovery'
            }
            
        except Exception as e:
            logger.error(f"Error discovering opportunities: {e}")
            return {"error": str(e)}
    
    def _could_be_compatible(self, entity1: str, entity2: str) -> bool:
        """Check if two entities could be compatible"""
        try:
            attrs1 = self.graph.nodes[entity1]
            attrs2 = self.graph.nodes[entity2]
            
            # Check industry compatibility
            industry1 = attrs1.get('industry', '').lower()
            industry2 = attrs2.get('industry', '').lower()
            
            # Different industries are often complementary
            if industry1 != industry2:
                return True
            
            # Check location compatibility
            location1 = attrs1.get('location', '').lower()
            location2 = attrs2.get('location', '').lower()
            
            # Same location is good for logistics
            if location1 == location2:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking compatibility: {e}")
            return False
    
    def _detect_communities(self) -> Dict[str, Any]:
        """Detect communities in the graph"""
        try:
            # Use Louvain community detection
            communities = nx.community.louvain_communities(self.graph)
            
            community_data = []
            for i, community in enumerate(communities):
                community_data.append({
                    'community_id': i,
                    'size': len(community),
                    'members': list(community),
                    'density': nx.density(self.graph.subgraph(community))
                })
            
            return {
                'communities_found': len(communities),
                'communities': community_data,
                'reasoning_type': 'community_detection'
            }
            
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return {"error": str(e)}
    
    def _analyze_centrality(self) -> Dict[str, Any]:
        """Analyze centrality of nodes"""
        try:
            # Calculate different centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            # Find top nodes by each measure
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'degree_centrality': dict(top_degree),
                'betweenness_centrality': dict(top_betweenness),
                'closeness_centrality': dict(top_closeness),
                'reasoning_type': 'centrality_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing centrality: {e}")
            return {"error": str(e)}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        try:
            stats = self.stats.copy()
            
            # Add more detailed statistics
            if self.graph.nodes():
                stats.update({
                    'average_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()),
                    'density': nx.density(self.graph),
                    'connected_components': nx.number_connected_components(self.graph),
                    'largest_component_size': len(max(nx.connected_components(self.graph), key=len)),
                    'average_clustering': nx.average_clustering(self.graph),
                    'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else None
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return self.stats
    
    def export_graph(self, format: str = "json") -> str:
        """Export graph in specified format"""
        try:
            if format == "json":
                return json.dumps(nx.node_link_data(self.graph), indent=2)
            elif format == "gexf":
                return nx.write_gexf(self.graph, "temp.gexf")
            else:
                return "Unsupported format"
                
        except Exception as e:
            logger.error(f"Error exporting graph: {e}")
            return f"Error: {str(e)}"
    
    def clear_graph(self):
        """Clear the entire graph"""
        try:
            self.graph.clear()
            self.node_embeddings.clear()
            self.edge_weights.clear()
            self.entity_types.clear()
            self.relationship_types.clear()
            
            # Reset statistics
            self.stats = {
                'nodes': 0,
                'edges': 0,
                'embeddings_available': False,
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info("Knowledge graph cleared")
            
        except Exception as e:
            logger.error(f"Error clearing graph: {e}")

    def load_companies_from_supabase(self, supabase_url, supabase_key):
        """Load all companies from Supabase and add them as nodes."""
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            response = supabase.table("companies").select("*").execute()
            companies = response.data
            count = 0
            for company in companies:
                company_id = company.get("id")
                attributes = {
                    "name": company.get("name"),
                    "industry": company.get("industry"),
                    "location": company.get("location"),
                    "employee_count": company.get("employee_count"),
                    "products": company.get("products"),
                    "main_materials": company.get("main_materials"),
                    "production_volume": company.get("production_volume"),
                    "process_description": company.get("process_description")
                }
                self.add_entity(str(company_id), attributes, entity_type="company")
                count += 1
            logger.info(f"Loaded {count} companies from Supabase into the knowledge graph.")
        except Exception as e:
            logger.error(f"Error loading companies from Supabase: {e}")

# Initialize global knowledge graph
knowledge_graph = KnowledgeGraph()
# Load companies from Supabase at startup
_supabase_url = os.getenv("SUPABASE_URL")
_supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
if _supabase_url and _supabase_key:
    knowledge_graph.load_companies_from_supabase(_supabase_url, _supabase_key)
else:
    logger.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY not set. Knowledge graph will not be populated from Supabase.") 