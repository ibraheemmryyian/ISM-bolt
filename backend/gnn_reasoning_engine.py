#!/usr/bin/env python3
"""
WORLD-CLASS GNN REASONING ENGINE
Advanced Graph Neural Networks for sophisticated material matching and reasoning
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    from .fallbacks.torch_geometric_fallback import *
    HAS_TORCH_GEOMETRIC = False.data import Data, HeteroData
from torch_geometric.nn import GCNConv, GATConv, HeteroConv, SAGEConv
from torch_geometric.utils import to_undirected
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchType(Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    SYMBIOTIC = "symbiotic"
    CIRCULAR = "circular"
    INNOVATIVE = "innovative"

@dataclass
class GNNMatch:
    """Advanced GNN match result"""
    source_company_id: str
    source_company_name: str
    source_material_name: str
    target_company_id: str
    target_company_name: str
    target_material_name: str
    match_score: float
    match_type: MatchType
    confidence_score: float
    reasoning_path: List[str]
    graph_features: Dict[str, float]
    similarity_metrics: Dict[str, float]
    business_value: float
    environmental_benefit: float
    technical_feasibility: float
    generated_at: str

class HeteroGNNModel(nn.Module):
    """World-class heterogeneous Graph Neural Network"""
    
    def __init__(self, 
                 hidden_channels: int = 128,
                 out_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 heads: int = 8):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('company', 'supplies', 'material'): GATConv(-1, hidden_channels, heads=heads, dropout=dropout),
                ('material', 'rev_supplies', 'company'): GATConv(-1, hidden_channels, heads=heads, dropout=dropout),
                ('material', 'compatible_with', 'material'): GATConv(-1, hidden_channels, heads=heads, dropout=dropout),
                ('company', 'located_in', 'location'): GATConv(-1, hidden_channels, heads=heads, dropout=dropout),
                ('location', 'rev_located_in', 'company'): GATConv(-1, hidden_channels, heads=heads, dropout=dropout),
                ('material', 'has_type', 'material_type'): GATConv(-1, hidden_channels, heads=heads, dropout=dropout),
                ('material_type', 'rev_has_type', 'material'): GATConv(-1, hidden_channels, heads=heads, dropout=dropout),
            }, aggr='sum')
            self.convs.append(conv)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        
        # Match prediction head
        self.match_head = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
        
        # Match type classification head
        self.type_head = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, len(MatchType)),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Forward pass through heterogeneous GNN"""
        
        # Apply heterogeneous convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Project to output dimensions
        x_dict = {key: self.out_proj(x) for key, x in x_dict.items()}
        
        return x_dict
    
    def predict_match(self, source_embedding, target_embedding):
        """Predict match score and type"""
        # Concatenate embeddings
        combined = torch.cat([source_embedding, target_embedding], dim=-1)
        
        # Predict match score
        match_score = self.match_head(combined)
        
        # Predict match type
        match_type_logits = self.type_head(combined)
        match_type_probs = torch.exp(match_type_logits)
        
        return match_score.squeeze(), match_type_probs

class WorldClassGNNReasoningEngine:
    """World-class GNN reasoning engine for material matching"""
    
    def __init__(self):
        self.logger = logger
        self.model = None
        self.graph_data = None
        self.company_embeddings = {}
        self.material_embeddings = {}
        self.knowledge_graph = self._build_knowledge_graph()
        
        # Configuration
        self.config = {
            'hidden_channels': 128,
            'out_channels': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'heads': 8,
            'min_match_score': 0.6,
            'max_matches_per_material': 15,
            'similarity_threshold': 0.7
        }
    
    def _build_knowledge_graph(self) -> Dict[str, Any]:
        """Build comprehensive knowledge graph for material matching"""
        return {
            'material_compatibility': {
                'steel': ['aluminum', 'copper', 'zinc', 'nickel', 'titanium'],
                'aluminum': ['steel', 'copper', 'magnesium', 'titanium'],
                'plastic': ['glass', 'paper', 'metal', 'wood'],
                'glass': ['plastic', 'metal', 'ceramic'],
                'chemical': ['organic', 'inorganic', 'polymer', 'composite'],
                'organic': ['chemical', 'biological', 'agricultural'],
                'waste': ['recyclable', 'compostable', 'energy_recovery', 'landfill']
            },
            'industry_synergies': {
                'manufacturing': ['automotive', 'aerospace', 'construction', 'electronics'],
                'chemical': ['pharmaceutical', 'agriculture', 'energy', 'materials'],
                'energy': ['chemical', 'manufacturing', 'transportation', 'construction'],
                'agriculture': ['food_processing', 'chemical', 'energy', 'materials'],
                'mining': ['manufacturing', 'energy', 'construction', 'materials'],
                'waste_management': ['energy', 'agriculture', 'manufacturing', 'construction']
            },
            'material_properties': {
                'steel': {'strength': 0.9, 'corrosion_resistance': 0.3, 'recyclability': 0.8, 'density': 7.85},
                'aluminum': {'strength': 0.7, 'corrosion_resistance': 0.8, 'recyclability': 0.9, 'density': 2.7},
                'copper': {'conductivity': 0.95, 'corrosion_resistance': 0.8, 'recyclability': 0.9, 'density': 8.96},
                'plastic': {'strength': 0.5, 'corrosion_resistance': 0.9, 'recyclability': 0.6, 'density': 0.92},
                'glass': {'strength': 0.6, 'corrosion_resistance': 0.9, 'recyclability': 0.8, 'density': 2.5},
                'chemical': {'toxicity': 0.7, 'stability': 0.8, 'reactivity': 0.6, 'density': 1.0}
            },
            'processing_compatibility': {
                'melting': ['steel', 'aluminum', 'copper', 'zinc'],
                'extrusion': ['aluminum', 'plastic', 'rubber'],
                'injection_molding': ['plastic', 'rubber'],
                'casting': ['steel', 'aluminum', 'copper', 'zinc'],
                'rolling': ['steel', 'aluminum', 'copper'],
                'forging': ['steel', 'aluminum', 'titanium']
            },
            'quality_standards': {
                'A': ['aerospace', 'medical', 'electronics'],
                'B': ['automotive', 'construction', 'manufacturing'],
                'C': ['general_industrial', 'packaging'],
                'D': ['waste_management', 'recycling']
            }
        }
    
    async def initialize_model(self):
        """Initialize the world-class GNN model"""
        self.logger.info("🚀 Initializing world-class GNN model...")
        
        try:
            self.model = HeteroGNNModel(
                hidden_channels=self.config['hidden_channels'],
                out_channels=self.config['out_channels'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                heads=self.config['heads']
            )
            
            self.logger.info("✅ World-class GNN model initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize GNN model: {e}")
            return False
    
    async def build_graph_from_data(self, companies: List[Dict[str, Any]], materials: List[Dict[str, Any]]):
        """Build heterogeneous graph from company and material data"""
        self.logger.info("🔗 Building world-class heterogeneous graph...")
        
        try:
            # Create node features
            company_features = self._create_company_features(companies)
            material_features = self._create_material_features(materials)
            location_features = self._create_location_features(companies)
            material_type_features = self._create_material_type_features(materials)
            
            # Create edge indices
            edge_indices = self._create_edge_indices(companies, materials)
            
            # Create heterogeneous data
            self.graph_data = HeteroData()
            
            # Add node features
            self.graph_data['company'].x = torch.tensor(company_features, dtype=torch.float)
            self.graph_data['material'].x = torch.tensor(material_features, dtype=torch.float)
            self.graph_data['location'].x = torch.tensor(location_features, dtype=torch.float)
            self.graph_data['material_type'].x = torch.tensor(material_type_features, dtype=torch.float)
            
            # Add edge indices
            for edge_type, edge_index in edge_indices.items():
                self.graph_data[edge_type].edge_index = torch.tensor(edge_index, dtype=torch.long)
            
            self.logger.info("✅ Heterogeneous graph built successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to build graph: {e}")
            return False
    
    def _create_company_features(self, companies: List[Dict[str, Any]]) -> np.ndarray:
        """Create sophisticated company features"""
        features = []
        
        for company in companies:
            # Basic features
            company_feature = [
                company.get('employee_count', 1000) / 10000,  # Normalized employee count
                len(company.get('materials', [])),  # Number of materials
                len(company.get('waste_streams', [])),  # Number of waste streams
                company.get('sustainability_score', 50) / 100,  # Normalized sustainability score
            ]
            
            # Industry encoding
            industry = company.get('industry', '')
            industry_features = self._encode_industry(industry)
            company_feature.extend(industry_features)
            
            # Location encoding
            location = company.get('location', '')
            location_features = self._encode_location(location)
            company_feature.extend(location_features)
            
            # Energy and environmental features
            energy_needs = company.get('energy_needs', 'moderate')
            water_usage = company.get('water_usage', 'moderate')
            carbon_footprint = company.get('carbon_footprint', 'moderate')
            
            company_feature.extend([
                self._encode_level(energy_needs),
                self._encode_level(water_usage),
                self._encode_level(carbon_footprint)
            ])
            
            features.append(company_feature)
        
        return np.array(features)
    
    def _create_material_features(self, materials: List[Dict[str, Any]]) -> np.ndarray:
        """Create sophisticated material features"""
        features = []
        
        for material in materials:
            material_name = material.get('material_name', '')
            material_type = material.get('material_type', 'unknown')
            
            # Basic features
            material_feature = [
                material.get('quantity', 100) / 1000,  # Normalized quantity
                material.get('potential_value', 1000) / 10000,  # Normalized value
                self._encode_quality_grade(material.get('quality_grade', 'B'))
            ]
            
            # Material type encoding
            type_features = self._encode_material_type(material_type)
            material_feature.extend(type_features)
            
            # Material name encoding
            name_features = self._encode_material_name(material_name)
            material_feature.extend(name_features)
            
            # Properties encoding
            properties = self._get_material_properties(material_name)
            property_features = self._encode_material_properties(properties)
            material_feature.extend(property_features)
            
            features.append(material_feature)
        
        return np.array(features)
    
    def _create_location_features(self, companies: List[Dict[str, Any]]) -> np.ndarray:
        """Create location features"""
        locations = list(set(company.get('location', '') for company in companies))
        features = []
        
        for location in locations:
            location_feature = [
                self._encode_location(location),
                1.0 if 'saudi' in location.lower() else 0.0,  # Saudi Arabia flag
                1.0 if 'uae' in location.lower() else 0.0,  # UAE flag
                1.0 if 'qatar' in location.lower() else 0.0,  # Qatar flag
                1.0 if 'kuwait' in location.lower() else 0.0,  # Kuwait flag
                1.0 if 'bahrain' in location.lower() else 0.0,  # Bahrain flag
                1.0 if 'oman' in location.lower() else 0.0,  # Oman flag
            ]
            features.append(location_feature)
        
        return np.array(features)
    
    def _create_material_type_features(self, materials: List[Dict[str, Any]]) -> np.ndarray:
        """Create material type features"""
        material_types = list(set(material.get('material_type', 'unknown') for material in materials))
        features = []
        
        for material_type in material_types:
            type_feature = self._encode_material_type(material_type)
            features.append(type_feature)
        
        return np.array(features)
    
    def _create_edge_indices(self, companies: List[Dict[str, Any]], materials: List[Dict[str, Any]]) -> Dict[str, List[List[int]]]:
        """Create edge indices for heterogeneous graph"""
        edges = {}
        
        # Company-material edges (supplies)
        company_material_edges = []
        for i, material in enumerate(materials):
            company_id = material.get('company_id', '')
            company_idx = self._get_company_index(company_id, companies)
            if company_idx is not None:
                company_material_edges.append([company_idx, i])
        
        edges[('company', 'supplies', 'material')] = company_material_edges
        edges[('material', 'rev_supplies', 'company')] = [[edge[1], edge[0]] for edge in company_material_edges]
        
        # Material-material edges (compatibility)
        material_material_edges = []
        for i, material1 in enumerate(materials):
            for j, material2 in enumerate(materials):
                if i != j and self._are_materials_compatible(material1, material2):
                    material_material_edges.append([i, j])
        
        edges[('material', 'compatible_with', 'material')] = material_material_edges
        
        # Company-location edges
        company_location_edges = []
        locations = list(set(company.get('location', '') for company in companies))
        for i, company in enumerate(companies):
            location = company.get('location', '')
            location_idx = locations.index(location) if location in locations else 0
            company_location_edges.append([i, location_idx])
        
        edges[('company', 'located_in', 'location')] = company_location_edges
        edges[('location', 'rev_located_in', 'company')] = [[edge[1], edge[0]] for edge in company_location_edges]
        
        # Material-material_type edges
        material_type_edges = []
        material_types = list(set(material.get('material_type', 'unknown') for material in materials))
        for i, material in enumerate(materials):
            material_type = material.get('material_type', 'unknown')
            type_idx = material_types.index(material_type) if material_type in material_types else 0
            material_type_edges.append([i, type_idx])
        
        edges[('material', 'has_type', 'material_type')] = material_type_edges
        edges[('material_type', 'rev_has_type', 'material')] = [[edge[1], edge[0]] for edge in material_type_edges]
        
        return edges
    
    async def find_gnn_matches(self, source_material: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find world-class GNN matches for a material"""
        self.logger.info(f"🧠 Finding GNN matches for: {source_material.get('material_name', 'Unknown')}")
        
        if self.model is None or self.graph_data is None:
            self.logger.warning("⚠️ GNN model or graph data not initialized")
            return []
        
        try:
            # Get source material embedding
            source_embedding = await self._get_material_embedding(source_material)
            if source_embedding is None:
                return []
            
            # Find compatible materials using GNN reasoning
            compatible_materials = await self._find_compatible_materials(source_material)
            
            # Generate matches
            matches = []
            for target_material in compatible_materials:
                target_embedding = await self._get_material_embedding(target_material)
                if target_embedding is not None:
                    match = await self._create_gnn_match(source_material, target_material, source_embedding, target_embedding)
                    if match and match.match_score >= self.config['min_match_score']:
                        matches.append(match)
            
            # Sort by match score and limit results
            matches.sort(key=lambda x: x.match_score, reverse=True)
            matches = matches[:self.config['max_matches_per_material']]
            
            # Convert to dictionary format
            match_dicts = []
            for match in matches:
                match_dict = {
                    'source_company_id': match.source_company_id,
                    'source_company_name': match.source_company_name,
                    'source_material_name': match.source_material_name,
                    'target_company_id': match.target_company_id,
                    'target_company_name': match.target_company_name,
                    'target_material_name': match.target_material_name,
                    'match_score': match.match_score,
                    'match_type': match.match_type.value,
                    'potential_value': match.business_value,
                    'ai_generated': True,
                    'generated_at': match.generated_at
                }
                match_dicts.append(match_dict)
            
            self.logger.info(f"✅ Found {len(match_dicts)} GNN matches")
            return match_dicts
            
        except Exception as e:
            self.logger.error(f"❌ Error finding GNN matches: {e}")
            return []
    
    async def _get_material_embedding(self, material: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Get material embedding from GNN"""
        try:
            # This would typically use the trained GNN model
            # For now, create a synthetic embedding based on material properties
            material_name = material.get('material_name', '')
            material_type = material.get('material_type', 'unknown')
            
            # Create feature vector
            features = []
            
            # Material type encoding
            type_features = self._encode_material_type(material_type)
            features.extend(type_features)
            
            # Material name encoding
            name_features = self._encode_material_name(material_name)
            features.extend(name_features)
            
            # Properties encoding
            properties = self._get_material_properties(material_name)
            property_features = self._encode_material_properties(properties)
            features.extend(property_features)
            
            # Normalize features
            features = np.array(features)
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            return torch.tensor(features, dtype=torch.float)
            
        except Exception as e:
            self.logger.error(f"❌ Error getting material embedding: {e}")
            return None
    
    async def _find_compatible_materials(self, source_material: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find compatible materials using knowledge graph"""
        compatible_materials = []
        source_material_name = source_material.get('material_name', '').lower()
        
        # Check material compatibility
        for material_type, compatible_types in self.knowledge_graph['material_compatibility'].items():
            if material_type in source_material_name:
                # Find materials of compatible types
                for compatible_type in compatible_types:
                    # This would typically query the material database
                    # For now, return empty list
                    pass
        
        return compatible_materials
    
    async def _create_gnn_match(self, source_material: Dict[str, Any], target_material: Dict[str, Any], 
                              source_embedding: torch.Tensor, target_embedding: torch.Tensor) -> Optional[GNNMatch]:
        """Create sophisticated GNN match"""
        try:
            # Calculate similarity metrics
            similarity_metrics = self._calculate_similarity_metrics(source_material, target_material, source_embedding, target_embedding)
            
            # Predict match score and type
            match_score, match_type_probs = self.model.predict_match(source_embedding, target_embedding)
            match_score = match_score.item()
            
            # Determine match type
            match_type_idx = torch.argmax(match_type_probs).item()
            match_type = list(MatchType)[match_type_idx]
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(similarity_metrics, match_score)
            
            # Generate reasoning path
            reasoning_path = self._generate_reasoning_path(source_material, target_material, similarity_metrics)
            
            # Calculate business value
            business_value = self._calculate_business_value(source_material, target_material, match_score)
            
            # Calculate environmental benefit
            environmental_benefit = self._calculate_environmental_benefit(source_material, target_material)
            
            # Calculate technical feasibility
            technical_feasibility = self._calculate_technical_feasibility(source_material, target_material)
            
            return GNNMatch(
                source_company_id=source_material.get('company_id', ''),
                source_company_name=source_material.get('company_name', ''),
                source_material_name=source_material.get('material_name', ''),
                target_company_id=target_material.get('company_id', ''),
                target_company_name=target_material.get('company_name', ''),
                target_material_name=target_material.get('material_name', ''),
                match_score=match_score,
                match_type=match_type,
                confidence_score=confidence_score,
                reasoning_path=reasoning_path,
                graph_features=similarity_metrics,
                similarity_metrics=similarity_metrics,
                business_value=business_value,
                environmental_benefit=environmental_benefit,
                technical_feasibility=technical_feasibility,
                generated_at=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error creating GNN match: {e}")
            return None
    
    def _calculate_similarity_metrics(self, source_material: Dict[str, Any], target_material: Dict[str, Any], 
                                    source_embedding: torch.Tensor, target_embedding: torch.Tensor) -> Dict[str, float]:
        """Calculate comprehensive similarity metrics"""
        metrics = {}
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(source_embedding.unsqueeze(0), target_embedding.unsqueeze(0)).item()
        metrics['cosine_similarity'] = cos_sim
        
        # Euclidean distance
        euclidean_dist = torch.norm(source_embedding - target_embedding).item()
        metrics['euclidean_distance'] = euclidean_dist
        
        # Material type similarity
        source_type = source_material.get('material_type', 'unknown')
        target_type = target_material.get('material_type', 'unknown')
        type_similarity = 1.0 if source_type == target_type else 0.0
        metrics['type_similarity'] = type_similarity
        
        # Quality grade similarity
        source_grade = source_material.get('quality_grade', 'B')
        target_grade = target_material.get('quality_grade', 'B')
        grade_similarity = 1.0 if source_grade == target_grade else 0.5
        metrics['grade_similarity'] = grade_similarity
        
        # Value similarity
        source_value = source_material.get('potential_value', 0)
        target_value = target_material.get('potential_value', 0)
        if source_value > 0 and target_value > 0:
            value_similarity = 1.0 - abs(source_value - target_value) / max(source_value, target_value)
        else:
            value_similarity = 0.5
        metrics['value_similarity'] = value_similarity
        
        return metrics
    
    def _calculate_confidence_score(self, similarity_metrics: Dict[str, float], match_score: float) -> float:
        """Calculate confidence score for the match"""
        # Base confidence from match score
        confidence = match_score * 0.6
        
        # Add similarity-based confidence
        cos_sim = similarity_metrics.get('cosine_similarity', 0.0)
        confidence += cos_sim * 0.2
        
        # Add type similarity confidence
        type_sim = similarity_metrics.get('type_similarity', 0.0)
        confidence += type_sim * 0.1
        
        # Add grade similarity confidence
        grade_sim = similarity_metrics.get('grade_similarity', 0.0)
        confidence += grade_sim * 0.1
        
        return min(1.0, confidence)
    
    def _generate_reasoning_path(self, source_material: Dict[str, Any], target_material: Dict[str, Any], 
                               similarity_metrics: Dict[str, float]) -> List[str]:
        """Generate reasoning path for the match"""
        reasoning = []
        
        # Material type reasoning
        source_type = source_material.get('material_type', 'unknown')
        target_type = target_material.get('material_type', 'unknown')
        if source_type == target_type:
            reasoning.append(f"Same material type: {source_type}")
        else:
            reasoning.append(f"Different material types: {source_type} vs {target_type}")
        
        # Quality reasoning
        source_grade = source_material.get('quality_grade', 'B')
        target_grade = target_material.get('quality_grade', 'B')
        if source_grade == target_grade:
            reasoning.append(f"Same quality grade: {source_grade}")
        else:
            reasoning.append(f"Quality grade difference: {source_grade} vs {target_grade}")
        
        # Similarity reasoning
        cos_sim = similarity_metrics.get('cosine_similarity', 0.0)
        if cos_sim > 0.8:
            reasoning.append("High feature similarity")
        elif cos_sim > 0.6:
            reasoning.append("Moderate feature similarity")
        else:
            reasoning.append("Low feature similarity")
        
        return reasoning
    
    def _calculate_business_value(self, source_material: Dict[str, Any], target_material: Dict[str, Any], match_score: float) -> float:
        """Calculate business value of the match"""
        source_value = source_material.get('potential_value', 0)
        target_value = target_material.get('potential_value', 0)
        
        # Base value
        base_value = min(source_value, target_value) * 0.1
        
        # Match score multiplier
        value_multiplier = 1.0 + match_score * 0.5
        
        return base_value * value_multiplier
    
    def _calculate_environmental_benefit(self, source_material: Dict[str, Any], target_material: Dict[str, Any]) -> float:
        """Calculate environmental benefit of the match"""
        benefit = 0.5  # Base benefit
        
        # Waste-to-resource conversion
        source_name = source_material.get('material_name', '').lower()
        target_name = target_material.get('material_name', '').lower()
        
        if 'waste' in source_name and 'waste' not in target_name:
            benefit += 0.3
        
        # Recyclable materials
        if any(word in source_name for word in ['steel', 'aluminum', 'copper', 'plastic']):
            benefit += 0.2
        
        return min(1.0, benefit)
    
    def _calculate_technical_feasibility(self, source_material: Dict[str, Any], target_material: Dict[str, Any]) -> float:
        """Calculate technical feasibility of the match"""
        feasibility = 0.7  # Base feasibility
        
        # Same material type increases feasibility
        source_type = source_material.get('material_type', 'unknown')
        target_type = target_material.get('material_type', 'unknown')
        if source_type == target_type:
            feasibility += 0.2
        
        # Similar quality grades increase feasibility
        source_grade = source_material.get('quality_grade', 'B')
        target_grade = target_material.get('quality_grade', 'B')
        if source_grade == target_grade:
            feasibility += 0.1
        
        return min(1.0, feasibility)
    
    # Helper methods for feature encoding
    def _encode_industry(self, industry: str) -> List[float]:
        """Encode industry as feature vector"""
        industries = ['manufacturing', 'chemical', 'energy', 'mining', 'waste_management', 'automotive', 'aerospace']
        encoding = [0.0] * len(industries)
        
        for i, ind in enumerate(industries):
            if ind in industry.lower():
                encoding[i] = 1.0
        
        return encoding
    
    def _encode_location(self, location: str) -> List[float]:
        """Encode location as feature vector"""
        locations = ['saudi_arabia', 'uae', 'qatar', 'kuwait', 'bahrain', 'oman']
        encoding = [0.0] * len(locations)
        
        for i, loc in enumerate(locations):
            if loc.replace('_', ' ') in location.lower():
                encoding[i] = 1.0
        
        return encoding
    
    def _encode_level(self, level: str) -> float:
        """Encode level (high, moderate, low) as float"""
        level_mapping = {'high': 1.0, 'moderate': 0.5, 'low': 0.0}
        return level_mapping.get(level.lower(), 0.5)
    
    def _encode_quality_grade(self, grade: str) -> float:
        """Encode quality grade as float"""
        grade_mapping = {'A': 1.0, 'B': 0.75, 'C': 0.5, 'D': 0.25}
        return grade_mapping.get(grade, 0.5)
    
    def _encode_material_type(self, material_type: str) -> List[float]:
        """Encode material type as feature vector"""
        types = ['metal', 'plastic', 'glass', 'chemical', 'organic', 'waste', 'composite']
        encoding = [0.0] * len(types)
        
        for i, type_name in enumerate(types):
            if type_name in material_type.lower():
                encoding[i] = 1.0
        
        return encoding
    
    def _encode_material_name(self, material_name: str) -> List[float]:
        """Encode material name as feature vector"""
        features = [
            1.0 if 'steel' in material_name.lower() else 0.0,
            1.0 if 'aluminum' in material_name.lower() else 0.0,
            1.0 if 'copper' in material_name.lower() else 0.0,
            1.0 if 'plastic' in material_name.lower() else 0.0,
            1.0 if 'chemical' in material_name.lower() else 0.0,
            1.0 if 'waste' in material_name.lower() else 0.0,
            1.0 if 'scrap' in material_name.lower() else 0.0,
            1.0 if 'premium' in material_name.lower() else 0.0,
            1.0 if 'industrial' in material_name.lower() else 0.0
        ]
        return features
    
    def _get_material_properties(self, material_name: str) -> Dict[str, float]:
        """Get material properties from knowledge graph"""
        material_lower = material_name.lower()
        
        for material_type, properties in self.knowledge_graph['material_properties'].items():
            if material_type in material_lower:
                return properties
        
        return {'strength': 0.5, 'corrosion_resistance': 0.5, 'recyclability': 0.5, 'density': 1.0}
    
    def _encode_material_properties(self, properties: Dict[str, float]) -> List[float]:
        """Encode material properties as feature vector"""
        return [
            properties.get('strength', 0.5),
            properties.get('corrosion_resistance', 0.5),
            properties.get('recyclability', 0.5),
            properties.get('density', 1.0) / 10.0,  # Normalize density
            properties.get('conductivity', 0.5),
            properties.get('toxicity', 0.5)
        ]
    
    def _are_materials_compatible(self, material1: Dict[str, Any], material2: Dict[str, Any]) -> bool:
        """Check if materials are compatible"""
        name1 = material1.get('material_name', '').lower()
        name2 = material2.get('material_name', '').lower()
        
        # Check knowledge graph compatibility
        for material_type, compatible_types in self.knowledge_graph['material_compatibility'].items():
            if material_type in name1:
                return any(comp_type in name2 for comp_type in compatible_types)
            elif material_type in name2:
                return any(comp_type in name1 for comp_type in compatible_types)
        
        return False
    
    def _get_company_index(self, company_id: str, companies: List[Dict[str, Any]]) -> Optional[int]:
        """Get company index by ID"""
        for i, company in enumerate(companies):
            if company.get('id') == company_id:
                return i
        return None

# Initialize the world-class GNN reasoning engine
gnn_reasoning_engine = WorldClassGNNReasoningEngine()
