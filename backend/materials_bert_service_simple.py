#!/usr/bin/env python3
"""
Simplified MaterialsBERT Service
Provides materials analysis using lightweight embeddings and knowledge base
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import hashlib
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class LightweightMaterialsBERT:
    """Lightweight materials analysis using knowledge base and simple embeddings"""
    
    def __init__(self):
        self.materials_knowledge = {
            # Material properties and characteristics
            'properties': {
                'strength': ['tensile', 'compressive', 'flexural', 'impact', 'shear'],
                'durability': ['corrosion', 'weathering', 'fatigue', 'wear', 'aging'],
                'thermal': ['conductivity', 'expansion', 'resistance', 'stability'],
                'electrical': ['conductivity', 'resistivity', 'dielectric', 'insulation'],
                'chemical': ['reactivity', 'stability', 'compatibility', 'resistance'],
                'mechanical': ['hardness', 'toughness', 'elasticity', 'plasticity'],
                'environmental': ['biodegradable', 'recyclable', 'sustainable', 'eco-friendly']
            },
            
            # Material categories
            'categories': {
                'metals': ['steel', 'aluminum', 'copper', 'titanium', 'nickel', 'zinc', 'magnesium'],
                'polymers': ['polyethylene', 'polypropylene', 'pvc', 'nylon', 'polyester', 'epoxy'],
                'ceramics': ['alumina', 'zirconia', 'silica', 'titanate', 'ferrite'],
                'composites': ['carbon_fiber', 'glass_fiber', 'aramid', 'hybrid'],
                'biomaterials': ['cellulose', 'chitin', 'collagen', 'silk', 'bamboo'],
                'nanomaterials': ['graphene', 'nanotubes', 'nanoparticles', 'quantum_dots']
            },
            
            # Material applications
            'applications': {
                'construction': ['concrete', 'steel', 'wood', 'glass', 'insulation'],
                'automotive': ['lightweight', 'crash_resistant', 'heat_resistant', 'corrosion_resistant'],
                'aerospace': ['high_strength', 'low_weight', 'temperature_resistant', 'fatigue_resistant'],
                'electronics': ['conductive', 'insulating', 'semiconductor', 'magnetic'],
                'medical': ['biocompatible', 'sterile', 'bioactive', 'biodegradable'],
                'energy': ['solar', 'battery', 'fuel_cell', 'thermal_storage']
            }
        }
        
        # Pre-computed embeddings for common materials
        self.material_embeddings = self._create_material_embeddings()
        
    def _create_material_embeddings(self):
        """Create simple embeddings for materials using hash-based approach"""
        embeddings = {}
        
        # Create embeddings for all materials in knowledge base
        all_materials = set()
        for category, materials in self.materials_knowledge['categories'].items():
            all_materials.update(materials)
        
        for material in all_materials:
            # Create a simple embedding using hash
            hash_obj = hashlib.md5(material.encode())
            embedding = np.frombuffer(hash_obj.digest(), dtype=np.float32)
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            embeddings[material] = embedding
            
        return embeddings
    
    def _text_to_embedding(self, text):
        """Convert text to embedding using simple hash-based approach"""
        # Clean and normalize text
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        # Create embedding from words
        if not words:
            return np.zeros(16, dtype=np.float32)
        
        # Combine word embeddings
        embeddings = []
        for word in words:
            hash_obj = hashlib.md5(word.encode())
            embedding = np.frombuffer(hash_obj.digest(), dtype=np.float32)
            embeddings.append(embedding)
        
        # Average the embeddings
        combined = np.mean(embeddings, axis=0)
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined
    
    def analyze_material(self, material_name, description=""):
        """Analyze material properties and characteristics"""
        try:
            # Create embedding for input
            input_embedding = self._text_to_embedding(f"{material_name} {description}")
            
            # Find similar materials
            similarities = {}
            for material, embedding in self.material_embeddings.items():
                similarity = np.dot(input_embedding, embedding)
                similarities[material] = float(similarity)
            
            # Sort by similarity
            sorted_materials = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Analyze properties
            analysis = {
                'material_name': material_name,
                'description': description,
                'similar_materials': sorted_materials[:5],
                'properties': self._analyze_properties(material_name, description),
                'applications': self._analyze_applications(material_name, description),
                'sustainability_score': self._calculate_sustainability_score(material_name, description),
                'performance_metrics': self._generate_performance_metrics(material_name, description)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing material: {e}")
            return {
                'error': str(e),
                'material_name': material_name,
                'description': description
            }
    
    def _analyze_properties(self, material_name, description):
        """Analyze material properties based on keywords"""
        text = f"{material_name} {description}".lower()
        properties = {}
        
        for prop_type, keywords in self.materials_knowledge['properties'].items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            if score > 0:
                properties[prop_type] = min(score / len(keywords), 1.0)
        
        return properties
    
    def _analyze_applications(self, material_name, description):
        """Analyze potential applications"""
        text = f"{material_name} {description}".lower()
        applications = {}
        
        for app_type, keywords in self.materials_knowledge['applications'].items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            if score > 0:
                applications[app_type] = min(score / len(keywords), 1.0)
        
        return applications
    
    def _calculate_sustainability_score(self, material_name, description):
        """Calculate sustainability score"""
        text = f"{material_name} {description}".lower()
        
        # Positive sustainability indicators
        positive_terms = ['recyclable', 'biodegradable', 'sustainable', 'eco-friendly', 'green', 'natural']
        # Negative sustainability indicators
        negative_terms = ['toxic', 'hazardous', 'polluting', 'non-recyclable', 'synthetic']
        
        positive_score = sum(1 for term in positive_terms if term in text)
        negative_score = sum(1 for term in negative_terms if term in text)
        
        # Calculate final score (0-100)
        base_score = 50
        score = base_score + (positive_score * 10) - (negative_score * 15)
        return max(0, min(100, score))
    
    def _generate_performance_metrics(self, material_name, description):
        """Generate performance metrics based on material characteristics"""
        text = f"{material_name} {description}".lower()
        
        metrics = {
            'strength_rating': 50,
            'durability_rating': 50,
            'cost_efficiency': 50,
            'environmental_impact': 50,
            'processing_ease': 50
        }
        
        # Adjust based on keywords
        if any(word in text for word in ['strong', 'tensile', 'compressive']):
            metrics['strength_rating'] += 20
        if any(word in text for word in ['durable', 'corrosion', 'weather']):
            metrics['durability_rating'] += 20
        if any(word in text for word in ['cheap', 'inexpensive', 'cost-effective']):
            metrics['cost_efficiency'] += 20
        if any(word in text for word in ['eco-friendly', 'sustainable', 'green']):
            metrics['environmental_impact'] += 20
        if any(word in text for word in ['easy', 'simple', 'machinable']):
            metrics['processing_ease'] += 20
        
        # Normalize to 0-100
        for key in metrics:
            metrics[key] = max(0, min(100, metrics[key]))
        
        return metrics

# Initialize the service
materials_bert = LightweightMaterialsBERT()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'MaterialsBERT Simple',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze_material():
    """Analyze material properties and characteristics"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        material_name = data.get('material_name', '')
        description = data.get('description', '')
        
        if not material_name:
            return jsonify({'error': 'Material name is required'}), 400
        
        logger.info(f"Analyzing material: {material_name}")
        
        # Perform analysis
        analysis = materials_bert.analyze_material(material_name, description)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/materials', methods=['GET'])
def list_materials():
    """List available materials in knowledge base"""
    try:
        materials = list(materials_bert.material_embeddings.keys())
        return jsonify({
            'materials': materials,
            'count': len(materials)
        })
    except Exception as e:
        logger.error(f"Error listing materials: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/properties', methods=['GET'])
def get_properties():
    """Get available material properties"""
    try:
        return jsonify({
            'properties': materials_bert.materials_knowledge['properties'],
            'categories': materials_bert.materials_knowledge['categories'],
            'applications': materials_bert.materials_knowledge['applications']
        })
    except Exception as e:
        logger.error(f"Error getting properties: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting MaterialsBERT Simple Service...")
    logger.info("Service will be available at http://localhost:5002")
    
    try:
        app.run(host='0.0.0.0', port=5002, debug=False)
    except Exception as e:
        logger.error(f"Failed to start service: {e}") 