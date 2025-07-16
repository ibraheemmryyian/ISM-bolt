import os
import json
import numpy as np
import requests
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
import aiohttp
from datetime import datetime, timedelta

# Materials Project API integration
try:
    from mp_api.client import MPRester
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("⚠️ Materials Project API not available, will use alternative sources")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedMaterialsBERTService:
    def __init__(self):
        self.app = Flask(__name__)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pca = PCA(n_components=100)
        self.materials_knowledge_base = {}
        self.symbiosis_patterns = {}
        self.embedding_cache = {}
        self.cache_lock = threading.Lock()
        
        # API configurations
        self.materials_project_api_key = os.getenv('MP_API_KEY', 'zSFjfpRg6m020aK84yOjM7oLIhjDNPjE')
        self.next_gen_materials_api_key = os.getenv('NEXT_GEN_MATERIALS_API_KEY')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY', 'sk-7ce79f30332d45d5b3acb8968b052132')
        
        # External API endpoints
        self.next_gen_materials_url = 'https://api.next-gen-materials.com/v1'
        self.deepseek_url = 'https://api.deepseek.com/v1/chat/completions'
        
        # Materials Project client
        self.mpr = None
        self._init_materials_project_client()
        
        # Cache configuration
        self.cache_timeout = 3600  # 1 hour
        self.cache_cleanup_interval = 300  # 5 minutes
        self.last_cache_cleanup = time.time()
        
        # Initialize dynamic materials analysis components
        self._initialize_dynamic_materials_knowledge_base()
        self._initialize_symbiosis_patterns()
        self._setup_routes()
        
        # Start background cache cleanup
        self._start_cache_cleanup()
        
    def _init_materials_project_client(self):
        """Initialize Materials Project API client with fallback"""
        if not MP_AVAILABLE:
            logger.warning("Materials Project API not available, using alternative sources")
            return
            
        try:
            self.mpr = MPRester(self.materials_project_api_key)
            # Test the connection
            self.mpr.summary.get_data_by_id("mp-149")
            logger.info("✅ Materials Project API client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Materials Project API client: {e}")
            self.mpr = None
        
    def _initialize_dynamic_materials_knowledge_base(self):
        """Initialize materials knowledge base dynamically from external sources"""
        logger.info("Initializing dynamic materials knowledge base from external sources...")
        
        # Start with empty knowledge base - will be populated dynamically
        self.materials_knowledge_base = {
            "metals": {},
            "polymers": {},
            "ceramics": {},
            "composites": {},
            "biomaterials": {},
            "nanomaterials": {},
            "smart_materials": {}
        }
        
        # Load initial data from external sources
        self._load_materials_from_external_sources()
        
        logger.info(f"Dynamic knowledge base initialized with {sum(len(cat) for cat in self.materials_knowledge_base.values())} materials")
        
    def _load_materials_from_external_sources(self):
        """Load materials data from multiple external sources"""
        try:
            # Load from Materials Project API
            if self.mpr:
                self._load_from_materials_project()
            
            # Load from Next Gen Materials API
            if self.next_gen_materials_api_key:
                self._load_from_next_gen_materials()
            
            # Load from scientific databases
            self._load_from_scientific_databases()
            
            # Load from market intelligence
            self._load_from_market_intelligence()
            
        except Exception as e:
            logger.error(f"Error loading materials from external sources: {e}")
            # Continue with empty knowledge base - will be populated on-demand
        
    def _load_from_materials_project(self):
        """Load materials data from Materials Project API"""
        try:
            logger.info("Loading materials from Materials Project API...")
            
            # Get common materials from Materials Project
            common_materials = [
                "mp-149",  # Silicon
                "mp-30",   # Aluminum
                "mp-13",   # Iron
                "mp-126",  # Copper
                "mp-568",  # Titanium
                "mp-48",   # Nickel
                "mp-33",   # Zinc
                "mp-23",   # Magnesium
                "mp-81",   # Lead
                "mp-32",   # Tin
            ]
            
            for mp_id in common_materials:
                try:
                    material_data = self.mpr.summary.get_data_by_id(mp_id)
                    if material_data:
                        material_info = self._process_materials_project_data(material_data)
                        category = self._classify_material_category(material_info)
                        if category:
                            self.materials_knowledge_base[category][material_info['name']] = material_info
                except Exception as e:
                    logger.warning(f"Failed to load material {mp_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error loading from Materials Project API: {e}")
    
    def _process_materials_project_data(self, material_data) -> Dict[str, Any]:
        """Process Materials Project data into standardized format"""
        try:
            return {
                "name": material_data.get('formula_pretty', 'Unknown'),
                "properties": self._extract_properties_from_mp(material_data),
                "applications": self._predict_applications_from_mp(material_data),
                "sustainability_score": self._calculate_sustainability_from_mp(material_data),
                "carbon_footprint": self._estimate_carbon_footprint(material_data),
                "recycling_rate": self._estimate_recycling_rate(material_data),
                "embodied_energy": self._estimate_embodied_energy(material_data),
                "chemical_formula": material_data.get('formula_pretty'),
                "crystal_structure": material_data.get('symmetry', {}).get('crystal_system'),
                "density": material_data.get('density', 0),
                "band_gap": material_data.get('band_gap', 0),
                "source": "materials_project"
            }
        except Exception as e:
            logger.error(f"Error processing Materials Project data: {e}")
            return {}
    
    def _extract_properties_from_mp(self, material_data) -> List[str]:
        """Extract material properties from Materials Project data"""
        properties = []
        
        # Mechanical properties
        if material_data.get('elasticity'):
            properties.extend(['elastic', 'deformable'])
        
        # Electronic properties
        if material_data.get('band_gap', 0) > 0:
            properties.append('semiconductor')
        elif material_data.get('band_gap', 0) == 0:
            properties.append('metallic')
        else:
            properties.append('insulator')
        
        # Thermal properties
        if material_data.get('thermal_conductivity'):
            properties.append('thermal_conductive')
        
        # Magnetic properties
        if material_data.get('magnetic_ordering'):
            properties.append('magnetic')
        
        return properties
    
    def _predict_applications_from_mp(self, material_data) -> List[str]:
        """Predict applications based on Materials Project properties"""
        applications = []
        
        # Semiconductor applications
        if material_data.get('band_gap', 0) > 0:
            applications.extend(['electronics', 'solar_cells', 'sensors'])
        
        # Metallic applications
        if material_data.get('band_gap', 0) == 0:
            applications.extend(['electrical_wiring', 'structural', 'thermal_management'])
        
        # Magnetic applications
        if material_data.get('magnetic_ordering'):
            applications.extend(['data_storage', 'motors', 'transformers'])
        
        return applications
    
    def _calculate_sustainability_from_mp(self, material_data) -> float:
        """Calculate sustainability score from Materials Project data"""
        score = 0.5  # Base score
        
        # Abundance factor
        if material_data.get('abundance'):
            score += 0.2
        
        # Toxicity factor (simplified)
        if not material_data.get('toxicity'):
            score += 0.1
        
        # Recyclability factor
        score += 0.2  # Assume most materials are recyclable
        
        return min(1.0, score)
    
    def _estimate_carbon_footprint(self, material_data) -> float:
        """Estimate carbon footprint based on material properties"""
        # Simplified estimation based on material type
        base_footprint = 2.0
        
        # Adjust based on density and processing requirements
        if material_data.get('density'):
            base_footprint *= (material_data['density'] / 1000)  # Normalize to kg/m³
        
        return base_footprint
    
    def _estimate_recycling_rate(self, material_data) -> float:
        """Estimate recycling rate based on material properties"""
        # Simplified estimation
        base_rate = 0.5
        
        # Metals typically have higher recycling rates
        if material_data.get('band_gap', 0) == 0:  # Metallic
            base_rate = 0.8
        
        return base_rate
    
    def _estimate_embodied_energy(self, material_data) -> float:
        """Estimate embodied energy based on material properties"""
        # Simplified estimation in MJ/kg
        base_energy = 50.0
        
        # Adjust based on density
        if material_data.get('density'):
            base_energy *= (material_data['density'] / 1000)
        
        return base_energy
    
    def _load_from_next_gen_materials(self):
        """Load materials data from Next Gen Materials API"""
        try:
            logger.info("Loading materials from Next Gen Materials API...")
            
            # This would be the actual API call to Next Gen Materials
            # For now, we'll simulate the response structure
            response = self._make_api_call(
                'GET',
                f"{self.next_gen_materials_url}/materials/catalog",
                headers={'Authorization': f'Bearer {self.next_gen_materials_api_key}'}
            )
            
            if response and response.get('materials'):
                for material in response['materials']:
                    material_info = self._process_next_gen_materials_data(material)
                    category = self._classify_material_category(material_info)
                    if category:
                        self.materials_knowledge_base[category][material_info['name']] = material_info
                        
        except Exception as e:
            logger.error(f"Error loading from Next Gen Materials API: {e}")
    
    def _process_next_gen_materials_data(self, material_data) -> Dict[str, Any]:
        """Process Next Gen Materials API data"""
        return {
            "name": material_data.get('name', 'Unknown'),
            "properties": material_data.get('properties', []),
            "applications": material_data.get('applications', []),
            "sustainability_score": material_data.get('sustainability_score', 0.5),
            "carbon_footprint": material_data.get('carbon_footprint', 2.0),
            "recycling_rate": material_data.get('recycling_rate', 0.5),
            "embodied_energy": material_data.get('embodied_energy', 50.0),
            "source": "next_gen_materials"
        }
    
    def _load_from_scientific_databases(self):
        """Load materials data from scientific databases"""
        try:
            logger.info("Loading materials from scientific databases...")
            
            # This would integrate with scientific databases like:
            # - PubChem
            # - ChemSpider
            # - Materials Data Facility
            # - NIST Materials Database
            
            # For now, we'll use a simplified approach
            scientific_materials = self._fetch_scientific_materials()
            
            for material in scientific_materials:
                material_info = self._process_scientific_data(material)
                category = self._classify_material_category(material_info)
                if category:
                    self.materials_knowledge_base[category][material_info['name']] = material_info
                    
        except Exception as e:
            logger.error(f"Error loading from scientific databases: {e}")
    
    def _fetch_scientific_materials(self) -> List[Dict[str, Any]]:
        """Fetch materials from scientific databases"""
        # This would be actual API calls to scientific databases
        # For now, return empty list - will be populated on-demand
        return []
    
    def _process_scientific_data(self, material_data) -> Dict[str, Any]:
        """Process scientific database data"""
        return {
            "name": material_data.get('name', 'Unknown'),
            "properties": material_data.get('properties', []),
            "applications": material_data.get('applications', []),
            "sustainability_score": material_data.get('sustainability_score', 0.5),
            "carbon_footprint": material_data.get('carbon_footprint', 2.0),
            "recycling_rate": material_data.get('recycling_rate', 0.5),
            "embodied_energy": material_data.get('embodied_energy', 50.0),
            "source": "scientific_database"
        }
    
    def _load_from_market_intelligence(self):
        """Load materials data from market intelligence sources"""
        try:
            logger.info("Loading materials from market intelligence...")
            
            # This would integrate with market intelligence APIs like:
            # - Bloomberg Terminal
            # - Reuters
            # - Market research databases
            
            # For now, we'll use a simplified approach
            market_materials = self._fetch_market_intelligence()
            
            for material in market_materials:
                material_info = self._process_market_data(material)
                category = self._classify_material_category(material_info)
                if category:
                    self.materials_knowledge_base[category][material_info['name']] = material_info
                    
        except Exception as e:
            logger.error(f"Error loading from market intelligence: {e}")
    
    def _fetch_market_intelligence(self) -> List[Dict[str, Any]]:
        """Fetch materials from market intelligence sources"""
        # This would be actual API calls to market intelligence sources
        # For now, return empty list - will be populated on-demand
        return []
    
    def _process_market_data(self, material_data) -> Dict[str, Any]:
        """Process market intelligence data"""
        return {
            "name": material_data.get('name', 'Unknown'),
            "properties": material_data.get('properties', []),
            "applications": material_data.get('applications', []),
            "sustainability_score": material_data.get('sustainability_score', 0.5),
            "carbon_footprint": material_data.get('carbon_footprint', 2.0),
            "recycling_rate": material_data.get('recycling_rate', 0.5),
            "embodied_energy": material_data.get('embodied_energy', 50.0),
            "source": "market_intelligence"
        }
    
    def _classify_material_category(self, material_info: Dict[str, Any]) -> Optional[str]:
        """Classify material into category based on properties and applications"""
        if not material_info or not material_info.get('properties'):
            return None
        
        properties = [p.lower() for p in material_info['properties']]
        name = material_info.get('name', '').lower()
        
        # Classification logic based on properties and name
        if any(p in properties for p in ['metallic', 'conductive', 'magnetic']) or 'metal' in name:
            return "metals"
        elif any(p in properties for p in ['polymer', 'plastic', 'flexible']) or 'polymer' in name:
            return "polymers"
        elif any(p in properties for p in ['ceramic', 'brittle', 'refractory']) or 'ceramic' in name:
            return "ceramics"
        elif any(p in properties for p in ['composite', 'fiber', 'reinforced']) or 'composite' in name:
            return "composites"
        elif any(p in properties for p in ['bio', 'organic', 'natural']) or 'bio' in name:
            return "biomaterials"
        elif any(p in properties for p in ['nano', 'quantum']) or 'nano' in name:
            return "nanomaterials"
        elif any(p in properties for p in ['smart', 'responsive', 'adaptive']) or 'smart' in name:
            return "smart_materials"
        
        return "metals"  # Default classification
    
    def _make_api_call(self, method: str, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make API call with error handling and retries"""
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API call failed to {url}: {e}")
            return None
        
    def _initialize_symbiosis_patterns(self):
        """Initialize industrial symbiosis patterns dynamically"""
        logger.info("Initializing dynamic industrial symbiosis patterns...")
        
        # Load symbiosis patterns from external sources
        self.symbiosis_patterns = self._load_symbiosis_patterns_from_sources()
        
        logger.info(f"Loaded {len(self.symbiosis_patterns)} symbiosis pattern categories")
        
    def _load_symbiosis_patterns_from_sources(self) -> Dict[str, Any]:
        """Load symbiosis patterns from external sources"""
        patterns = {
            "waste_to_resource": {},
            "energy_cascading": {},
            "water_recycling": {},
            "byproduct_exchange": {},
            "infrastructure_sharing": {},
            "knowledge_sharing": {}
        }
        
        try:
            # Load from scientific literature databases
            patterns.update(self._load_patterns_from_literature())
            
            # Load from industrial case studies
            patterns.update(self._load_patterns_from_case_studies())
            
            # Load from expert knowledge bases
            patterns.update(self._load_patterns_from_expert_knowledge())
            
        except Exception as e:
            logger.error(f"Error loading symbiosis patterns: {e}")
        
        return patterns
    
    def _load_patterns_from_literature(self) -> Dict[str, Any]:
        """Load symbiosis patterns from scientific literature"""
        # This would integrate with scientific literature databases
        # For now, return basic patterns
        return {
            "waste_to_resource": {
                "steel_slag": ["cement_production", "road_construction", "agricultural_soil"],
                "fly_ash": ["concrete_production", "brick_manufacturing", "soil_stabilization"],
                "plastic_waste": ["recycled_pellets", "energy_recovery", "3d_printing_filament"],
                "organic_waste": ["biogas_production", "compost_manufacturing", "animal_feed"],
                "glass_waste": ["new_glass_products", "construction_aggregate", "abrasive_materials"]
            }
        }
    
    def _load_patterns_from_case_studies(self) -> Dict[str, Any]:
        """Load symbiosis patterns from industrial case studies"""
        # This would integrate with case study databases
        return {
            "energy_cascading": {
                "high_temperature": ["steel_production", "cement_manufacturing", "glass_melting"],
                "medium_temperature": ["food_processing", "chemical_manufacturing", "textile_dyeing"],
                "low_temperature": ["greenhouse_heating", "district_heating", "water_heating"]
            }
        }
    
    def _load_patterns_from_expert_knowledge(self) -> Dict[str, Any]:
        """Load symbiosis patterns from expert knowledge bases"""
        # This would integrate with expert knowledge systems
        return {
            "water_recycling": {
                "industrial_water": ["cooling_systems", "cleaning_processes", "irrigation"],
                "treated_wastewater": ["landscape_irrigation", "industrial_cooling", "toilet_flushing"]
            },
            "byproduct_exchange": {
                "hydrogen": ["fuel_cells", "chemical_synthesis", "steel_production"],
                "carbon_dioxide": ["greenhouse_growing", "carbonation_processes", "enhanced_oil_recovery"],
                "steam": ["heating_systems", "turbine_generation", "drying_processes"]
            }
        }
        
    def _start_cache_cleanup(self):
        """Start background cache cleanup process"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cache_cleanup_interval)
                    self._cleanup_expired_cache()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        with self.cache_lock:
            for key, (data, timestamp) in self.embedding_cache.items():
                if current_time - timestamp > self.cache_timeout:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.embedding_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_material_data_dynamically(self, material_name: str) -> Dict[str, Any]:
        """Get material data dynamically from external sources"""
        cache_key = f"material_{material_name.lower()}"
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.embedding_cache:
                data, timestamp = self.embedding_cache[cache_key]
                if time.time() - timestamp < self.cache_timeout:
                    return data
        
        # Fetch from external sources
        material_data = self._fetch_material_from_sources(material_name)
        
        # Cache the result
        with self.cache_lock:
            self.embedding_cache[cache_key] = (material_data, time.time())
        
        return material_data
    
    def _fetch_material_from_sources(self, material_name: str) -> Dict[str, Any]:
        """Fetch material data from multiple external sources"""
        material_data = {
            "name": material_name,
            "properties": [],
            "applications": [],
            "sustainability_score": 0.5,
            "carbon_footprint": 2.0,
            "recycling_rate": 0.5,
            "embodied_energy": 50.0,
            "sources": []
        }
        
        # Try Materials Project API
        if self.mpr:
            mp_data = self._fetch_from_materials_project(material_name)
            if mp_data:
                material_data.update(mp_data)
                material_data["sources"].append("materials_project")
        
        # Try Next Gen Materials API
        if self.next_gen_materials_api_key:
            ng_data = self._fetch_from_next_gen_materials(material_name)
            if ng_data:
                material_data.update(ng_data)
                material_data["sources"].append("next_gen_materials")
        
        # Try AI-powered analysis
        ai_data = self._analyze_with_ai(material_name)
        if ai_data:
            material_data.update(ai_data)
            material_data["sources"].append("ai_analysis")
        
        return material_data
    
    def _fetch_from_materials_project(self, material_name: str) -> Optional[Dict[str, Any]]:
        """Fetch material data from Materials Project API"""
        try:
            # Search for material in Materials Project
            search_results = self.mpr.summary.search(material_name, fields=["formula_pretty"])
            
            if search_results:
                # Get detailed data for the first result
                material_id = search_results[0].material_id
                material_data = self.mpr.summary.get_data_by_id(material_id)
                
                if material_data:
                    return self._process_materials_project_data(material_data)
            
        except Exception as e:
            logger.error(f"Error fetching from Materials Project: {e}")
        
        return None
    
    def _fetch_from_next_gen_materials(self, material_name: str) -> Optional[Dict[str, Any]]:
        """Fetch material data from Next Gen Materials API"""
        try:
            response = self._make_api_call(
                'GET',
                f"{self.next_gen_materials_url}/materials/search",
                params={'query': material_name},
                headers={'Authorization': f'Bearer {self.next_gen_materials_api_key}'}
            )
            
            if response and response.get('materials'):
                return self._process_next_gen_materials_data(response['materials'][0])
            
        except Exception as e:
            logger.error(f"Error fetching from Next Gen Materials: {e}")
        
        return None
    
    def _analyze_with_ai(self, material_name: str) -> Optional[Dict[str, Any]]:
        """Analyze material using AI models"""
        try:
            # Use DeepSeek API for material analysis
            prompt = f"""
            Analyze the material "{material_name}" and provide:
            1. Key properties and characteristics
            2. Common applications
            3. Sustainability metrics
            4. Environmental impact
            5. Circular economy opportunities
            
            Provide the response as a JSON object with the following structure:
            {{
                "properties": ["property1", "property2"],
                "applications": ["application1", "application2"],
                "sustainability_score": 0.0-1.0,
                "carbon_footprint": float,
                "recycling_rate": 0.0-1.0,
                "embodied_energy": float,
                "environmental_impact": "low/medium/high",
                "circular_opportunities": ["opportunity1", "opportunity2"]
            }}
            """
            
            response = self._make_api_call(
                'POST',
                self.deepseek_url,
                json={
                    'model': 'deepseek-reasoner',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.1
                },
                headers={'Authorization': f'Bearer {self.deepseek_api_key}'}
            )
            
            if response and response.get('choices'):
                content = response['choices'][0]['message']['content']
                try:
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        ai_data = json.loads(json_match.group())
                        return ai_data
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse AI response as JSON: {content}")
            
        except Exception as e:
            logger.error(f"Error analyzing with AI: {e}")
        
        return None
    
    def _setup_routes(self):
        """Setup Flask routes for the service"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'materials_loaded': sum(len(cat) for cat in self.materials_knowledge_base.values()),
                'cache_size': len(self.embedding_cache),
                'materials_project_available': self.mpr is not None,
                'next_gen_materials_available': bool(self.next_gen_materials_api_key)
            })
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_material():
            try:
                data = request.get_json()
                material_name = data.get('material', '')
                context = data.get('context', {})
                
                if not material_name:
                    return jsonify({'error': 'Material name is required'}), 400
                
                # Get dynamic material analysis
                analysis = self.get_material_data_dynamically(material_name)
                
                # Add symbiosis opportunities
                analysis['symbiosis_opportunities'] = self._find_symbiosis_opportunities(material_name, analysis)
                
                # Add recommendations
                analysis['recommendations'] = self._generate_recommendations(analysis)
                
                return jsonify({
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error in material analysis: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/materials/search', methods=['POST'])
        def search_materials():
            try:
                data = request.get_json()
                query = data.get('query', '')
                
                if not query:
                    return jsonify({'error': 'Search query is required'}), 400
                
                # Search across all categories
                results = []
                for category, materials in self.materials_knowledge_base.items():
                    for name, info in materials.items():
                        if query.lower() in name.lower() or any(query.lower() in prop.lower() for prop in info.get('properties', [])):
                            results.append({
                                'name': name,
                                'category': category,
                                'info': info
                            })
                
                # Also search dynamically
                dynamic_result = self.get_material_data_dynamically(query)
                if dynamic_result and dynamic_result.get('name'):
                    results.append({
                        'name': dynamic_result['name'],
                        'category': 'dynamic',
                        'info': dynamic_result
                    })
                
                return jsonify({
                    'results': results,
                    'count': len(results)
                })
                
            except Exception as e:
                logger.error(f"Error in material search: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _find_symbiosis_opportunities(self, material_name: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find symbiosis opportunities for the material"""
        opportunities = []
        
        # Search through symbiosis patterns
        for pattern_type, patterns in self.symbiosis_patterns.items():
            for pattern_name, applications in patterns.items():
                if material_name.lower() in pattern_name.lower():
                    opportunities.append({
                        'pattern_type': pattern_type,
                        'pattern_name': pattern_name,
                        'applications': applications,
                        'confidence': 0.8
                    })
        
        return opportunities
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on material analysis"""
        recommendations = []
        
        # Sustainability recommendations
        if analysis.get('sustainability_score', 0) < 0.6:
            recommendations.append("Consider alternative materials with higher sustainability scores")
        
        # Recycling recommendations
        if analysis.get('recycling_rate', 0) < 0.5:
            recommendations.append("Explore recycling partnerships to improve material recovery")
        
        # Circular economy recommendations
        if analysis.get('circular_opportunities'):
            recommendations.extend([f"Explore {opp} opportunities" for opp in analysis['circular_opportunities'][:3]])
        
        return recommendations
    
    def run(self, host='0.0.0.0', port=8001, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting Advanced MaterialsBERT Service on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    service = AdvancedMaterialsBERTService()
    service.run() 