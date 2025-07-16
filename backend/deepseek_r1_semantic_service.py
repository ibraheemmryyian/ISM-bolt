#!/usr/bin/env python3
"""
DeepSeek R1 Semantic Analysis Service
Advanced semantic analysis using DeepSeek R1 for industrial symbiosis
"""

import os
import json
import logging
import requests
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekR1SemanticService:
    """Advanced semantic analysis service using DeepSeek R1"""
    
    def __init__(self):
        self.api_key = 'sk-7ce79f30332d45d5b3acb8968b052132'
        self.base_url = 'https://api.deepseek.com/v1/chat/completions'
        self.model = 'deepseek-r1'
        
    def analyze_semantic_similarity(self, text1: str, text2: str, context: str = "") -> Dict[str, Any]:
        """Analyze semantic similarity using DeepSeek R1's advanced reasoning"""
        try:
            prompt = f"""You are DeepSeek R1, an expert in semantic analysis and industrial symbiosis. Analyze the semantic similarity between two texts using advanced reasoning.

TEXT 1: "{text1}"
TEXT 2: "{text2}"
CONTEXT: "{context}"

TASK: Analyze the semantic similarity between these texts considering:
1. Conceptual overlap and shared meaning
2. Industry relevance and business context
3. Material compatibility and technical alignment
4. Business synergy potential and partnership opportunities
5. Operational compatibility and logistical considerations

Provide your analysis as JSON with this exact structure:
{{
    "similarity_score": 0.0-1.0,
    "reasoning": "detailed explanation of similarity analysis",
    "key_overlaps": ["list of key overlapping concepts"],
    "synergy_potential": "assessment of business synergy potential",
    "technical_alignment": "assessment of technical compatibility",
    "confidence": 0.0-1.0,
    "recommendations": ["specific recommendations for collaboration"]
}}"""

            response = self._call_api({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are DeepSeek R1, an expert in semantic analysis for industrial symbiosis. Use your advanced reasoning to provide precise similarity analysis."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": 1500
            })
            
            if response and 'choices' in response:
                result = response['choices'][0]['message']['content']
                analysis = json.loads(result)
                return {
                    'success': True,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'No response from DeepSeek R1',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"DeepSeek R1 semantic analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_company_compatibility(self, company1: Dict[str, Any], company2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze company compatibility using DeepSeek R1"""
        try:
            company1_text = f"Company: {company1.get('name', 'Unknown')}, Industry: {company1.get('industry', 'Unknown')}, Products: {company1.get('products', 'Unknown')}, Materials: {company1.get('main_materials', 'Unknown')}"
            company2_text = f"Company: {company2.get('name', 'Unknown')}, Industry: {company2.get('industry', 'Unknown')}, Products: {company2.get('products', 'Unknown')}, Materials: {company2.get('main_materials', 'Unknown')}"
            
            prompt = f"""You are DeepSeek R1, an expert in industrial symbiosis and business compatibility analysis. Analyze the compatibility between two companies for potential industrial symbiosis partnerships.

COMPANY 1: {company1_text}
COMPANY 2: {company2_text}

TASK: Analyze the compatibility and synergy potential between these companies considering:
1. Material flow compatibility (waste-to-resource matching)
2. Industry synergy potential
3. Geographic and logistical considerations
4. Business model compatibility
5. Risk factors and challenges
6. Implementation feasibility

Provide your analysis as JSON with this exact structure:
{{
    "compatibility_score": 0.0-1.0,
    "synergy_opportunities": ["list of specific synergy opportunities"],
    "material_matches": ["specific material matches between companies"],
    "business_benefits": ["quantified business benefits"],
    "implementation_steps": ["step-by-step implementation plan"],
    "risk_assessment": ["potential risks and mitigation strategies"],
    "confidence": 0.0-1.0,
    "recommendations": ["specific recommendations for partnership"]
}}"""

            response = self._call_api({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are DeepSeek R1, an expert in industrial symbiosis analysis. Use your advanced reasoning to provide comprehensive compatibility analysis."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
                "max_tokens": 2000
            })
            
            if response and 'choices' in response:
                result = response['choices'][0]['message']['content']
                analysis = json.loads(result)
                return {
                    'success': True,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'No response from DeepSeek R1',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"DeepSeek R1 company compatibility analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _call_api(self, prompt_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API call to DeepSeek R1"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=prompt_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"DeepSeek R1 API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"DeepSeek R1 API call failed: {e}")
            return None

# Flask app
app = Flask(__name__)
CORS(app)

# Initialize service
semantic_service = DeepSeekR1SemanticService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'deepseek_r1_semantic',
        'model': 'deepseek-r1',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze/similarity', methods=['POST'])
def analyze_similarity():
    """Analyze semantic similarity between two texts"""
    try:
        data = request.get_json()
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        context = data.get('context', '')
        
        if not text1 or not text2:
            return jsonify({'error': 'text1 and text2 are required'}), 400
        
        result = semantic_service.analyze_semantic_similarity(text1, text2, context)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in similarity analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/compatibility', methods=['POST'])
def analyze_compatibility():
    """Analyze company compatibility"""
    try:
        data = request.get_json()
        company1 = data.get('company1', {})
        company2 = data.get('company2', {})
        
        if not company1 or not company2:
            return jsonify({'error': 'company1 and company2 are required'}), 400
        
        result = semantic_service.analyze_company_compatibility(company1, company2)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in compatibility analysis: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
