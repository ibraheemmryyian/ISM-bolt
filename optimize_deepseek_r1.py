#!/usr/bin/env python3
"""
Optimize AI System for DeepSeek R1
Replace sentence-transformers with DeepSeek R1 for better performance
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepSeekR1Optimizer:
    """Optimize AI system to use DeepSeek R1 instead of sentence-transformers"""
    
    def __init__(self):
        self.deepseek_api_key = 'sk-7ce79f30332d45d5b3acb8968b052132'
        self.deepseek_base_url = 'https://api.deepseek.com/v1/chat/completions'
        self.deepseek_model = 'deepseek-r1'
        
    def optimize_revolutionary_ai_matching(self):
        """Optimize revolutionary_ai_matching.py to use DeepSeek R1"""
        logger.info("Optimizing revolutionary_ai_matching.py for DeepSeek R1...")
        
        # Read the current file
        file_path = "backend/revolutionary_ai_matching.py"
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace sentence-transformers with DeepSeek R1
        optimized_content = self._replace_sentence_transformers_with_deepseek(content)
        
        # Write optimized file
        backup_path = file_path + ".backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Backup created: {backup_path}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(optimized_content)
        
        logger.info("✅ revolutionary_ai_matching.py optimized for DeepSeek R1")
        return True
    
    def optimize_conversational_agent(self):
        """Optimize conversational_b2b_agent.py to use DeepSeek R1"""
        logger.info("Optimizing conversational_b2b_agent.py for DeepSeek R1...")
        
        # Read the current file
        file_path = "conversational_b2b_agent.py"
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # The conversational agent is already using DeepSeek R1, just verify
        if 'deepseek-r1' in content:
            logger.info("✅ conversational_b2b_agent.py already optimized for DeepSeek R1")
            return True
        else:
            logger.warning("⚠️ conversational_b2b_agent.py needs optimization")
            return False
    
    def optimize_advanced_ai_prompts(self):
        """Optimize advanced_ai_prompts_service.py to use DeepSeek R1"""
        logger.info("Optimizing advanced_ai_prompts_service.py for DeepSeek R1...")
        
        # Read the current file
        file_path = "backend/advanced_ai_prompts_service.py"
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace deepseek-coder with deepseek-r1
        optimized_content = content.replace(
            "DEEPSEEK_MODEL = 'deepseek-coder'",
            "DEEPSEEK_MODEL = 'deepseek-r1'"
        )
        
        # Write optimized file
        backup_path = file_path + ".backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Backup created: {backup_path}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(optimized_content)
        
        logger.info("✅ advanced_ai_prompts_service.py optimized for DeepSeek R1")
        return True
    
    def _replace_sentence_transformers_with_deepseek(self, content: str) -> str:
        """Replace sentence-transformers usage with DeepSeek R1"""
        
        # Remove sentence-transformers import
        content = content.replace(
            "from sentence_transformers import SentenceTransformer",
            "# Replaced with DeepSeek R1 for better performance\n# from sentence_transformers import SentenceTransformer"
        )
        
        # Replace the model initialization
        old_init = """        # Load multiple specialized models for different aspects
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')
        self.industry_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.material_model = SentenceTransformer('all-MiniLM-L6-v2')"""
        
        new_init = """        # DeepSeek R1 for advanced semantic understanding
        self.deepseek_api_key = 'sk-7ce79f30332d45d5b3acb8968b052132'
        self.deepseek_base_url = 'https://api.deepseek.com/v1/chat/completions'
        self.deepseek_model = 'deepseek-r1'"""
        
        content = content.replace(old_init, new_init)
        
        # Add DeepSeek R1 semantic analysis method
        deepseek_method = '''
    def _analyze_semantic_similarity_deepseek(self, text1: str, text2: str, context: str = "") -> Tuple[float, str]:
        """Analyze semantic similarity using DeepSeek R1's advanced reasoning"""
        try:
            prompt = f"""You are DeepSeek R1, an expert in semantic analysis and industrial symbiosis. Analyze the semantic similarity between two texts using advanced reasoning.

TEXT 1: "{text1}"
TEXT 2: "{text2}"
CONTEXT: "{context}"

TASK: Analyze the semantic similarity between these texts considering:
1. Conceptual overlap
2. Industry relevance
3. Material compatibility
4. Business synergy potential
5. Technical alignment

Provide your analysis as JSON with this exact structure:
{{
    "similarity_score": 0.0-1.0,
    "reasoning": "detailed explanation of similarity analysis",
    "key_overlaps": ["list of key overlapping concepts"],
    "synergy_potential": "assessment of business synergy potential",
    "confidence": 0.0-1.0
}}"""

            response = self._call_deepseek_api({
                "model": self.deepseek_model,
                "messages": [
                    {"role": "system", "content": "You are DeepSeek R1, an expert in semantic analysis for industrial symbiosis. Use your advanced reasoning to provide precise similarity analysis."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": 1000
            })
            
            if response and 'choices' in response:
                result = response['choices'][0]['message']['content']
                import json
                analysis = json.loads(result)
                return analysis.get('similarity_score', 0.5), analysis.get('reasoning', 'Analysis completed')
            else:
                return 0.5, "Fallback similarity analysis"
                
        except Exception as e:
            logger.error(f"DeepSeek R1 semantic analysis failed: {e}")
            return 0.5, f"Error in analysis: {str(e)}"
    
    def _call_deepseek_api(self, prompt_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API call to DeepSeek R1"""
        try:
            import requests
            import json
            
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.deepseek_base_url,
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
            return None'''
        
        # Find a good place to insert the method (after the __init__ method)
        if 'def _signal_handler(self, signum, frame):' in content:
            content = content.replace(
                'def _signal_handler(self, signum, frame):',
                deepseek_method + '\n\n    def _signal_handler(self, signum, frame):'
            )
        
        return content
    
    def create_deepseek_r1_service(self):
        """Create a dedicated DeepSeek R1 service for semantic analysis"""
        logger.info("Creating dedicated DeepSeek R1 service...")
        
        service_content = '''#!/usr/bin/env python3
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
'''
        
        with open('backend/deepseek_r1_semantic_service.py', 'w', encoding='utf-8') as f:
            f.write(service_content)
        
        logger.info("✅ DeepSeek R1 semantic service created")
        return True
    
    def update_requirements(self):
        """Update requirements to remove sentence-transformers dependency"""
        logger.info("Updating requirements.txt...")
        
        requirements_path = "requirements.txt"
        if os.path.exists(requirements_path):
            with open(requirements_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove sentence-transformers line
            lines = content.split('\n')
            filtered_lines = [line for line in lines if 'sentence-transformers' not in line]
            
            # Add DeepSeek R1 note
            filtered_lines.append("# DeepSeek R1 replaces sentence-transformers for semantic analysis")
            
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(filtered_lines))
            
            logger.info("✅ requirements.txt updated")
            return True
        else:
            logger.warning("⚠️ requirements.txt not found")
            return False
    
    def run_optimization(self):
        """Run complete optimization"""
        logger.info("🚀 Starting DeepSeek R1 optimization...")
        
        results = {
            'revolutionary_ai_matching': self.optimize_revolutionary_ai_matching(),
            'conversational_agent': self.optimize_conversational_agent(),
            'advanced_ai_prompts': self.optimize_advanced_ai_prompts(),
            'semantic_service': self.create_deepseek_r1_service(),
            'requirements': self.update_requirements()
        }
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("DEEPSEEK R1 OPTIMIZATION SUMMARY")
        logger.info("="*50)
        
        for component, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"{component}: {status}")
        
        success_count = sum(results.values())
        total_count = len(results)
        
        if success_count == total_count:
            logger.info("\n🎉 All optimizations completed successfully!")
            logger.info("Your AI system now uses DeepSeek R1 for advanced reasoning.")
        else:
            logger.warning(f"\n⚠️ {total_count - success_count} optimizations failed.")
        
        return success_count == total_count

def main():
    """Main entry point"""
    optimizer = DeepSeekR1Optimizer()
    success = optimizer.run_optimization()
    
    if success:
        print("\n✅ DeepSeek R1 optimization completed!")
        print("Your AI system now leverages DeepSeek R1's advanced reasoning capabilities.")
    else:
        print("\n❌ Some optimizations failed. Check the logs for details.")

if __name__ == "__main__":
    main() 