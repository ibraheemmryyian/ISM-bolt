import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json
import sys

class RevolutionaryAIMatching:
    """Integrated Industrial Symbiosis Matching AI"""
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
    def predict_compatibility(self, buyer, seller):
        """Predict compatibility with sustainability scoring"""
        # Semantic matching
        buyer_text = self._prepare_buyer_text(buyer)
        seller_text = self._prepare_seller_text(seller)
        semantic_score = self._calculate_semantic_similarity(buyer_text, seller_text)
        
        # Sustainability impact
        sustainability_score = self._calculate_sustainability_impact(buyer, seller)
        
        # Composite revolutionary score
        revolutionary_score = (0.7 * semantic_score + 0.3 * sustainability_score)
        
        return {
            "semantic_score": round(semantic_score, 3),
            "sustainability_score": round(sustainability_score, 3),
            "revolutionary_score": round(revolutionary_score, 3),
            "match_quality": self._quality_label(revolutionary_score)
        }
    
    def _calculate_sustainability_impact(self, buyer, seller):
        """Calculate CO₂ reduction impact"""
        # Emission factors (kgCO₂/kg)
        EMISSION_FACTORS = {
            'cement': 0.95, 'steel': 1.85, 'plastic': 3.5,
            'textiles': 2.1, 'chemicals': 1.7, 'metals': 1.9
        }
        
        # Calculate saved emissions
        waste_type = buyer.get('waste_type', 'generic')
        factor = EMISSION_FACTORS.get(waste_type, 1.0)
        distance_km = buyer.get('distance_to_seller', 50)
        quantity = buyer.get('quantity', 1000)
        
        return factor * distance_km * (quantity / 1000)
    
    def _quality_label(self, score):
        """Categorize match quality"""
        if score >= 0.8: return "Perfect Symbiosis"
        if score >= 0.6: return "High Value"
        if score >= 0.4: return "Viable Match"
        return "Low Potential"
    
    def _prepare_buyer_text(self, buyer):
        return (
            f"Industry: {buyer['industry']}. "
            f"Waste Type: {buyer['waste_type']}. "
            f"Quantity: {buyer['quantity']} tons."
        )
    
    def _prepare_seller_text(self, seller):
        return (
            f"Material Needed: {seller['material_needed']}. "
            f"Capabilities: {', '.join(seller.get('capabilities', []))}."
        )
    
    def _calculate_semantic_similarity(self, text1, text2):
        embeddings = self.model.encode([text1, text2])
        # Convert to numpy arrays with proper shape
        emb1 = np.array([embeddings[0]])
        emb2 = np.array([embeddings[1]])
        return cosine_similarity(emb1, emb2)[0][0]

if __name__ == "__main__":
    ai = RevolutionaryAIMatching()
    input_data = json.loads(sys.argv[1])
    result = ai.predict_compatibility(input_data['buyer'], input_data['seller'])
    print(json.dumps(result))
