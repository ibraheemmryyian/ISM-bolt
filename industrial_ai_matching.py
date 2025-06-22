import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class IndustrialAIMatchingService:
    def __init__(self):
        # Load pre-trained Hugging Face model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def match_buyers_sellers(self, buyer_needs: Dict, seller_profiles: List[Dict]) -> List[Tuple[float, int]]:
        """Find best matches between buyers and sellers using AI"""
        # Prepare buyer embedding
        buyer_text = self._prepare_buyer_text(buyer_needs)
        buyer_embedding = self.model.encode([buyer_text])[0]
        
        # Prepare seller embeddings
        seller_embeddings = []
        seller_data = []
        for seller in seller_profiles:
            seller_text = self._prepare_seller_text(seller)
            embedding = self.model.encode([seller_text])[0]
            seller_embeddings.append(embedding)
            seller_data.append(seller)
        
        # Calculate semantic similarity scores
        similarities = cosine_similarity([buyer_embedding], seller_embeddings)[0]
        
        # Calculate industry compatibility scores
        industry_scores = []
        for seller in seller_profiles:
            score = 1.0 if buyer_needs["industry"] in seller["industries"] else 0.0
            industry_scores.append(score)
        
        # Calculate capability matching scores
        capability_scores = []
        for seller in seller_profiles:
            common_capabilities = set(buyer_needs["required_capabilities"]) & set(seller["capabilities"])
            score = len(common_capabilities) / max(len(buyer_needs["required_capabilities"]), 1)
            capability_scores.append(score)
        
        # Calculate pricing compatibility scores
        pricing_scores = []
        for seller in seller_profiles:
            if seller["pricing_range"][0] <= buyer_needs["budget"] <= seller["pricing_range"][1]:
                score = 1.0
            elif buyer_needs["budget"] < seller["pricing_range"][0]:
                score = 0.2  # Buyer budget too low
            else:
                score = 0.5  # Buyer budget higher than seller range
            pricing_scores.append(score)
        
        # Combine scores (weighted average)
        combined_scores = []
        for i, seller in enumerate(seller_profiles):
            combined = (
                0.4 * similarities[i] + 
                0.2 * industry_scores[i] +
                0.2 * capability_scores[i] +
                0.2 * pricing_scores[i]
            )
            combined_scores.append((combined, seller["id"]))
        
        # Sort matches by combined score descending and return top matches
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        return combined_scores[:10]
    
    def _prepare_buyer_text(self, buyer: Dict) -> str:
        """Prepare text for buyer embedding"""
        return (
            f"Industry: {buyer['industry']}. "
            f"Needs: {buyer['description']}. "
            f"Required capabilities: {', '.join(buyer['required_capabilities'])}. "
            f"Budget: {buyer['budget']}. "
            f"Timeline: {buyer['timeline']}."
        )
    
    def _prepare_seller_text(self, seller: Dict) -> str:
        """Prepare text for seller embedding"""
        return (
            f"Industries: {', '.join(seller['industries'])}. "
            f"Capabilities: {', '.join(seller['capabilities'])}. "
            f"Services: {seller['description']}. "
            f"Pricing range: {seller['pricing_range'][0]}-{seller['pricing_range'][1]}. "
            f"Experience: {seller['years_experience']} years."
        )

# Example usage
if __name__ == "__main__":
    service = IndustrialAIMatchingService()
    
    buyer = {
        "industry": "Manufacturing",
        "description": "Need predictive maintenance solution for factory equipment",
        "required_capabilities": ["machine learning", "IoT", "predictive analytics"],
        "budget": 50000,
        "timeline": "3 months"
    }
    
    sellers = [
        {
            "id": 1,
            "industries": ["Manufacturing", "Energy"],
            "capabilities": ["machine learning", "predictive analytics", "data visualization"],
            "description": "AI-powered predictive maintenance solutions",
            "pricing_range": [40000, 100000],
            "years_experience": 5
        },
        {
            "id": 2,
            "industries": ["Healthcare", "Retail"],
            "capabilities": ["computer vision", "NLP", "recommendation systems"],
            "description": "Computer vision solutions for quality control",
            "pricing_range": [30000, 70000],
            "years_experience": 3
        }
    ]
    
    matches = service.match_buyers_sellers(buyer, sellers)
    print("Top matches:", matches)
