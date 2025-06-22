import sys
from pathlib import Path

# Add the absolute path to the application directory
app_path = Path("c:/Users/amrey/Desktop/industrial symbiosis market place")
sys.path.insert(0, str(app_path))

from app.services.ai_service import AIService

def run_test():
    print("Testing Industrial AI Matching Service...")
    service = AIService()
    
    # Define a buyer with specific needs
    buyer = {
        "industry": "Manufacturing",
        "description": "Need predictive maintenance solution for factory equipment",
        "required_capabilities": ["machine learning", "IoT", "predictive analytics"],
        "budget": 50000,
        "timeline": "3 months"
    }
    
    # Define seller profiles
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
        },
        {
            "id": 3,
            "industries": ["Manufacturing", "Logistics"],
            "capabilities": ["IoT", "predictive analytics", "supply chain optimization"],
            "description": "Industrial IoT and predictive maintenance solutions",
            "pricing_range": [45000, 90000],
            "years_experience": 4
        }
    ]
    
    # Find matches
    matches = service.find_matches(buyer, sellers)
    
    print("\nBuyer Requirements:")
    print(f"Industry: {buyer['industry']}")
    print(f"Description: {buyer['description']}")
    print(f"Required Capabilities: {', '.join(buyer['required_capabilities'])}")
    print(f"Budget: ${buyer['budget']}")
    
    print("\nSeller Profiles:")
    for seller in sellers:
        print(f"\nSeller ID: {seller['id']}")
        print(f"Industries: {', '.join(seller['industries'])}")
        print(f"Capabilities: {', '.join(seller['capabilities'])}")
        print(f"Pricing Range: ${seller['pricing_range'][0]}-${seller['pricing_range'][1]}")
    
    print("\nTop Matches:")
    for score, seller_id in matches:
        seller = next(s for s in sellers if s["id"] == seller_id)
        print(f"Seller ID: {seller_id}, Score: {score:.2f}")
        print(f"  - Industries: {', '.join(seller['industries'])}")
        print(f"  - Capabilities: {', '.join(seller['capabilities'])}")
        print(f"  - Pricing: ${seller['pricing_range'][0]}-${seller['pricing_range'][1]}")

if __name__ == "__main__":
    run_test()
