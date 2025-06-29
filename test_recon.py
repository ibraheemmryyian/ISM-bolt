#!/usr/bin/env python3
import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_onboarding_ai import AdvancedOnboardingAI

def test_recon_materials():
    """Test the onboarding AI with ReCon Materials data"""
    
    # Test data
    company_data = {
        "name": "ReCon Materials",
        "industry": "Building Materials",
        "products": "Recycled concrete aggregates, Composite decking",
        "location": "Rotterdam, Netherlands",
        "productionVolume": "150,000 cubic meters",
        "mainMaterials": "Demolition concrete (70%), Recycled plastics (25%), Wood chips (5%)",
        "processDescription": "Crushing → Screening → Material blending → Extrusion → Curing → Cutting"
    }
    
    print("🧪 Testing Advanced Onboarding AI with ReCon Materials")
    print("=" * 60)
    
    try:
        # Initialize AI
        ai = AdvancedOnboardingAI()
        
        # Generate listings
        listings = ai.generate_advanced_listings(company_data)
        
        print(f"✅ Generated {len(listings)} listings")
        print("\n📊 LISTINGS SUMMARY:")
        print("-" * 60)
        
        for i, listing in enumerate(listings, 1):
            print(f"\n{i}. {listing.material_name} ({listing.material_type})")
            print(f"   Quantity: {listing.quantity} {listing.unit}")
            print(f"   Confidence: {listing.confidence_score}")
            print(f"   Description: {listing.description}")
            print(f"   Reasoning: {listing.reasoning}")
            print(f"   Industry Relevance: {listing.industry_relevance}")
            print(f"   Sustainability Impact: {listing.sustainability_impact}")
            print(f"   Market Demand: {listing.market_demand}")
            print(f"   Regulatory Compliance: {listing.regulatory_compliance}")
        
        # Convert to JSON for verification
        result = []
        for listing in listings:
            result.append({
                'name': listing.material_name,
                'type': listing.material_type,
                'quantity': listing.quantity,
                'unit': listing.unit,
                'description': listing.description,
                'confidence_score': listing.confidence_score,
                'reasoning': listing.reasoning,
                'industry_relevance': listing.industry_relevance,
                'sustainability_impact': listing.sustainability_impact,
                'market_demand': listing.market_demand,
                'regulatory_compliance': listing.regulatory_compliance,
                'ai_generated': listing.ai_generated
            })
        
        print(f"\n📋 JSON OUTPUT:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recon_materials() 