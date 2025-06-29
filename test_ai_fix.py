import json
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_onboarding_ai import AdvancedOnboardingAI

def test_ai_fixes():
    """Test the AI with CircuTex Mills data to verify fixes are working"""
    
    # Test data for CircuTex Mills
    test_data = {
        "company_name": "CircuTex Mills",
        "name": "CircuTex Mills", 
        "industry": "Textile Manufacturing",
        "products": "Recycled cotton yarns, PET-based fabrics",
        "location": "Milan, Italy",
        "productionVolume": "3,200 tons",
        "mainMaterials": "Post-industrial cotton scraps, Recycled PET bottles, Denim waste",
        "processDescription": "Fiber shredding → Carding → Spinning → Dyeing → Weaving → Finishing"
    }
    
    print("🧪 Testing AI Fixes with CircuTex Mills...")
    print(f"Company: {test_data['company_name']}")
    print(f"Industry: {test_data['industry']}")
    print(f"Products: {test_data['products']}")
    print(f"Volume: {test_data['productionVolume']}")
    print(f"Materials: {test_data['mainMaterials']}")
    print(f"Processes: {test_data['processDescription']}")
    print("\n" + "="*80)
    
    try:
        # Initialize AI
        ai = AdvancedOnboardingAI()
        
        # Generate listings
        listings = ai.generate_advanced_listings(test_data)
        
        print(f"\n📊 Generated {len(listings)} listings")
        print("\n" + "="*80)
        
        # Analyze listings by type
        products = [l for l in listings if l.material_type == 'product']
        waste = [l for l in listings if l.material_type == 'waste']
        requirements = [l for l in listings if l.material_type == 'requirement']
        
        print(f"\n🏭 PRODUCTS ({len(products)}):")
        for i, product in enumerate(products, 1):
            print(f"  {i}. {product.material_name}")
            print(f"     Quantity: {product.quantity} {product.unit}")
            print(f"     Description: {product.description}")
            print(f"     Confidence: {product.confidence_score}")
            print()
        
        print(f"\n🗑️  WASTE ({len(waste)}):")
        for i, waste_item in enumerate(waste, 1):
            print(f"  {i}. {waste_item.material_name}")
            print(f"     Quantity: {waste_item.quantity} {waste_item.unit}")
            print(f"     Description: {waste_item.description}")
            print(f"     Confidence: {waste_item.confidence_score}")
            print()
        
        print(f"\n📦 REQUIREMENTS ({len(requirements)}):")
        for i, req in enumerate(requirements, 1):
            print(f"  {i}. {req.material_name}")
            print(f"     Quantity: {req.quantity} {req.unit}")
            print(f"     Description: {req.description}")
            print(f"     Confidence: {req.confidence_score}")
            print()
        
        # Quality assessment
        print("\n" + "="*80)
        print("🎯 QUALITY ASSESSMENT:")
        
        # Check for textile-specific content
        textile_indicators = ['cotton', 'PET', 'denim', 'yarn', 'fabric', 'textile', 'dyeing', 'spinning', 'weaving', 'fiber']
        generic_indicators = ['carbon capture', 'hydrogen', 'blockchain', 'digital twin', 'smart manufacturing']
        
        textile_count = 0
        generic_count = 0
        
        for listing in listings:
            description = listing.description.lower()
            name = listing.material_name.lower()
            
            if any(indicator in description or indicator in name for indicator in textile_indicators):
                textile_count += 1
            if any(indicator in description or indicator in name for indicator in generic_indicators):
                generic_count += 1
        
        print(f"  Textile-specific listings: {textile_count}")
        print(f"  Generic buzzword listings: {generic_count}")
        print(f"  Relevance ratio: {textile_count/(textile_count + generic_count)*100:.1f}%")
        
        if textile_count > generic_count:
            print("  ✅ EXCELLENT: AI generates relevant, industry-specific content")
        else:
            print("  ❌ POOR: AI still generates too much generic content")
        
        # Check for specific product types
        expected_products = ['cotton', 'PET', 'denim', 'yarn', 'fabric']
        found_products = sum(1 for p in products if any(ep in p.material_name.lower() for ep in expected_products))
        
        print(f"  Expected products found: {found_products}/{len(products)}")
        if found_products >= len(products) * 0.7:
            print("  ✅ GOOD: Most products are relevant to textile industry")
        else:
            print("  ❌ POOR: Many products are not relevant to textile industry")
        
        # Overall assessment
        print(f"\n📈 OVERALL ASSESSMENT:")
        if textile_count > generic_count and found_products >= len(products) * 0.7:
            print("  🎉 SUCCESS: AI is now generating relevant, industry-specific content!")
            print("  🚀 READY FOR LAUNCH!")
        else:
            print("  ⚠️  NEEDS IMPROVEMENT: AI still has issues with relevance")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_fixes() 