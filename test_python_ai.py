#!/usr/bin/env python3
"""
Comprehensive test script for the Revolutionary AI Matching Engine
Tests all dependencies and core functionality
"""

import sys
import json
import traceback
from datetime import datetime

def test_imports():
    """Test all required imports"""
    print("🔍 Testing Python imports...")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence_transformers imported successfully")
    except ImportError as e:
        print(f"❌ sentence_transformers import failed: {e}")
        return False
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ sklearn.metrics.pairwise imported successfully")
    except ImportError as e:
        print(f"❌ sklearn.metrics.pairwise import failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        print("✅ sklearn.ensemble imported successfully")
    except ImportError as e:
        print(f"❌ sklearn.ensemble import failed: {e}")
        return False
    
    try:
        import networkx as nx
        print("✅ networkx imported successfully")
    except ImportError as e:
        print(f"❌ networkx import failed: {e}")
        return False
    
    print("✅ All core imports successful")
    return True

def test_sentence_transformer():
    """Test sentence transformer model loading"""
    print("\n🔍 Testing sentence transformer model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-mpnet-base-v2')
        print("✅ Sentence transformer model loaded successfully")
        
        # Test encoding
        test_texts = ["chemical waste", "industrial solvents", "waste glycerin"]
        embeddings = model.encode(test_texts)
        print(f"✅ Model encoding test successful - shape: {embeddings.shape}")
        return True
    except Exception as e:
        print(f"❌ Sentence transformer test failed: {e}")
        return False

def test_ai_engine():
    """Test the main AI engine"""
    print("\n🔍 Testing AI engine...")
    
    try:
        # Import the main AI class
        from revolutionary_ai_matching import RevolutionaryAIMatching
        
        # Initialize the AI
        ai = RevolutionaryAIMatching()
        print("✅ AI engine initialized successfully")
        
        # Test data
        test_company = {
            "id": "test_company",
            "name": "NovaChem Solutions",
            "industry": "Chemical Production",
            "location": "Antwerp, Belgium",
            "productionVolume": "120,000 liters",
            "mainMaterials": "Waste glycerin, Ethylene oxide, Used catalysts",
            "processDescription": "Catalytic conversion → Distillation → Purification → Quality testing → Drum filling",
            "products": "Industrial solvents, Plasticizers, PH adjusters"
        }
        
        # Test AI listings generation
        listings = ai.generate_ai_listings(test_company, [], [])
        print(f"✅ AI listings generation successful - generated {len(listings)} listings")
        
        # Test compatibility prediction
        buyer = {
            "id": "buyer1",
            "industry": "Chemical Production",
            "location": "Antwerp, Belgium"
        }
        
        seller = {
            "id": "seller1",
            "industry": "Chemical Production",
            "location": "Brussels, Belgium"
        }
        
        result = ai.predict_compatibility(buyer, seller)
        print(f"✅ Compatibility prediction successful - score: {result.get('revolutionary_score', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ AI engine test failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

def test_command_line_interface():
    """Test the command line interface"""
    print("\n🔍 Testing command line interface...")
    
    try:
        # Test data
        test_data = {
            "currentCompany": {
                "id": "test_company",
                "name": "NovaChem Solutions",
                "industry": "Chemical Production",
                "location": "Antwerp, Belgium",
                "productionVolume": "120,000 liters",
                "mainMaterials": "Waste glycerin, Ethylene oxide, Used catalysts",
                "processDescription": "Catalytic conversion → Distillation → Purification → Quality testing → Drum filling",
                "products": "Industrial solvents, Plasticizers, PH adjusters"
            },
            "allCompanies": [],
            "allMaterials": []
        }
        
        # Simulate command line call
        sys.argv = [
            'revolutionary_ai_matching.py',
            '--action', 'infer_listings',
            '--data', json.dumps(test_data)
        ]
        
        # Import and run main function
        from revolutionary_ai_matching import main
        main()
        print("✅ Command line interface test successful")
        return True
    except Exception as e:
        print(f"❌ Command line interface test failed: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting comprehensive AI engine tests...")
    print(f"📅 Test started at: {datetime.now()}")
    print(f"🐍 Python version: {sys.version}")
    
    tests = [
        ("Import Test", test_imports),
        ("Sentence Transformer Test", test_sentence_transformer),
        ("AI Engine Test", test_ai_engine),
        ("Command Line Interface Test", test_command_line_interface)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 Running {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! AI engine is ready for production.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 