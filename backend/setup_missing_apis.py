#!/usr/bin/env python3
"""
Setup Missing APIs
Helps identify and guide setup of missing API keys for complete system functionality.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_api_keys():
    """Check which API keys are missing and provide setup guidance"""
    
    print("🔍 CHECKING API KEY STATUS")
    print("=" * 50)
    
    # Define all required API keys
    api_keys = {
        "Materials Project": {
            "env_var": "MP_API_KEY",
            "url": "https://materialsproject.org/api",
            "status": "✅ CONFIGURED" if os.getenv("MP_API_KEY") else "❌ MISSING",
            "description": "Scientific materials database"
        },
        "DeepSeek AI": {
            "env_var": "DEEPSEEK_API_KEY", 
            "url": "https://platform.deepseek.com/",
            "status": "✅ CONFIGURED" if os.getenv("DEEPSEEK_API_KEY") else "❌ MISSING",
            "description": "AI analysis and reasoning"
        },
        "DeepSeek R1": {
            "env_var": "DEEPSEEK_R1_API_KEY",
            "url": "https://platform.deepseek.com/",
            "status": "✅ CONFIGURED" if os.getenv("DEEPSEEK_R1_API_KEY") else "❌ MISSING", 
            "description": "Advanced AI reasoning"
        },
        "Freightos": {
            "env_var": "FREIGHTOS_API_KEY",
            "url": "https://www.freightos.com/developers/",
            "status": "✅ CONFIGURED" if os.getenv("FREIGHTOS_API_KEY") else "❌ MISSING",
            "description": "Logistics and shipping data"
        },
        "News API": {
            "env_var": "NEWS_API_KEY",
            "url": "https://newsapi.org/register",
            "status": "❌ MISSING",  # You need to add this
            "description": "Market intelligence and news"
        },
        "Next Gen Materials": {
            "env_var": "NEXT_GEN_MATERIALS_API_KEY", 
            "url": "https://next-gen-materials.com/api",
            "status": "❌ MISSING",  # You need to add this
            "description": "Advanced materials market data"
        }
    }
    
    # Check each API
    configured_count = 0
    total_count = len(api_keys)
    
    for name, info in api_keys.items():
        print(f"{info['status']} {name}")
        print(f"   Description: {info['description']}")
        print(f"   URL: {info['url']}")
        if info['status'] == "❌ MISSING":
            print(f"   Action: Add {info['env_var']} to your .env file")
        print()
        if info['status'] == "✅ CONFIGURED":
            configured_count += 1
    
    print("=" * 50)
    print(f"📊 SUMMARY: {configured_count}/{total_count} APIs configured")
    print(f"🎯 Coverage: {(configured_count/total_count)*100:.1f}%")
    
    if configured_count == total_count:
        print("🎉 ALL APIS CONFIGURED! Your system is ready for maximum functionality!")
    elif configured_count >= total_count * 0.7:
        print("✅ Excellent coverage! System will work with most features.")
    elif configured_count >= total_count * 0.5:
        print("⚠️  Good coverage. Consider adding missing APIs for better functionality.")
    else:
        print("❌ Limited coverage. Adding more APIs will significantly improve functionality.")
    
    # Provide setup instructions for missing APIs
    missing_apis = [name for name, info in api_keys.items() if info['status'] == "❌ MISSING"]
    
    if missing_apis:
        print("\n" + "=" * 50)
        print("🔧 SETUP INSTRUCTIONS FOR MISSING APIS:")
        print()
        
        for api_name in missing_apis:
            info = api_keys[api_name]
            print(f"📋 {api_name}:")
            print(f"   1. Visit: {info['url']}")
            print(f"   2. Sign up for an API key")
            print(f"   3. Add to .env file: {info['env_var']}=your_api_key_here")
            print()
    
    print("=" * 50)
    print("💡 TIP: Even with missing APIs, the system will work using available sources!")
    print("   The system gracefully handles missing APIs and uses fallback mechanisms.")

if __name__ == "__main__":
    check_api_keys() 