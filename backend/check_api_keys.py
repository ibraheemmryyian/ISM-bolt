#!/usr/bin/env python3
"""
Check API Keys
Script to check what API keys are available and diagnose issues.
"""

import os
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_api_keys():
    """Check what API keys are available"""
    
    logger.info("🔑 Checking Available API Keys")
    logger.info("=" * 50)
    
    # Check all possible API keys
    api_keys = {
        "Materials Project": [
            "MP_API_KEY",
            "MATERIALS_PROJECT_API_KEY"
        ],
        "Next Gen Materials": [
            "NEXT_GEN_MATERIALS_API_KEY",
            "NEXTGEN_MATERIALS_API_KEY",
            "NEXT_GEN_API_KEY"
        ],
        "DeepSeek AI": [
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_KEY"
        ],
        "DeepSeek R1": [
            "DEEPSEEK_R1_API_KEY",
            "DEEPSEEK_R1_KEY"
        ],
        "News API": [
            "NEWS_API_KEY",
            "NEWSAPI_KEY"
        ],
        "Freightos": [
            "FREIGHTOS_API_KEY",
            "FREIGHTOS_KEY"
        ],
        "PubChem": [
            "PUBCHEM_API_KEY",
            "PUBCHEM_KEY"
        ]
    }
    
    available_keys = {}
    
    for service, key_names in api_keys.items():
        found_key = None
        for key_name in key_names:
            key_value = os.getenv(key_name)
            if key_value:
                found_key = key_name
                break
        
        if found_key:
            # Show first few characters of the key
            masked_key = key_value[:8] + "..." if len(key_value) > 8 else key_value
            logger.info(f"✅ {service}: {found_key} = {masked_key}")
            available_keys[service] = True
        else:
            logger.info(f"❌ {service}: No API key found")
            available_keys[service] = False
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 SUMMARY:")
    
    total_services = len(api_keys)
    available_services = sum(available_keys.values())
    
    logger.info(f"   Total Services: {total_services}")
    logger.info(f"   Available: {available_services}")
    logger.info(f"   Missing: {total_services - available_services}")
    
    if available_services == 0:
        logger.warning("⚠️  No API keys found! The system will use fallback data.")
    elif available_services < total_services:
        logger.info("ℹ️  Some API keys missing. The system will work with available sources.")
    else:
        logger.info("🎉 All API keys available!")
    
    logger.info("\n" + "=" * 50)
    logger.info("🔧 TO FIX MISSING KEYS:")
    
    missing_services = [service for service, available in available_keys.items() if not available]
    
    for service in missing_services:
        if service == "Next Gen Materials":
            logger.info(f"   {service}: Set NEXT_GEN_MATERIALS_API_KEY environment variable")
        elif service == "News API":
            logger.info(f"   {service}: Set NEWS_API_KEY environment variable")
        elif service == "PubChem":
            logger.info(f"   {service}: PubChem is free, no API key needed")
    
    return available_keys

if __name__ == "__main__":
    check_api_keys() 