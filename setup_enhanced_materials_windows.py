#!/usr/bin/env python3
"""
Windows Setup Script for Enhanced Materials System
Installs required packages and sets up the system for Windows compatibility.
"""

import os
import sys
import subprocess
import logging

# Configure logging for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_package(package_name):
    """Install a Python package"""
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"[OK] {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Failed to install {package_name}: {str(e)}")
        return False

def main():
    """Main setup function"""
    logger.info("WINDOWS SETUP FOR ENHANCED MATERIALS SYSTEM")
    logger.info("=" * 50)
    
    # Required packages for the enhanced materials system
    required_packages = [
        'torch',
        'transformers',
        'tokenizers',
        'numpy',
        'scipy',
        'scikit-learn',
        'flask',
        'flask-cors',
        'requests',
        'datasets',
        'psutil',
        'pyyaml'
    ]
    
    logger.info("Installing required packages...")
    
    success_count = 0
    for package in required_packages:
        if install_package(package):
            success_count += 1
    
    logger.info(f"Installation complete: {success_count}/{len(required_packages)} packages installed")
    
    if success_count == len(required_packages):
        logger.info("[OK] All packages installed successfully!")
        logger.info("You can now run the enhanced materials system.")
    else:
        logger.warning("[WARN] Some packages failed to install. The system may have limited functionality.")
    
    # Create environment file
    logger.info("Creating environment configuration...")
    
    env_content = """# Enhanced Materials Integration Configuration
# Generated on: {datetime}

# Next Gen Materials API
NEXT_GEN_MATERIALS_API_KEY=
NEXT_GEN_MATERIALS_BASE_URL=https://api.next-gen-materials.com/v1
NEXT_GEN_MATERIALS_RATE_LIMIT=1000

# MaterialsBERT Service
MATERIALSBERT_ENABLED=true
MATERIALSBERT_ENDPOINT=http://localhost:8001
MATERIALSBERT_MODEL_PATH=/app/models/materialsbert
MATERIALSBERT_CACHE_SIZE=1000

# Enhanced Materials Service
ENHANCED_MATERIALS_CACHE_TIMEOUT=3600000
ENHANCED_MATERIALS_MAX_CONCURRENT_REQUESTS=10
ENHANCED_MATERIALS_RETRY_ATTEMPTS=3

# AI Integration
DEEPSEEK_API_KEY=sk-7ce79f30332d45d5b3acb8968b052132
DEEPSEEK_MODEL=deepseek-coder

# Performance Settings
MATERIALS_ANALYSIS_TIMEOUT=30000
CROSS_VALIDATION_ENABLED=true
AI_ENHANCED_INSIGHTS_ENABLED=true

# Monitoring
MATERIALS_ANALYTICS_ENABLED=true
PERFORMANCE_MONITORING_ENABLED=true
""".format(datetime=__import__('datetime').datetime.now().isoformat())
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        logger.info("[OK] Environment file created: .env")
    except Exception as e:
        logger.error(f"[ERROR] Failed to create environment file: {str(e)}")
    
    logger.info("=" * 50)
    logger.info("SETUP COMPLETE!")
    logger.info("=" * 50)
    logger.info("Next steps:")
    logger.info("1. Edit .env file to add your API keys")
    logger.info("2. Run: python start_enhanced_materials_system.py")
    logger.info("3. Or run: python enhanced_materials_integration_demo.py")

if __name__ == "__main__":
    main() 