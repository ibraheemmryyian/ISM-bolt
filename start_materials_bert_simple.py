#!/usr/bin/env python3
"""
Startup script for Simplified MaterialsBERT Service
"""

import os
import sys
import subprocess
import time
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'numpy', 'requests', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Please run the installation script first:")
        logger.error("  install_materials_bert.bat (Windows)")
        logger.error("  install_materials_bert.sh (Linux/Mac)")
        return False
    
    return True

def start_service():
    """Start the MaterialsBERT service"""
    try:
        # Change to backend directory
        backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
        if not os.path.exists(backend_dir):
            logger.error(f"Backend directory not found: {backend_dir}")
            return False
        
        os.chdir(backend_dir)
        logger.info(f"Changed to directory: {os.getcwd()}")
        
        # Check if the service file exists
        service_file = 'materials_bert_service_simple.py'
        if not os.path.exists(service_file):
            logger.error(f"Service file not found: {service_file}")
            return False
        
        logger.info("Starting MaterialsBERT Simple Service...")
        logger.info("Service will be available at http://localhost:5002")
        logger.info("Press Ctrl+C to stop the service")
        
        # Start the service
        subprocess.run([sys.executable, service_file])
        
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Error starting service: {e}")
        return False
    
    return True

def test_service():
    """Test if the service is running correctly"""
    try:
        # Wait a moment for service to start
        time.sleep(2)
        
        # Test health endpoint
        response = requests.get('http://localhost:5002/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ Service is healthy: {data}")
            return True
        else:
            logger.error(f"✗ Service health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Service test failed: {e}")
        return False

if __name__ == '__main__':
    logger.info("MaterialsBERT Simple Service Startup")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start the service
    if start_service():
        logger.info("Service started successfully")
    else:
        logger.error("Failed to start service")
        sys.exit(1) 