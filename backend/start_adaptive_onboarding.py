#!/usr/bin/env python3
"""
Startup script for Adaptive AI Onboarding Flask Server
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_python_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'flask',
        'flask_cors',
        'numpy',
        'pandas',
        'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("✅ Dependencies installed")
    else:
        print("✅ All dependencies are installed")

def start_flask_server():
    """Start the Flask server for adaptive onboarding"""
    try:
        print("🚀 Starting Adaptive AI Onboarding Flask Server...")
        
        # Change to the backend directory
        backend_dir = Path(__file__).parent
        os.chdir(backend_dir)
        
        # Start the Flask server
        subprocess.run([
            sys.executable, 
            'adaptive_onboarding_server.py'
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def health_check():
    """Check if the server is running"""
    try:
        response = requests.get('http://localhost:5003/health', timeout=5)
        if response.status_code == 200:
            print("✅ Adaptive onboarding server is running")
            return True
        else:
            print("❌ Server responded with error")
            return False
    except requests.exceptions.RequestException:
        print("❌ Server is not responding")
        return False

if __name__ == '__main__':
    print("🔧 Setting up Adaptive AI Onboarding Server...")
    
    # Check dependencies
    check_python_dependencies()
    
    # Start server
    start_flask_server() 