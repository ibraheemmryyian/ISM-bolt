#!/usr/bin/env python3
"""
Start Backend Server Only - For AI Onboarding Testing
"""

import subprocess
import time
import os
import sys
from pathlib import Path

def start_backend():
    """Start the backend server"""
    print("🚀 Starting SymbioFlows Backend Server")
    print("=" * 50)
    
    # Change to backend directory
    backend_dir = Path("C:\\Users\\amrey\\Desktop\\SymbioFlows\\backend")
    if not backend_dir.exists():
        backend_dir = Path("backend")
    
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False
    
    os.chdir(backend_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check if app.js exists
    if not Path("app.js").exists():
        print("❌ app.js not found in backend directory")
        return False
    
    print("🔧 Starting Node.js backend server...")
    
    try:
        # Start the backend server
        process = subprocess.Popen(
            ["node", "app.js"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"✅ Backend server started with PID: {process.pid}")
        print("📡 Backend should be available at: http://localhost:3000")
        print("🔍 Health check: http://localhost:3000/api/health")
        print()
        print("💡 Keep this window open to keep the backend running")
        print("💡 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Wait for process to complete
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping backend server...")
            process.terminate()
            process.wait()
            print("✅ Backend server stopped")
            
    except FileNotFoundError:
        print("❌ Node.js not found. Please install Node.js")
        return False
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_backend()