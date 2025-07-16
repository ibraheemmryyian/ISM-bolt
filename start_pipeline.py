#!/usr/bin/env python3
"""
Simple Pipeline Starter - Can be run from any directory
"""

import os
import sys
import subprocess

def main():
    print("🚀 ISM AI Pipeline Starter")
    print("=" * 40)
    
    # Find the project directory
    current_dir = os.getcwd()
    project_dir = None
    
    # Look for the project directory
    for root, dirs, files in os.walk(current_dir):
        if "backend" in dirs and "frontend" in dirs:
            project_dir = root
            break
    
    if not project_dir:
        # Try common locations
        possible_paths = [
            "C:\\Users\\amrey\\Desktop\\ISM [AI]",
            os.path.join(current_dir, "ISM [AI]"),
            os.path.join(os.path.dirname(current_dir), "ISM [AI]")
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "backend")):
                project_dir = path
                break
    
    if not project_dir:
        print("❌ Could not find ISM AI project directory")
        print("Please run this script from the project directory or Desktop")
        return False
    
    print(f"📁 Found project at: {project_dir}")
    
    # Change to project directory
    os.chdir(project_dir)
    print(f"📁 Changed to: {os.getcwd()}")
    
    # Check if backend is running
    print("\n🔍 Checking if backend is running...")
    try:
        import requests
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running!")
        else:
            print("❌ Backend is not responding properly")
            return False
    except:
        print("❌ Backend is not running. Please start the backend first:")
        print("   cd backend && npm start")
        return False
    
    # Run the pipeline
    print("\n🚀 Starting complete pipeline...")
    try:
        result = subprocess.run([sys.executable, "run_complete_pipeline.py"], 
                              capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        if result.returncode == 0:
            print("\n🎉 Pipeline completed successfully!")
            return True
        else:
            print(f"\n❌ Pipeline failed with code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Pipeline timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 