#!/usr/bin/env python3
"""
Complete ISM AI System Setup
Imports 50 companies, generates AI listings, and runs advanced matching
"""

import os
import sys
import json
import requests
import time
import subprocess
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def call_api_endpoint(endpoint, method='POST', data=None, description=""):
    """Call a backend API endpoint"""
    print(f"\n🔄 {description}...")
    try:
        url = f"http://localhost:5000{endpoint}"
        if method == 'POST':
            response = requests.post(url, json=data or {})
        else:
            response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {description} completed successfully")
            return result
        else:
            print(f"❌ {description} failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Error calling {description}: {e}")
        return None

def main():
    """Main setup function"""
    print("=" * 60)
    print("🚀 ISM AI - COMPLETE SYSTEM SETUP")
    print("=" * 60)
    
    # Check if backend is running
    print("\n🔍 Checking if backend is running...")
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not running. Please start the backend first.")
            return False
    except:
        print("❌ Backend is not running. Please start the backend first.")
        return False
    
    print("✅ Backend is running")
    
    # Step 1: Import 50 companies
    print("\n" + "="*50)
    print("📥 STEP 1: Importing 50 Gulf Companies")
    print("="*50)
    
    if not run_command("python backend/real_data_bulk_importer.py", "Importing companies"):
        return False
    
    # Step 2: Generate AI listings
    print("\n" + "="*50)
    print("🤖 STEP 2: Generating AI Listings")
    print("="*50)
    
    listings_result = call_api_endpoint(
        "/api/ai/generate-all-listings",
        description="Generating AI listings for all companies"
    )
    
    if not listings_result:
        return False
    
    print(f"📊 Generated {listings_result.get('summary', {}).get('total_listings_generated', 0)} listings")
    
    # Step 3: Run AI matching engine
    print("\n" + "="*50)
    print("🔗 STEP 3: Running Advanced AI Matching")
    print("="*50)
    
    if not run_command("python backend/revolutionary_ai_matching.py", "Running revolutionary AI matching"):
        return False
    
    # Step 4: Run GNN reasoning
    print("\n" + "="*50)
    print("🧠 STEP 4: Running GNN Reasoning Engine")
    print("="*50)
    
    if not run_command("python backend/gnn_reasoning_engine.py", "Running GNN reasoning"):
        return False
    
    # Step 5: Run multi-hop symbiosis
    print("\n" + "="*50)
    print("🕸️ STEP 5: Running Multi-Hop Symbiosis")
    print("="*50)
    
    if not run_command("python backend/multi_hop_symbiosis_service.py", "Running symbiosis analysis"):
        return False
    
    # Step 6: Generate comprehensive matches
    print("\n" + "="*50)
    print("🎯 STEP 6: Generating Comprehensive Matches")
    print("="*50)
    
    matches_result = call_api_endpoint(
        "/api/comprehensive-match-analysis",
        data={"include_ai_insights": True, "include_gnn_analysis": True},
        description="Generating comprehensive matches"
    )
    
    if not matches_result:
        return False
    
    # Step 7: Get system statistics
    print("\n" + "="*50)
    print("📊 STEP 7: System Statistics")
    print("="*50)
    
    stats_result = call_api_endpoint(
        "/api/admin/stats",
        method='GET',
        description="Getting system statistics"
    )
    
    if stats_result:
        stats = stats_result.get('stats', {})
        print(f"📈 System Statistics:")
        print(f"   - Total Users: {stats.get('total_users', 0)}")
        print(f"   - Total Companies: {stats.get('total_companies', 0)}")
        print(f"   - Total Materials: {stats.get('total_materials', 0)}")
        print(f"   - Total Matches: {stats.get('total_matches', 0)}")
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 SYSTEM SETUP COMPLETE!")
    print("="*60)
    print("\n📊 What was generated:")
    print("   ✅ 50 Gulf companies imported")
    print("   ✅ AI-generated material listings")
    print("   ✅ Advanced AI matches (4-factor matching)")
    print("   ✅ GNN-based reasoning and insights")
    print("   ✅ Multi-hop symbiosis networks")
    print("   ✅ Comprehensive match analysis")
    print("\n🔗 Access your system:")
    print("   - Frontend: http://localhost:5173")
    print("   - Backend API: http://localhost:5000")
    print("   - Admin Dashboard: Available in frontend")
    print("\n🧪 Test the system:")
    print("   - Browse companies and AI listings")
    print("   - View AI-generated matches")
    print("   - Explore symbiosis networks")
    print("   - Use admin features to upgrade users")
    print("\n" + "="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!") 