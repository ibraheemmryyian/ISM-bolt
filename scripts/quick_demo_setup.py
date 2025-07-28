#!/usr/bin/env python3
"""
Quick Demo Setup - One Command Demo Preparation
Run this to prepare your system for video capture demo
"""

import asyncio
import json
import os
import sys
from pathlib import Path

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    🎬 SYMBIOFLOWS DEMO SETUP                 ║
║              Prepare Your System for Video Capture          ║
╚══════════════════════════════════════════════════════════════╝
""")

async def main():
    print_banner()
    
    print("🔍 Checking for company data file...")
    
    # Look for data files
    data_files = []
    search_paths = [
        "data/",
        "../data/", 
        "./",
        "../"
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in os.listdir(search_path):
                if file.endswith('.json') and any(keyword in file.lower() for keyword in ['company', 'companies', 'data', 'gulf', 'real']):
                    data_files.append(os.path.join(search_path, file))
    
    if data_files:
        print(f"✅ Found data files:")
        for i, file in enumerate(data_files, 1):
            print(f"   {i}. {file}")
        
        # Use first file or let user choose
        if len(data_files) == 1:
            selected_file = data_files[0]
            print(f"📁 Using: {selected_file}")
        else:
            print(f"\n📁 Using first file: {data_files[0]}")
            selected_file = data_files[0]
    else:
        print("⚠️  No company data file found. Will create sample data.")
        selected_file = None
    
    print("\n🚀 Starting demo setup...")
    
    # Import and run the demo setup
    try:
        from setup_demo_environment import DemoEnvironmentSetup
        
        setup = DemoEnvironmentSetup(
            data_file=selected_file,
            force_setup=True
        )
        
        await setup.setup_complete_demo_environment()
        
        print("\n✅ Demo setup completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Start your frontend: cd frontend && npm run dev")
        print("2. Navigate to http://localhost:5173")
        print("3. Click 'Get Started' to begin demo")
        print("4. Follow the AI onboarding flow")
        print("5. View your material listings and matches")
        print("\n🎬 Ready for video capture!")
        
    except ImportError:
        print("❌ Demo setup service not found. Please ensure you're in the project root directory.")
        return 1
    except Exception as e:
        print(f"❌ Demo setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Demo setup cancelled by user")
        sys.exit(1)