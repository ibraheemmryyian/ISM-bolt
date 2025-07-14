#!/usr/bin/env python3
"""
Materials Project API Key Helper
Helps you get a new API key for the Materials Project API.
"""

import webbrowser
import os
import sys

def main():
    print("🔬 MATERIALS PROJECT API KEY HELPER")
    print("=" * 50)
    print()
    print("The Materials Project API now requires 32-character API keys.")
    print("Your current key appears to be the old 16-character format.")
    print()
    print("To get a new API key:")
    print("1. Visit: https://materialsproject.org/api")
    print("2. Sign up or log in to your account")
    print("3. Generate a new API key (32 characters)")
    print("4. Update your .env file with the new key")
    print()
    
    # Open the API page in browser
    try:
        webbrowser.open("https://materialsproject.org/api")
        print("✅ Opened Materials Project API page in your browser")
    except:
        print("⚠️  Could not open browser automatically")
        print("   Please manually visit: https://materialsproject.org/api")
    
    print()
    print("After getting your new key:")
    print("1. Open your .env file")
    print("2. Replace the MP_API_KEY line with:")
    print("   MP_API_KEY=your_new_32_character_key_here")
    print("3. Save the file")
    print("4. Run your tests again")
    print()
    
    # Check current key
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('MP_API_KEY='):
                    current_key = line.split('=')[1].strip()
                    print(f"Current key length: {len(current_key)} characters")
                    if len(current_key) != 32:
                        print("❌ This appears to be an old format key")
                    else:
                        print("✅ This appears to be the correct format")
                    break
    else:
        print("❌ .env file not found")
    
    print()
    print("The Materials Project API provides:")
    print("- Scientific material properties")
    print("- Crystal structure data")
    print("- Electronic properties")
    print("- Thermodynamic data")
    print("- And much more!")
    print()
    print("This will be your primary scientific data source for materials analysis.")

if __name__ == "__main__":
    main() 