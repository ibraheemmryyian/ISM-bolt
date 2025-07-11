#!/usr/bin/env python3
"""
Simple Database Debug Script
Checks what data exists in key tables
"""

import os
import requests
import json

def check_table_data():
    """Check what data exists in key tables"""
    
    # Get environment variables
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Missing environment variables!")
        print("Please set SUPABASE_URL and SUPABASE_ANON_KEY")
        return
    
    headers = {
        'apikey': supabase_key,
        'Authorization': f'Bearer {supabase_key}',
        'Content-Type': 'application/json'
    }
    
    # Tables to check
    tables = [
        'companies',
        'company_applications', 
        'materials',
        'users',
        'profiles',
        'subscriptions'
    ]
    
    print("🔍 Checking Database Tables")
    print("=" * 40)
    
    for table in tables:
        try:
            response = requests.get(
                f"{supabase_url}/rest/v1/{table}?select=*&limit=5",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                count = len(data)
                print(f"✅ {table}: {count} records")
                
                if count > 0:
                    # Show first record structure
                    first_record = data[0]
                    print(f"   Sample: {list(first_record.keys())}")
                else:
                    print(f"   ⚠️  Table is empty")
                    
            else:
                print(f"❌ {table}: Error {response.status_code}")
                print(f"   {response.text}")
                
        except Exception as e:
            print(f"❌ {table}: Error - {e}")
        
        print()

if __name__ == "__main__":
    check_table_data() 