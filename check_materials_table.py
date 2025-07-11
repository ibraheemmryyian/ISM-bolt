import os
import httpx
from supabase import create_client, Client

# Initialize Supabase client
url = "https://jifkiwbxnttrkdrdcose.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE5NzI5NzQsImV4cCI6MjA0NzU0ODk3NH0.Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8Ej8"
supabase: Client = create_client(url, key)

def check_materials_table():
    """Check the actual structure of the materials table"""
    print("🔍 Checking materials table structure...")
    
    try:
        # Try to get a single row to see what columns exist
        result = supabase.table('materials').select('*').limit(1).execute()
        print(f"✅ Materials table exists and is accessible")
        print(f"📊 Sample data: {result.data}")
        
        if result.data:
            # Show the column names from the first row
            columns = list(result.data[0].keys())
            print(f"📋 Available columns: {columns}")
        else:
            print("⚠️ Table exists but is empty")
            
    except Exception as e:
        print(f"❌ Error accessing materials table: {str(e)}")
        
        # Try to check if the table exists at all
        try:
            # Try a different approach - get table info
            result = supabase.table('materials').select('id').limit(1).execute()
            print("✅ Table exists but may have different structure")
        except Exception as e2:
            print(f"❌ Table may not exist: {str(e2)}")

def check_requirements_table():
    """Check the actual structure of the requirements table"""
    print("\n🔍 Checking requirements table structure...")
    
    try:
        result = supabase.table('requirements').select('*').limit(1).execute()
        print(f"✅ Requirements table exists and is accessible")
        print(f"📊 Sample data: {result.data}")
        
        if result.data:
            columns = list(result.data[0].keys())
            print(f"📋 Available columns: {columns}")
        else:
            print("⚠️ Table exists but is empty")
            
    except Exception as e:
        print(f"❌ Error accessing requirements table: {str(e)}")

if __name__ == "__main__":
    check_materials_table()
    check_requirements_table() 