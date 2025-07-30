#!/usr/bin/env python3
"""
Script to directly update the database with material listings
"""

import os
import json
import time
import sys
import urllib.request
import urllib.parse
import urllib.error

# Get Supabase credentials
SUPABASE_URL = "https://your-supabase-url.supabase.co"
SUPABASE_ANON_KEY = "your-supabase-anon-key"

print("🚀 SymbioFlows Material Listings Updater")
print("=" * 50)

# Get user ID
user_id = input("Enter your user ID: ")

if not user_id:
    print("❌ No user ID provided")
    sys.exit(1)

# Create material listings
material_listings = [
    {
        'name': 'Chemical Waste',
        'type': 'waste',
        'description': "High-quality chemical waste from your production processes. Can be recycled or repurposed for various applications.",
        'quantity': 500,
        'unit': 'tons',
        'frequency': 'monthly',
        'specifications': 'pH 6-8, low heavy metal content',
        'sustainability_impact': 'Reduces landfill waste by 30%',
        'market_value': '$200-300 per ton',
        'logistics_notes': 'Available for pickup at our facility',
        'user_id': user_id
    },
    {
        'name': 'Organic Waste',
        'type': 'waste',
        'description': "Organic waste byproducts from your manufacturing. Suitable for composting or biogas production.",
        'quantity': 300,
        'unit': 'tons',
        'frequency': 'monthly',
        'specifications': 'High carbon content, biodegradable',
        'sustainability_impact': 'Can be converted to renewable energy',
        'market_value': '$150-200 per ton',
        'logistics_notes': 'Available in bulk quantities',
        'user_id': user_id
    },
    {
        'name': 'Aqueous Waste',
        'type': 'waste',
        'description': "Aqueous waste streams from your processes. Contains recoverable minerals and compounds.",
        'quantity': 1000,
        'unit': 'liters',
        'frequency': 'weekly',
        'specifications': 'Low toxicity, treatable',
        'sustainability_impact': 'Reduces water pollution',
        'market_value': '$0.50-1.00 per liter',
        'logistics_notes': 'Available in IBC containers',
        'user_id': user_id
    },
    {
        'name': 'Raw Chemicals',
        'type': 'requirement',
        'description': "Seeking high-quality raw chemicals for your production processes.",
        'quantity': 1000,
        'unit': 'kg',
        'frequency': 'monthly',
        'specifications': 'USP grade, 99% purity',
        'sustainability_impact': 'Prefer suppliers with sustainable practices',
        'market_value': '$5-10 per kg',
        'logistics_notes': 'Need delivery to our facility',
        'user_id': user_id
    },
    {
        'name': 'Catalysts',
        'type': 'requirement',
        'description': "Looking for industrial catalysts for your chemical processes.",
        'quantity': 200,
        'unit': 'kg',
        'frequency': 'quarterly',
        'specifications': 'High activity, low metal content',
        'sustainability_impact': 'Reduces energy consumption in processes',
        'market_value': '$20-30 per kg',
        'logistics_notes': 'Need specialized handling',
        'user_id': user_id
    },
    {
        'name': 'Solvents',
        'type': 'requirement',
        'description': "Require industrial solvents for your manufacturing.",
        'quantity': 2000,
        'unit': 'liters',
        'frequency': 'monthly',
        'specifications': 'High purity, low water content',
        'sustainability_impact': 'Prefer recycled or bio-based options',
        'market_value': '$2-5 per liter',
        'logistics_notes': 'Bulk delivery preferred',
        'user_id': user_id
    }
]

# Save material listings to file
with open('material_listings.json', 'w') as f:
    json.dump(material_listings, f, indent=2)

print(f"✅ Created {len(material_listings)} material listings")
print(f"✅ Saved material listings to material_listings.json")
print("\n🎉 Material listings have been created!")
print("🔄 To add these to your account:")
print("1. Go to your Supabase dashboard")
print("2. Open the 'materials' table")
print("3. Click 'Import' and select the material_listings.json file")
print("4. Refresh your SymbioFlows dashboard to see the new listings")
print("=" * 50)