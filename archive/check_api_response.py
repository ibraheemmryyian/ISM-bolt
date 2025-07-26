#!/usr/bin/env python3
"""
Check API Response Format
"""

import asyncio
import aiohttp
import json

async def check_api_response():
    url = "http://localhost:5001/api/companies"
    
    print("🔍 Checking API response format...")
    print(f"URL: {url}")
    print("=" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                print(f"Status: {response.status}")
                print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"Response type: {type(data)}")
                    print(f"Response length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
                    
                    if isinstance(data, list):
                        print(f"List length: {len(data)}")
                        if data:
                            print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                    elif isinstance(data, dict):
                        print(f"Dict keys: {list(data.keys())}")
                        if 'data' in data:
                            print(f"Data field type: {type(data['data'])}")
                            print(f"Data field length: {len(data['data']) if isinstance(data['data'], (list, dict)) else 'N/A'}")
                    
                    print("\n📋 Sample response structure:")
                    print(json.dumps(data[:2] if isinstance(data, list) else data, indent=2)[:500] + "...")
                else:
                    print(f"Error: {response.status}")
                    text = await response.text()
                    print(f"Response: {text}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_api_response()) 