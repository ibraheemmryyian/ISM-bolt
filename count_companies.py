import json

with open('fixed_realworlddata.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f"Number of companies: {len(data)}")