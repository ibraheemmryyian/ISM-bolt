import json
from collections import Counter

with open('fixed_realworlddata.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

names = [company.get('name', '').strip() for company in data]
name_counts = Counter(names)

print(f"Total companies: {len(names)}")
print(f"Unique company names: {len(set(names))}")

duplicates = [name for name, count in name_counts.items() if count > 1]
if duplicates:
    print("Duplicate company names:")
    for name in duplicates:
        print(f"- {name} (appears {name_counts[name]} times)")
else:
    print("No duplicate company names found.") 