import json
from collections import defaultdict

def is_more_comprehensive(entry1, entry2):
    """Return True if entry1 is more comprehensive than entry2."""
    def score(entry):
        s = 0
        for k, v in entry.items():
            if isinstance(v, list):
                s += len(v)
            elif v not in (None, '', [], {}):
                s += 1
        return s
    return score(entry1) > score(entry2)

with open('fixed_realworlddata.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

company_dict = defaultdict(list)
for company in data:
    name = company.get('name', '').strip()
    company_dict[name].append(company)

deduped = []
kept = {}

for name, entries in company_dict.items():
    if len(entries) == 1:
        deduped.append(entries[0])
        kept[name] = "unique"
    else:
        best = entries[0]
        for entry in entries[1:]:
            if is_more_comprehensive(entry, best):
                best = entry
        deduped.append(best)
        kept[name] = f"kept entry with {sum(1 for k,v in best.items() if v not in (None, '', [], {}))} non-empty fields"

with open('deduped_fixed_realworlddata.json', 'w', encoding='utf-8') as f:
    json.dump(deduped, f, indent=2, ensure_ascii=False)

print(f"Deduplicated list written to deduped_fixed_realworlddata.json")
print(f"Total companies after deduplication: {len(deduped)}")
print("Summary of kept entries for duplicates:")
for name, info in kept.items():
    if info != "unique":
        print(f"- {name}: {info}") 