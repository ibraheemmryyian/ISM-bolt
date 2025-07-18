import json
import re
from collections import OrderedDict

INPUT_FILE = 'realworlddata.json'
OUTPUT_FILE = 'realworlddata_deduped.json'

def tolerant_json_load(filename):
    """Load JSON file, fixing common issues like trailing commas and missing brackets."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    # Remove trailing commas before closing brackets
    content = re.sub(r',\s*([}\]])', r'\1', content)
    # Remove duplicate commas
    content = re.sub(r',\s*,', ',', content)
    # Fix unclosed arrays/objects (best effort)
    if not content.strip().endswith(']'):
        content += '\n]'
    try:
        data = json.loads(content)
    except Exception as e:
        # Try to fix common issues
        content = content.replace('\n', '').replace('\r', '')
        data = json.loads(content)
    return data

def deduplicate_companies(companies):
    seen = OrderedDict()
    for company in companies:
        name = company.get('name')
        # Use full JSON string as a hash for exact duplicates
        company_str = json.dumps(company, sort_keys=True)
        if name in seen:
            # If exact duplicate, skip
            if seen[name][-1] == company_str:
                continue
            # If not exact, replace with the newer version
            seen[name] = (company, company_str)
        else:
            seen[name] = (company, company_str)
    # Only keep the company dicts
    return [v[0] for v in seen.values()]

def main():
    companies = tolerant_json_load(INPUT_FILE)
    cleaned = deduplicate_companies(companies)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"Deduplication complete. {len(companies) - len(cleaned)} duplicates removed. Cleaned data written to {OUTPUT_FILE}.")

if __name__ == '__main__':
    main() 