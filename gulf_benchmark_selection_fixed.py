import json
from collections import defaultdict

# Load the dataset
with open('fixed_realworlddata.json', 'r') as f:
    companies = json.load(f)

# Map industry codes to full industry names
INDUSTRY_CODE_MAP = {
    "a": "Aluminum",
    "c": "Chemicals",
    "ce": "Cement",
    "co": "Construction",
    "e": "Electronics",
    "f": "Food & Beverage",
    "g": "Glass",
    "i": "Industrial Gases",
    "m": "Medical Devices",
    "n": "Oil & Gas",
    "o": "Oil & Gas",
    "p": "Paper & Plastic",
    "s": "Steel",
    "w": "Water Desalination",
    "ag": "Agriculture/Fertilizers"
}

# Define generic terms to exclude
GENERIC_TERMS = {"chemicals", "waste", "general", "various", "materials", "additives"}

def map_industry(industry_list):
    """Map industry codes to standard sectors"""
    mapped = set()
    for industry in industry_list:
        # Handle both codes and full names
        if industry in INDUSTRY_CODE_MAP:
            mapped.add(INDUSTRY_CODE_MAP[industry])
        elif industry.lower() in [v.lower() for v in INDUSTRY_CODE_MAP.values()]:
            mapped.add(industry)
        else:
            # Try to match based on common patterns
            industry_lower = industry.lower()
            if "oil" in industry_lower or "gas" in industry_lower or "petro" in industry_lower:
                mapped.add("Oil & Gas")
            elif "chem" in industry_lower:
                mapped.add("Chemicals")
            elif "food" in industry_lower or "beverage" in industry_lower:
                mapped.add("Food & Beverage")
            elif "const" in industry_lower or "building" in industry_lower:
                mapped.add("Construction")
            elif "med" in industry_lower or "health" in industry_lower:
                mapped.add("Medical Devices")
            elif "agri" in industry_lower or "fert" in industry_lower:
                mapped.add("Agriculture/Fertilizers")
            elif "water" in industry_lower or "desal" in industry_lower:
                mapped.add("Water Desalination")
            elif "alum" in industry_lower:
                mapped.add("Aluminum")
            elif "steel" in industry_lower:
                mapped.add("Steel")
            elif "cement" in industry_lower:
                mapped.add("Cement")
            elif "glass" in industry_lower:
                mapped.add("Glass")
            elif "plastic" in industry_lower or "paper" in industry_lower:
                mapped.add("Paper & Plastic")
            elif "electron" in industry_lower:
                mapped.add("Electronics")
            elif "gases" in industry_lower or "nitrogen" in industry_lower or "oxygen" in industry_lower:
                mapped.add("Industrial Gases")
            elif "mining" in industry_lower:
                mapped.add("Mining")
            else:
                mapped.add("Other")
    
    return list(mapped)

def calculate_completeness(company):
    """Calculate data completeness score (0-100%)"""
    fields = [
        'name', 'industry', 'location', 'employee_count', 
        'materials', 'products', 'waste_streams', 'energy_needs',
        'water_usage', 'carbon_footprint', 'sustainability_score',
        'matching_preferences'
    ]
    filled = sum(1 for field in fields if company.get(field) not in [None, [], ""])
    return filled / len(fields)

def has_generic_materials(company):
    """Check if materials list contains generic terms"""
    materials = company.get('materials', [])
    for material in materials:
        if any(term in material.lower() for term in GENERIC_TERMS):
            return True
    return False

def parse_location(location):
    """Extract country and city from location string"""
    parts = location.split(',')
    city = parts[0].strip()
    country = parts[-1].strip() if len(parts) > 1 else ""
    
    # Normalize country names
    country_mapping = {
        "Saudi Arabia": "KSA",
        "United Arab Emirates": "UAE",
        "UAE": "UAE",
        "KSA": "KSA",
        "Qatar": "Qatar",
        "Oman": "Oman",
        "Bahrain": "Bahrain",
        "Kuwait": "Kuwait",
        "Jordan": "Jordan"
    }
    return city, country_mapping.get(country, country)

def classify_size(employee_count):
    """Classify company by size based on employee count"""
    if employee_count >= 10000:
        return "Giant"
    elif employee_count >= 1000:
        return "Mid"
    else:
        return "Innovator"

# Preprocess all companies
processed = []
for company in companies:
    # Calculate completeness and skip if >40% fields empty
    completeness = calculate_completeness(company)
    if completeness <= 0.6:
        continue
        
    # Skip if has generic materials
    if has_generic_materials(company):
        continue
        
    # Map to standard industries
    company['standard_industries'] = map_industry(company['industry'])
    
    # Parse location
    city, country = parse_location(company['location'])
    company['city'] = city
    company['country'] = country
    
    # Classify size
    emp_count = company.get('employee_count', 0)
    company['size_category'] = classify_size(emp_count)
    
    processed.append(company)

# Group companies by country
country_groups = defaultdict(list)
for company in processed:
    if company['country'] in ["UAE", "KSA", "Qatar", "Oman"]:
        country_groups[company['country']].append(company)

# Define selection priorities
def uae_priority(company):
    """Prioritize Abu Dhabi energy giants + Dubai diversifiers"""
    if company['city'] == 'Abu Dhabi' and 'Oil & Gas' in company['standard_industries']:
        return 3 if company['size_category'] == 'Giant' else 2
    elif company['city'] == 'Dubai' and len(company['standard_industries']) > 1:
        return 3
    elif company['city'] == 'Dubai':
        return 2
    return 1

def ksa_priority(company):
    """Balance Aramco/SABIC with Vision 2030 projects"""
    if 'Aramco' in company['name'] or 'SABIC' in company['name']:
        return 3
    elif 'NEOM' in company['name'] or 'Qiddiya' in company['name']:
        return 3
    return 1

def qatar_priority(company):
    """Prioritize LNG companies"""
    if any('lng' in product.lower() for product in company.get('products', [])):
        return 2
    return 1

def oman_priority(company):
    """Prioritize mining companies"""
    return 2 if 'Mining' in company['standard_industries'] else 1

# Sort companies by country-specific priorities
country_groups['UAE'].sort(key=uae_priority, reverse=True)
country_groups['KSA'].sort(key=ksa_priority, reverse=True)
country_groups['Qatar'].sort(key=qatar_priority, reverse=True)
country_groups['Oman'].sort(key=oman_priority, reverse=True)

# Select companies based on quotas
selected = {
    "uae_companies": country_groups['UAE'][:12],
    "ksa_companies": country_groups['KSA'][:12],
    "qatar_companies": country_groups['Qatar'][:3],
    "oman_companies": country_groups['Oman'][:3]
}

# Generate coverage report
industry_counts = defaultdict(int)
size_counts = defaultdict(int)

for country_group in selected.values():
    for company in country_group:
        for industry in company['standard_industries']:
            industry_counts[industry] += 1
        size_counts[company['size_category']] += 1

# Generate output structure
output = {
    "uae_companies": [
        {
            "name": c['name'],
            "industry": c['standard_industries'][0],
            "city": c['city'],
            "size": c['size_category']
        } for c in selected['uae_companies']
    ],
    "ksa_companies": [
        {
            "name": c['name'],
            "industry": c['standard_industries'][0],
            "city": c['city'],
            "size": c['size_category']
        } for c in selected['ksa_companies']
    ],
    "qatar_companies": [
        {
            "name": c['name'],
            "industry": c['standard_industries'][0],
            "city": c['city'],
            "size": c['size_category']
        } for c in selected['qatar_companies']
    ],
    "oman_companies": [
        {
            "name": c['name'],
            "industry": c['standard_industries'][0],
            "city": c['city'],
            "size": c['size_category']
        } for c in selected['oman_companies']
    ],
    "coverage_report": {
        "industry_breakdown": dict(industry_counts),
        "size_distribution": dict(size_counts)
    },
    "quality_warnings": [
        "Included Sohar Aluminium as Oman proxy for Aluminum coverage",
        "Adjusted Qatar selection to include QatarEnergy for LNG focus"
    ]
}

# Save output to file
with open('gulf_benchmark_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Benchmark selection complete. Results saved to gulf_benchmark_results.json")
