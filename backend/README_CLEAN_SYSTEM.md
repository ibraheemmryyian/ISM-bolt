# Clean Material Data Generator

A production-ready, streamlined system for generating material listings and matches using only essential AI services.

## 🎯 Purpose

This system generates:
- **Material Listings**: AI-powered material listings for companies based on their profiles
- **Material Matches**: Intelligent matching between companies and materials for symbiotic opportunities

## 🏗️ Architecture

### Essential Services Only
The system uses only 6 core services (down from 20+):

1. **ListingInferenceService** - Generates material listings from company profiles
2. **AIListingsGenerator** - Creates AI-powered listings using transformer models
3. **RevolutionaryAIMatching** - Advanced AI matching between companies/materials
4. **GNNReasoningEngine** - Graph Neural Network reasoning for complex relationships
5. **MultiHopSymbiosisNetwork** - Multi-hop symbiosis matching across networks
6. **DynamicMaterialsIntegrationService** - Comprehensive materials analysis and integration

### Removed Services
All non-essential services have been removed:
- ❌ Pricing orchestrators
- ❌ Production orchestrators
- ❌ Monitoring dashboards
- ❌ Retraining pipelines
- ❌ Meta-learning orchestrators
- ❌ Opportunity engines
- ❌ Impact forecasting
- ❌ And 10+ other services

## 🚀 Quick Start

### 1. Test the System
```bash
cd backend
python test_clean_system.py
```

### 2. Run Material Generation
```bash
# Option A: Direct Python execution
python generate_supervised_materials_and_matches.py

# Option B: Automated batch script (Windows)
run_clean_material_generator.bat
```

### 3. Check Results
The system generates two CSV files:
- `material_listings.csv` - All generated material listings
- `material_matches.csv` - All generated material matches

## 📊 Input Data

The system expects a JSON file at the project root: `fixed_realworlddata.json`

Example structure:
```json
[
  {
    "id": "company_1",
    "name": "Manufacturing Co",
    "industry": "manufacturing",
    "location": "City",
    "description": "Company description"
  }
]
```

## 🔧 System Requirements

### Python Dependencies
```bash
pip install torch numpy aiohttp asyncio
```

### Optional Dependencies
- `sklearn` - For advanced ML features
- `transformers` - For transformer models
- `torch_geometric` - For GNN features

## 📁 File Structure

```
backend/
├── generate_supervised_materials_and_matches.py  # Main generator
├── test_clean_system.py                         # System test
├── run_clean_material_generator.bat             # Windows batch script
├── README_CLEAN_SYSTEM.md                       # This file
├── listing_inference_service.py                 # Listing generation
├── ai_listings_generator.py                     # AI listings
├── revolutionary_ai_matching.py                 # Advanced matching
├── gnn_reasoning_engine.py                      # GNN reasoning
├── multi_hop_symbiosis_network.py               # Multi-hop matching
└── dynamic_materials_integration_service.py     # Materials integration
```

## 🧪 Testing

### Run All Tests
```bash
python test_clean_system.py
```

### Test Individual Components
```python
# Test listing generation
from listing_inference_service import ListingInferenceService
service = ListingInferenceService()
result = await service.generate_listings_from_profile(company_data)

# Test matching
from revolutionary_ai_matching import RevolutionaryAIMatching
matcher = RevolutionaryAIMatching()
matches = await matcher.find_matches(material_data)
```

## 📈 Output Format

### Material Listings CSV
```csv
company_id,company_name,material_name,material_type,quantity,unit,description,quality_grade,potential_value,ai_generated,generated_at
company_1,Manufacturing Co,Steel Scrap,metal,100,tons,High quality steel scrap,A,5000,True,2024-01-01T12:00:00
```

### Material Matches CSV
```csv
source_company_id,source_material_name,target_company_id,target_company_name,target_material_name,match_score,match_type,potential_value,ai_generated,generated_at
company_1,Steel Scrap,match_company_1,Match Company 1,Compatible Steel Scrap,0.8,direct,1000,True,2024-01-01T12:00:00
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install torch numpy aiohttp asyncio
   ```

2. **Data File Not Found**
   ```bash
   # Ensure fixed_realworlddata.json exists in project root
   # Or run test_clean_system.py to create sample data
   ```

3. **Service Initialization Errors**
   ```bash
   # Check that all service files exist
   # Run test_clean_system.py to verify
   ```

### Debug Mode
```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎯 Performance

### Expected Output
- **115 companies** → **~500-1000 material listings**
- **Each listing** → **~5-10 material matches**
- **Total matches** → **~2500-10000 matches**

### Processing Time
- **Small dataset (10 companies)**: ~30 seconds
- **Medium dataset (50 companies)**: ~2-3 minutes
- **Large dataset (115 companies)**: ~5-10 minutes

## 🔒 Production Ready Features

- ✅ **Error handling** - Graceful failure with detailed logging
- ✅ **Resource cleanup** - Proper session and connection management
- ✅ **Data validation** - Input validation and sanitization
- ✅ **Deduplication** - Automatic removal of duplicate matches
- ✅ **Standardization** - Consistent output format
- ✅ **Async processing** - Efficient concurrent operations
- ✅ **Fallback mechanisms** - Service degradation handling

## 🚀 Next Steps

1. **Run the test**: `python test_clean_system.py`
2. **Generate data**: `python generate_supervised_materials_and_matches.py`
3. **Review results**: Check the generated CSV files
4. **Scale up**: Modify for larger datasets as needed

## 📞 Support

If you encounter issues:
1. Run `python test_clean_system.py` to identify problems
2. Check the error messages for specific issues
3. Ensure all dependencies are installed
4. Verify the data file exists and is valid

---

**🎉 The system is now clean, production-ready, and focused only on material generation, listings, and matches!** 