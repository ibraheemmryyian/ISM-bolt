# AI-Driven Matchmaking & Portfolio Generation Pipeline

## 🎯 Master Directive Implementation

This implementation provides a comprehensive two-phase AI pipeline that restores and significantly upgrades the platform's intelligence using the DeepSeek API.

### The Problem
The platform's core AI functionality—proactive matchmaking based on a company's profile—was broken. This implementation fixes it with a sophisticated two-stage AI pipeline.

### The Solution
1. **Phase 1**: DeepSeek API generates detailed portfolio for new companies
2. **Phase 2**: DeepSeek API finds high-potential matches across the network
3. **Orchestration**: Node.js backend coordinates both phases seamlessly

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Company       │    │   Phase 1:       │    │   Database      │
│   Onboarding    │───▶│   Portfolio      │───▶│   (Materials &  │
│                 │    │   Generation     │    │    Requirements) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Database      │    │   Phase 2:       │    │   Matches       │
│   (Materials)   │───▶│   Matchmaking    │───▶│   Table         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Phase 1: Deep Portfolio Generation

### Target Service: `listing_inference_service.py`

**Purpose**: Automatically understand a new company's resources using DeepSeek API

**Process**:
1. Receive company profile
2. Call DeepSeek API with exact prompt structure
3. Parse JSON response
4. Save to database with company ID

### DeepSeek Prompt Structure (Phase 1)
```json
{
  "model": "deepseek-coder",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert in industrial symbiosis and circular economies. Your task is to analyze a company's profile and predict its material 'outputs' (waste streams, byproducts) and 'inputs' (operational needs). Provide a detailed response as a JSON object with two main keys: 'predicted_outputs' and 'predicted_inputs'. Each item in these lists must include the following fields: 'name' (the material), 'category' (e.g., 'textile', 'chemical', 'metal'), 'description' (brief explanation), 'quantity' (an estimated amount, e.g., '10 tons'), 'frequency' (e.g., 'weekly', 'monthly', 'batch'), and 'notes' (any special considerations, e.g., 'Requires cold storage')."
    },
    {
      "role": "user",
      "content": "Analyze the following company profile. Company Profile -- Industry: 'Commercial Furniture Manufacturing', Primary Products: 'Office chairs, wooden desks, conference tables', Description: 'A large-scale factory that produces high-end office furniture primarily from oak, steel, and synthetic fabrics.'"
    }
  ],
  "response_format": { "type": "json_object" }
}
```

## 🤝 Phase 2: AI-Powered Matchmaking

### Target Service: `ai_matchmaking_service.py`

**Purpose**: Find partner companies for generated materials using DeepSeek API

**Process**:
1. Receive company_id and material data
2. Call DeepSeek API for partner recommendations
3. Find matching companies in database
4. Create match records

### DeepSeek Prompt Structure (Phase 2)
```json
{
  "model": "deepseek-coder",
  "messages": [
    {
      "role": "system",
      "content": "You are an AI-powered industrial matchmaking expert. Your task is to analyze a specific industrial material and recommend the top 3 types of companies that would be ideal symbiotic partners for it. Provide the response as a JSON object with a single key: 'recommendations'. This key should contain a list of objects, with each object having two fields: 'company_type' (the type of company, e.g., 'Cement Manufacturer') and 'match_reason' (a concise explanation of why it's a good match)."
    },
    {
      "role": "user",
      "content": "Find the best company matches for the following material. Material -- Type: 'Output/Waste', Name: 'Coal Fly Ash', Description: 'A fine powder byproduct from burning pulverized coal in electric generation power plants.'"
    }
  ],
  "response_format": { "type": "json_object" }
}
```

## 🔄 Orchestration

### Node.js Backend (`app.js`)

**Endpoints**:
- `/api/ai-portfolio-generation` - Phase 1 only
- `/api/ai-matchmaking` - Phase 2 only  
- `/api/ai-pipeline` - Complete orchestrated pipeline

**Flow**:
1. Company onboards
2. Backend calls Phase 1 service
3. Upon successful completion, triggers Phase 2 service
4. Results saved to database

## 🎯 Definition of Done

**Example Scenario**: A new "Brewery" signs up

**Expected Results**:
1. ✅ Brewery's portfolio populated with "Spent Grain"
2. ✅ New entries in matches table
3. ✅ Links to existing "Cattle Farm" companies
4. ✅ AI recommended the pairing

## 🚀 Getting Started

### Prerequisites
- Node.js backend running on port 3001
- Python 3.7+ with required packages
- DeepSeek API key configured
- Database with updated schema

### Installation

1. **Update Database Schema**
```sql
-- Run the updated schema.sql to add new fields
-- quantity, frequency, notes for materials and requirements
-- match_reason for matches table
```

2. **Install Python Dependencies**
```bash
pip install requests json logging
```

3. **Configure DeepSeek API**
```python
# Already configured in services
DEEPSEEK_API_KEY = 'sk-7ce79f30332d45d5b3acb8968b052132'
```

### Testing

1. **Start Backend Server**
```bash
cd backend
npm start
```

2. **Run Test Pipeline**
```bash
python test_ai_pipeline.py
```

3. **Test Individual Phases**
```bash
python test_ai_pipeline.py phase1  # Portfolio generation only
python test_ai_pipeline.py phase2  # Matchmaking only
python test_ai_pipeline.py full    # Complete pipeline
```

## 📊 API Endpoints

### Phase 1: Portfolio Generation
```http
POST /api/ai-portfolio-generation
Content-Type: application/json

{
  "id": "company_001",
  "name": "Craft Beer Brewery",
  "industry": "Brewery",
  "primary_products": "Craft beer, ale, lager",
  "description": "A large-scale brewery that produces craft beer from grains, hops, and yeast"
}
```

**Response**:
```json
{
  "success": true,
  "phase": 1,
  "portfolio": {
    "materials": [...],
    "requirements": [...],
    "total_materials": 3,
    "total_requirements": 2
  },
  "message": "Portfolio generation completed successfully"
}
```

### Phase 2: Matchmaking
```http
POST /api/ai-matchmaking
Content-Type: application/json

{
  "company_id": "company_001",
  "material_data": {
    "name": "Spent Grain",
    "description": "Wet grain byproduct from beer brewing process",
    "category": "agricultural_waste",
    "quantity": "5 tons",
    "frequency": "daily",
    "notes": "High protein content, good for animal feed"
  }
}
```

**Response**:
```json
{
  "success": true,
  "phase": 2,
  "partner_companies": [...],
  "created_matches": [...],
  "message": "Matchmaking completed successfully"
}
```

### Complete Pipeline
```http
POST /api/ai-pipeline
Content-Type: application/json

{
  "id": "company_001",
  "name": "Craft Beer Brewery",
  "industry": "Brewery",
  "primary_products": "Craft beer, ale, lager",
  "description": "A large-scale brewery that produces craft beer from grains, hops, and yeast"
}
```

**Response**:
```json
{
  "success": true,
  "phase": "complete",
  "portfolio": {...},
  "matches_created": 6,
  "total_matches": [...],
  "message": "Two-phase AI pipeline completed successfully"
}
```

## 🔧 Configuration

### Environment Variables
```bash
DEEPSEEK_API_KEY=sk-7ce79f30332d45d5b3acb8968b052132
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-coder
```

### Database Schema Updates
The following fields were added to support the new AI pipeline:

**Materials Table**:
- `quantity` - Estimated amount (e.g., "10 tons")
- `frequency` - Frequency (e.g., "weekly", "monthly", "batch")
- `notes` - Special considerations

**Requirements Table**:
- `quantity` - Estimated amount needed
- `frequency` - Frequency needed
- `notes` - Special considerations

**Matches Table**:
- `match_reason` - AI-generated explanation for the match

## 🧪 Testing Examples

### Brewery Example (Master Directive)
```python
brewery_profile = {
    "id": "brewery_001",
    "name": "Craft Beer Brewery",
    "industry": "Brewery",
    "primary_products": "Craft beer, ale, lager",
    "description": "A large-scale brewery that produces craft beer from grains, hops, and yeast"
}
```

**Expected Results**:
- Portfolio: Spent Grain, Yeast Waste, etc.
- Matches: Cattle Farms, Compost Facilities, etc.
- Database: New entries in materials, requirements, and matches tables

### Furniture Manufacturer Example
```python
furniture_profile = {
    "id": "furniture_001",
    "name": "Office Furniture Co",
    "industry": "Commercial Furniture Manufacturing",
    "primary_products": "Office chairs, wooden desks, conference tables",
    "description": "A large-scale factory that produces high-end office furniture primarily from oak, steel, and synthetic fabrics"
}
```

**Expected Results**:
- Portfolio: Wood Scraps, Metal Waste, Fabric Offcuts
- Matches: Paper Mills, Metal Recyclers, Textile Manufacturers
- Database: Complete portfolio and match records

## 🚨 Error Handling

### Common Issues
1. **DeepSeek API Errors**: Check API key and rate limits
2. **Database Errors**: Verify schema updates and connections
3. **Python Script Errors**: Check dependencies and file paths

### Debugging
```bash
# Check backend logs
npm start

# Test individual services
python listing_inference_service.py
python ai_matchmaking_service.py

# Run test pipeline
python test_ai_pipeline.py
```

## 📈 Performance Considerations

### Optimization
- Batch processing for multiple materials
- Caching of AI responses
- Database indexing for faster queries
- Rate limiting for API calls

### Monitoring
- API response times
- Database query performance
- Error rates and types
- Success metrics

## 🔮 Future Enhancements

### Planned Features
1. **Multi-language Support**: Expand beyond English
2. **Industry Specialization**: Custom prompts per industry
3. **Real-time Updates**: Live match notifications
4. **Advanced Analytics**: Match success tracking
5. **Integration APIs**: Connect with external systems

### Scalability
- Microservices architecture
- Load balancing
- Database sharding
- CDN for static assets

## 📞 Support

For questions or issues:
1. Check the test scripts for examples
2. Review the API documentation
3. Examine the error logs
4. Test with the provided examples

---

**Status**: ✅ Implementation Complete
**Version**: 1.0.0
**Last Updated**: January 2025 