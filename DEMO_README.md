# SymbioFlows Demo Setup Guide

This guide helps you prepare your SymbioFlows system for a video-ready demo that showcases the complete user journey:
1. Creating an account
2. Going through the AI onboarding process
3. Viewing generated material listings
4. Seeing potential matches

## Quick Start

Run the following command to set up the demo:

```bash
python setup_demo_system.py
```

This script will:
- Import sample company data (with waste streams)
- Enrich the data with quantitative values
- Generate material listings
- Create potential matches
- Set up the AI onboarding flow

## Demo Flow for Video Recording

### 1. Start Required Services

Make sure all services are running:

```bash
# Start backend server
cd backend
npm start

# In a separate terminal, start frontend
cd frontend
npm run dev
```

### 2. Record Your Demo

#### Account Creation
1. Open the application in your browser (typically http://localhost:5173)
2. Click "Sign Up" to create a new account
3. Fill in your details and complete the registration

#### AI Onboarding
1. After logging in, the AI onboarding process should start automatically
2. Answer the questions about your company, industry, and waste streams
3. Show how the AI asks follow-up questions based on your responses
4. Complete the onboarding process

#### Material Listings
1. Review the AI-generated material listings
2. Show how to edit an existing listing
3. Demonstrate adding a new material listing
4. Highlight different waste streams and requirements

#### Matches
1. Navigate to the matches section
2. Show potential matches between your company and others
3. Highlight match scores and details
4. Demonstrate how to initiate contact or express interest

## Using Your Own Company Data

If you want to use your own company data:

1. Format your data in JSON format following this structure:
```json
[
  {
    "name": "Your Company Name",
    "industry": "Your Industry",
    "location": "Your Location",
    "employee_count": 150,
    "products": ["Product A", "Product B"],
    "main_materials": ["Material A", "Material B"],
    "waste_streams": ["Waste Stream A", "Waste Stream B"],
    "process_description": "Description of your processes",
    "sustainability_goals": ["Goal A", "Goal B"],
    "current_waste_management": "Your current waste management approach",
    "email": "your-email@example.com"
  }
]
```

2. Save this file as `/workspace/data/fixed_realworlddata.json`

3. Run the setup script again:
```bash
python setup_demo_system.py
```

## Troubleshooting

### No Material Listings Appear
If material listings don't appear after the AI onboarding:

1. Check the enriched data directory:
```bash
ls -la /workspace/data/enriched/
```

2. Use the fallback method to import listings manually:
```bash
python backend/enrich_company_data.py
```

### Onboarding Process Not Starting
If the AI onboarding doesn't start automatically:

1. Try initializing it manually:
```bash
curl -X POST http://localhost:3000/api/adaptive-onboarding/initialize -H "Content-Type: application/json" -d '{"reset":true}'
```

### No Matches Generated
If no matches appear in the system:

1. Ensure you have multiple companies with complementary waste/requirement listings
2. Run the matching process manually:
```bash
python backend/improved_ai_matching_engine.py
```

## Support

If you encounter any issues, check the log files:
- `demo_setup.log` - Contains setup process logs
- Backend logs in the backend terminal
- Frontend logs in the browser console