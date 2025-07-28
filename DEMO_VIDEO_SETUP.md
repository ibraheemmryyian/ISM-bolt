# 🎬 SymbioFlows Demo Setup for Video Capture

This guide will help you prepare your SymbioFlows system for video capture demo, including importing your real company data and intelligently estimating missing quantitative factors.

## 🚀 Quick Start (One Command Setup)

```bash
cd scripts
python quick_demo_setup.py
```

This will:
- ✅ Find your company data files automatically
- ✅ Import and enhance company data with intelligent estimations  
- ✅ Generate realistic material listings and matches
- ✅ Optimize the system for smooth video capture
- ✅ Create demo configuration files

## 📊 What the System Does with Your Data

### Your Real Data (What You Have):
- ✅ Company names
- ✅ Industries  
- ✅ Employee counts
- ✅ Waste streams (text descriptions)
- ✅ Locations

### What the AI Estimates (Missing Quantitative Factors):
- 🤖 **Waste Tonnage**: Based on employee count + industry benchmarks
- 🤖 **Production Volume**: Estimated from workforce size and industry type  
- 🤖 **Material Values**: Market-based pricing for waste streams
- 🤖 **Quantities**: Realistic monthly/annual quantities for each waste stream

### Industry Benchmarks Used:
- **Manufacturing**: 2.5 tonnes waste per employee per year
- **Textiles**: 1.8 tonnes per employee per year
- **Food & Beverage**: 3.2 tonnes per employee per year
- **Chemicals**: 4.1 tonnes per employee per year
- **Construction**: 5.8 tonnes per employee per year
- And more...

## 📁 Data File Format

Your company data should be in JSON format:

```json
[
  {
    "name": "Gulf Manufacturing Co.",
    "industry": "Manufacturing", 
    "location": "Dubai, UAE",
    "employee_count": 250,
    "waste_streams": ["metal shavings", "plastic offcuts", "packaging waste"],
    "description": "Optional company description"
  }
]
```

### Required Fields:
- `name`: Company name
- `industry`: Industry type
- `location`: Company location
- `employee_count`: Number of employees
- `waste_streams`: Array of waste stream names

### The System Will Estimate:
- Monthly tonnage for each waste stream
- Market value per tonne
- Production volumes
- Sustainability scores
- Material categories and recyclability

## 🎯 Demo Flow for Video Capture

### 1. Account Creation (30 seconds)
- Navigate to homepage
- Click "Get Started"
- Quick sign-up with realistic company info

### 2. AI Onboarding (90 seconds)  
- Industry selection (auto-filled for demo)
- Products description (pre-populated)
- Production volume (estimated from your data)
- Processes description (realistic examples)
- Watch AI generate portfolio in real-time

### 3. Material Listings (60 seconds)
- View intelligent material listings
- Show estimated quantities and values
- Demonstrate waste streams from your real data
- Browse different material categories

### 4. Matching System (90 seconds)
- View AI-generated matches
- Show potential savings calculations
- Demonstrate match scores and reasoning
- Browse partner companies

## 🛠️ Manual Setup (Advanced)

If you prefer manual control:

### Step 1: Prepare Data
```bash
# Place your company data file in data/ directory
cp your_company_data.json data/company_data.json
```

### Step 2: Run Demo Import Service
```bash
cd backend
python demo_data_import_service.py
```

### Step 3: Setup Demo Environment
```bash
cd scripts  
python setup_demo_environment.py --data-file ../data/company_data.json
```

### Step 4: Start Frontend
```bash
cd frontend
npm run dev
```

## 📈 What Gets Generated

From your real data, the system creates:

### Enhanced Company Profiles:
- Original data + estimated quantitative factors
- Sustainability scores (65-95 range)
- Industry classifications and benchmarks

### Material Listings:
- **Waste Materials**: From your waste streams with estimated quantities
- **Requirements**: What companies typically need in your industry
- **Pricing**: Market-based pricing per tonne
- **Availability**: Continuous/seasonal based on industry

### Intelligent Matches:
- Cross-industry material exchanges
- Potential savings calculations  
- Carbon reduction estimates
- Match confidence scores (75-95%)

## 🎬 Video Capture Tips

### Pre-Recording Checklist:
- [ ] Run demo setup script
- [ ] Clear browser cache
- [ ] Test account creation flow
- [ ] Verify material listings appear
- [ ] Confirm matches are generated
- [ ] Check onboarding prefill works

### Demo Script Timing:
- **Account Creation**: 30 seconds
- **AI Onboarding**: 90 seconds  
- **Portfolio Review**: 60 seconds
- **Marketplace Browse**: 90 seconds
- **Total Demo Time**: 4-5 minutes

### Key Points to Highlight:
1. **Real Data**: "This is based on actual company data"
2. **AI Intelligence**: "Watch the AI estimate tonnage from employee count"
3. **Industry Benchmarks**: "Using real industry waste generation rates"
4. **Instant Matching**: "AI finds matches across the Gulf region"
5. **Quantified Impact**: "Specific savings and carbon reduction"

## 🔧 Troubleshooting

### Demo Data Not Loading?
```bash
# Check if demo files were created
ls frontend/src/data/demo-marketplace.json
ls frontend/src/config/demo-config.json
```

### No Company Data File?
The system will create sample Gulf region companies automatically.

### Frontend Not Connecting?
Ensure you're using the development server:
```bash
cd frontend
npm install
npm run dev
```

### API Errors?
The demo works offline - no backend services required for basic demo.

## 💡 Pro Tips

1. **Use Real Company Names**: Makes the demo more authentic
2. **Diverse Industries**: Mix manufacturing, textiles, food, etc.
3. **Realistic Employee Counts**: 50-500 employees per company
4. **Specific Waste Streams**: "metal shavings" vs "waste materials"
5. **Gulf Region Focus**: Use UAE, Saudi, Qatar locations

## 📞 Need Help?

If you encounter issues:
1. Check the generated `demo_setup_report.json`
2. Review console logs for errors
3. Verify your data file format
4. Try the sample data option first

Your demo environment is now ready for professional video capture! 🎬✨