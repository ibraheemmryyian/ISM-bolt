# ISM AI Real Data Import Process

This directory contains scripts to import 50 real Gulf companies into your ISM AI platform and generate AI listings and matches.

## 🚀 Quick Start

### Option 1: Complete Startup (Recommended)
```cmd
scripts\start-everything.bat
```
This will:
- Start all services (Backend, AI Gateway, Frontend)
- Import all 50 companies
- Generate AI listings and matches
- Open the admin dashboard

### Option 2: Manual Import
If services are already running:
```cmd
scripts\import-real-data.bat
```

### Option 3: Test Import
For testing with just 5 companies:
```cmd
scripts\test-import.bat
```

## 📊 What Gets Imported

### Companies (50 Real Gulf Companies)
- **Borouge PLC** - Chemicals/Petrochemicals
- **SABIC** - Chemicals/Industrial Polymers
- **ADNOC** - Oil & Gas
- **ADNOC Distribution** - Oil & Gas/Logistics
- **Oman Cement Company** - Construction/Cement
- And 45 more companies...

Each company includes:
- Detailed profile information
- Industry classification
- Sustainability metrics
- Carbon footprint data
- Water usage statistics
- Matching preferences

### AI-Generated Listings
For each company, the system generates:
- **Waste Listings**: Materials the company wants to dispose of
- **Requirement Listings**: Materials the company needs
- **Sustainability Metrics**: Environmental impact data
- **Logistics Information**: Shipping and handling details

### AI Matches
The system creates symbiotic matches between companies:
- **Waste-to-Resource**: Company A's waste becomes Company B's input
- **Energy Sharing**: Excess energy from one company powers another
- **Water Reuse**: Wastewater treatment and reuse opportunities
- **Logistics Optimization**: Shared transportation and storage

## 🔧 Technical Details

### Files Created
- `backend/real_data_bulk_importer.py` - Main import script
- `scripts/import-real-data.bat` - CMD import script
- `scripts/test-import.bat` - Test import script
- `scripts/start-everything.bat` - Complete startup script

### Data Flow
1. **Load Data**: Reads from `data/50_real_gulf_companies_cleaned.json`
2. **Import Companies**: Creates company profiles in Supabase
3. **Generate Listings**: Uses AI to create waste/requirement listings
4. **Create Matches**: AI matching engine finds symbiotic relationships
5. **Store Results**: Everything saved to Supabase database

### API Endpoints Used
- `POST /api/companies` - Create company profiles
- `POST /api/ai/listings/generate` - Generate AI listings
- `POST /api/ai/matching/run` - Run AI matching

## 📈 Expected Results

After successful import, you should see:
- **50 companies** in the Companies tab
- **100+ AI listings** in the Materials tab
- **50+ matches** in the Matches tab
- **AI insights** in the AI Insights tab

## 🛠️ Troubleshooting

### Services Not Running
If you get errors about services not running:
1. Start backend: `cd backend && npm run dev`
2. Start AI gateway: `cd ai_service_flask && python ai_gateway.py`
3. Start frontend: `cd frontend && npm run dev`

### Import Errors
- Check that `data/50_real_gulf_companies_cleaned.json` exists
- Ensure all services are running on correct ports
- Check Supabase connection in environment files

### Database Issues
- Run database migrations if needed
- Check Supabase dashboard for connection issues
- Verify environment variables are set correctly

## 🌐 Access Points

After successful import:
- **Admin Dashboard**: http://localhost:5173/admin
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3000
- **AI Gateway**: http://localhost:5000

## 📋 Next Steps

1. **Explore the Admin Dashboard** - View all imported data
2. **Test AI Features** - Try the matching and listing generation
3. **Customize Data** - Modify company profiles or add new ones
4. **Scale Up** - Add more companies or regions
5. **Production Ready** - Deploy to production with real data

## 🎯 Success Metrics

A successful import should show:
- ✅ 50 companies imported
- ✅ 100+ AI listings generated
- ✅ 50+ matches created
- ✅ All data visible in admin dashboard
- ✅ AI features working correctly 