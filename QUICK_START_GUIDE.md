# 🚀 SymbioFlows Production Demo - Quick Start Guide

## ⚡ Get Started in 5 Minutes

### 1. Prerequisites Check
```bash
# Check Python (3.8+)
python --version

# Check Node.js (18+)
node --version

# Check if you have the required API keys
echo $SUPABASE_URL
echo $DEEPSEEK_API_KEY
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your actual values
# Required: SUPABASE_URL, SUPABASE_ANON_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY
```

### 3. Start the System

#### Option A: Windows with Debugging (Recommended)
```cmd
# Double-click or run:
run_production_demo.bat
```

This script includes:
- ✅ Environment checks (Python, Node.js, npm)
- ✅ Test script execution verification
- ✅ Detailed error messages and troubleshooting
- ✅ Real-time progress updates

#### Option B: Python Script with Enhanced Logging
```bash
# Run the production orchestrator with enhanced debugging
python start_production_demo.py
```

#### Option C: Test Environment First
```bash
# If you're having issues, test your environment first
python test_script_execution.py
```

#### Option C: Manual Start
```bash
# Terminal 1: Backend
cd backend && npm install && npm start

# Terminal 2: AI Services
cd ai_service_flask && pip install -r requirements.txt && python ai_gateway.py

# Terminal 3: Frontend
cd frontend && npm install && npm run dev
```

### 4. Access the System
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3000
- **AI Services**: http://localhost:5000
- **API Docs**: http://localhost:3000/api-docs

### 5. Test the System
```bash
# Run comprehensive tests
python test_production_system.py
```

## 🎭 Demo User Flow

### Step 1: User Signup
1. Visit http://localhost:5173
2. Click "Get Started"
3. Fill out company information:
   - **Company Name**: EcoTech Manufacturing
   - **Industry**: Electronics Manufacturing
   - **Waste Streams**: Electronic waste, plastic components
   - **Sustainability Goals**: Zero waste by 2030

### Step 2: AI Onboarding
1. Complete the AI questionnaire
2. Answer questions about your company
3. Watch AI generate personalized insights

### Step 3: Material Listings
1. AI automatically generates material listings
2. Review and customize the listings
3. See AI-powered descriptions and pricing

### Step 4: Matchmaking
1. AI finds potential partners
2. View match recommendations
3. Explore symbiosis opportunities

## 🔧 Troubleshooting

### Script Shows "Nothing Happened"
If the script appears to do nothing or shows no output:

1. **Use the debugging batch file**:
   ```cmd
   run_production_demo.bat
   ```

2. **Test your environment first**:
   ```cmd
   python test_script_execution.py
   ```

3. **Check Python execution**:
   ```cmd
   python --version
   python -c "print('Python is working')"
   ```

4. **Run with explicit Python**:
   ```cmd
   python start_production_demo.py
   ```

5. **Check log files**:
   - `production_demo.log` - Main execution logs
   - `test_execution.log` - Environment test logs

### Services Not Starting
```bash
# Check if ports are in use
netstat -tulpn | grep :3000
netstat -tulpn | grep :5000
netstat -tulpn | grep :5173

# Kill processes if needed
taskkill /f /im node.exe
taskkill /f /im python.exe
```

### Missing Dependencies
```bash
# Install Python dependencies
pip install -r ai_service_flask/requirements.txt

# Install Node.js dependencies
cd backend && npm install
cd frontend && npm install
```

### Environment Variables
```bash
# Check if variables are set
echo $SUPABASE_URL
echo $DEEPSEEK_API_KEY

# Set them if missing
export SUPABASE_URL="your-supabase-url"
export DEEPSEEK_API_KEY="your-deepseek-key"
```

## 📊 System Status

### Health Checks
- **Backend**: http://localhost:3000/api/health
- **AI Gateway**: http://localhost:5000/health
- **Frontend**: http://localhost:5173

### Monitoring
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001

## 🎯 Key Features Demonstrated

### AI-Powered Features
- ✅ **Intelligent Onboarding**: AI analyzes company profile
- ✅ **Material Generation**: AI creates realistic material listings
- ✅ **Smart Matchmaking**: AI finds optimal partnerships
- ✅ **Predictive Analytics**: AI forecasts opportunities

### Production Features
- ✅ **Microservices Architecture**: 8 AI services running
- ✅ **Real-time Processing**: Live AI inference
- ✅ **Scalable Design**: Docker-ready deployment
- ✅ **Monitoring**: Health checks and metrics

### User Experience
- ✅ **Seamless Flow**: Signup → Onboarding → Listings → Matches
- ✅ **Modern UI**: React + Tailwind CSS
- ✅ **Responsive Design**: Works on all devices
- ✅ **Real-time Updates**: Live data synchronization

## 🚀 Next Steps

### For Development
```bash
# Run in development mode
export NODE_ENV=development
export LOG_LEVEL=DEBUG

# Start with debug flags
python start_production_demo.py --debug
```

### For Production
```bash
# Use Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Scale services
docker-compose -f docker-compose.production.yml up -d --scale ai-gateway=3
```

### For Testing
```bash
# Run all tests
python test_production_system.py

# Run specific tests
python -m pytest tests/ -v
```

## 📞 Support

### Quick Help
- **Documentation**: `/docs/` directory
- **API Reference**: http://localhost:3000/api-docs
- **System Status**: Check health endpoints above

### Common Issues
1. **Port conflicts**: Kill existing processes
2. **Missing API keys**: Set environment variables
3. **Dependency issues**: Reinstall with `pip install -r requirements.txt`
4. **Database connection**: Check Supabase configuration

---

**🎉 You're ready to experience the future of circular economy AI!**

Visit http://localhost:5173 to start your journey. 