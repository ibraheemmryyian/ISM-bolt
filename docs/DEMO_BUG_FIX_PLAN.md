# Demo Bug Fix Plan - Critical Issues

## 🚨 CRITICAL ISSUES BLOCKING DEMO

### 1. PyTorch Installation (CRITICAL)
**Issue**: PyTorch installation is corrupted/broken
**Impact**: All ML services fail to start
**Fix Priority**: URGENT

```bash
# Fix PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Missing Dependencies (CRITICAL)
**Issue**: Multiple missing Python packages
**Impact**: Services can't import required modules
**Fix Priority**: URGENT

```bash
# Install missing dependencies
pip install transformers datasets scikit-learn pandas numpy
pip install flask flask-restx aiohttp asyncio
pip install redis sqlite3 prometheus-client
```

### 3. Database Schema Issues (HIGH)
**Issue**: Missing columns in health_metrics table
**Impact**: Monitoring system fails
**Fix Priority**: HIGH

### 4. Service Import Errors (HIGH)
**Issue**: Incorrect import paths and missing modules
**Impact**: Services fail to start
**Fix Priority**: HIGH

### 5. Frontend Dependencies (MEDIUM)
**Issue**: Missing npm packages
**Impact**: Frontend build fails
**Fix Priority**: MEDIUM

---

## 🔧 IMMEDIATE FIXES NEEDED

### Phase 1: Core Infrastructure (30 minutes)
1. **Fix PyTorch Installation**
2. **Install Missing Dependencies**
3. **Fix Database Schema**
4. **Test Basic Service Startup**

### Phase 2: Service Fixes (45 minutes)
1. **Fix Import Paths**
2. **Create Missing Modules**
3. **Fix Service Configuration**
4. **Test Individual Services**

### Phase 3: Integration Testing (30 minutes)
1. **Test Service Communication**
2. **Fix API Endpoints**
3. **Test Frontend-Backend Integration**
4. **Demo Readiness Check**

---

## 📋 DETAILED FIX STEPS

### Step 1: Environment Setup
```bash
# Create clean virtual environment
python -m venv demo_env
demo_env\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets scikit-learn
pip install flask flask-restx aiohttp
pip install pandas numpy matplotlib
```

### Step 2: Database Fixes
```sql
-- Fix health_metrics table
ALTER TABLE health_metrics ADD COLUMN metric_value REAL;
ALTER TABLE health_metrics ADD COLUMN metric_unit TEXT;
```

### Step 3: Service Fixes
- Fix import paths in `ai_service_flask/`
- Create missing stub modules
- Fix service configuration files
- Add error handling for missing dependencies

### Step 4: Frontend Fixes
```bash
cd frontend
npm install
npm install react-hot-toast react-icons clsx
```

---

## 🎯 DEMO-READY COMPONENTS

### What Should Work After Fixes:
1. **Basic AI Matching Service** - Core matching functionality
2. **Frontend Dashboard** - Main UI components
3. **Database Operations** - CRUD operations
4. **API Endpoints** - Basic REST API
5. **Simple Demo Flow** - End-to-end matching demo

### What Can Be Stubbed for Demo:
1. **Advanced Analytics** - Use mock data
2. **Real-time Monitoring** - Disable complex monitoring
3. **Federated Learning** - Use local models only
4. **Complex ML Pipelines** - Use simplified versions

---

## 🚀 QUICK DEMO SETUP

### Minimal Working Demo:
1. **Start Backend Services**
   ```bash
   cd backend
   python simple_ai_service.py
   ```

2. **Start Frontend**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Demo Flow**:
   - Upload material data
   - Run AI matching
   - View results in dashboard
   - Show match quality metrics

---

## 📊 SUCCESS METRICS

### Demo Success Criteria:
- [ ] Backend services start without errors
- [ ] Frontend loads and displays dashboard
- [ ] Basic AI matching works
- [ ] Database operations succeed
- [ ] API endpoints respond correctly
- [ ] End-to-end demo flow works

### Quality Gates:
- [ ] No critical errors in logs
- [ ] All core features functional
- [ ] UI responsive and working
- [ ] Data persistence working
- [ ] Basic error handling in place

---

## 🔄 ROLLBACK PLAN

If fixes cause more issues:
1. **Revert to last working state**
2. **Use simplified demo version**
3. **Focus on core functionality only**
4. **Use mock data for complex features**

---

*Last Updated: [Current Date]*
*Priority: URGENT - Demo Blocking*