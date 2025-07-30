# 🚀 SymbioFlows Quick Reference Guide
## Essential Commands, Files, and Information for Mastery

---

## 📁 **CRITICAL FILE LOCATIONS**

### **Core Backend Files**
```
backend/app.js                    # Main Express server (5,331 lines)
backend/revolutionary_ai_matching.py  # Advanced AI matching (1,621 lines)
backend/gnn_reasoning_engine.py   # GNN implementation (36KB)
backend/ai_listings_generator.py  # AI listing generation (1,808 lines)
backend/adaptive_ai_onboarding.py # Smart onboarding (1,172 lines)
backend/world_class_ai_intelligence.py # Core AI system (2,562 lines)
```

### **Core Frontend Files**
```
frontend/src/App.tsx              # Main React app (355 lines)
frontend/src/components/Dashboard.tsx # Business dashboard (1,136 lines)
frontend/src/components/Marketplace.tsx # Marketplace interface (840 lines)
frontend/src/components/AdaptiveAIOnboarding.tsx # AI onboarding (468 lines)
frontend/src/components/RevolutionaryAIMatching.tsx # AI matching (674 lines)
```

### **AI Service Files**
```
ai_service_flask/ai_gateway.py    # AI service orchestrator
ai_service_flask/gnn_reasoning_service.py # GNN service
ai_service_flask/federated_learning_service.py # Federated learning
ai_service_flask/multi_hop_symbiosis_service.py # Multi-hop analysis
ai_service_flask/advanced_analytics_service.py # Analytics engine
```

### **Configuration Files**
```
backend/.env                      # Environment variables
backend/package.json              # Node.js dependencies
frontend/package.json             # React dependencies
backend/requirements.txt          # Python dependencies
supabase/migrations/              # Database schema
```

---

## 🔧 **ESSENTIAL COMMANDS**

### **Development Setup**
```bash
# Backend setup
cd backend
npm install
npm start

# Frontend setup
cd frontend
npm install
npm run dev

# Python AI services
cd ai_service_flask
pip install -r requirements.txt
python ai_gateway.py
```

### **Database Operations**
```bash
# Supabase CLI
supabase start
supabase db reset
supabase db push

# Migration commands
supabase migration new create_users_table
supabase migration up
```

### **Testing Commands**
```bash
# Backend tests
cd backend
npm test

# Frontend tests
cd frontend
npm test

# AI service tests
cd ai_service_flask
python -m pytest tests/
```

### **Deployment Commands**
```bash
# Frontend deployment (Vercel)
cd frontend
vercel --prod

# Backend deployment (Railway)
cd backend
railway up

# AI services deployment
docker build -t ai-services .
docker run -p 5000:5000 ai-services
```

---

## 🧠 **AI SERVICES PORTS & ENDPOINTS**

### **Core AI Services**
```
AI Gateway:          http://localhost:5000
GNN Inference:       http://localhost:5001
Federated Learning:  http://localhost:5002
Multi-Hop Symbiosis: http://localhost:5003
Advanced Analytics:  http://localhost:5004
AI Pricing:          http://localhost:5005
Logistics:           http://localhost:5006
Materials BERT:      http://localhost:5007
```

### **Key API Endpoints**
```
GET  /api/health                    # Health check
POST /api/ai-infer-listings         # AI listing generation
POST /api/match                     # AI matching
POST /api/ai-pipeline              # AI pipeline orchestration
POST /api/ai-chat                  # AI chat interface
GET  /api/materials                # Material listings
POST /api/transactions             # Transaction processing
GET  /api/analytics/carbon         # Carbon footprint
```

---

## 🗄️ **DATABASE SCHEMA QUICK REFERENCE**

### **Core Tables**
```sql
-- Users & Companies
users (id, email, created_at, updated_at)
companies (id, name, industry, location, size, sustainability_score)
user_profiles (id, user_id, preferences, settings)
company_profiles (id, company_id, description, certifications)

-- Materials & Listings
materials (id, name, type, category, properties, sustainability_metrics)
material_listings (id, material_id, company_id, quantity, price, status)
categories (id, name, parent_id, industry_specific)
material_properties (id, material_id, property_name, value, unit)

-- Matching & Transactions
matches (id, material_id, consumer_id, producer_id, score, status)
match_analytics (id, match_id, performance_metrics, feedback)
transactions (id, match_id, amount, status, payment_method)
transaction_history (id, transaction_id, status_changes, timestamps)

-- AI & Analytics
ai_insights (id, company_id, insight_type, confidence, recommendations)
ai_models (id, model_name, version, performance_metrics, last_updated)
analytics_events (id, event_type, user_id, data, timestamp)
performance_metrics (id, service_name, metric_name, value, timestamp)
```

### **Key Relationships**
```sql
users (1) ──── (1) user_profiles
users (1) ──── (1) companies
companies (1) ──── (1) company_profiles
companies (1) ──── (n) material_listings
materials (1) ──── (n) material_listings
material_listings (1) ──── (n) matches
matches (1) ──── (n) transactions
```

---

## 🎨 **FRONTEND COMPONENT HIERARCHY**

### **Main Components**
```
App.tsx
├── LandingPage
├── AuthModal
├── Dashboard
│   ├── PersonalPortfolio
│   ├── Marketplace
│   ├── TransactionPage
│   └── AdminHub
├── AdaptiveAIOnboarding
├── RevolutionaryAIMatching
├── NotificationsPanel
└── ChatsPanel
```

### **Key Component Files**
```
components/
├── Dashboard.tsx (1,136 lines)           # Main business dashboard
├── Marketplace.tsx (840 lines)           # Marketplace interface
├── AdaptiveAIOnboarding.tsx (468 lines)  # AI onboarding wizard
├── RevolutionaryAIMatching.tsx (674 lines) # AI matching interface
├── PersonalPortfolio.tsx (624 lines)     # User portfolio
├── AdminHub.tsx (693 lines)              # Admin dashboard
├── AuthModal.tsx (501 lines)             # Authentication
└── ChatInterface.tsx (444 lines)         # Chat system
```

---

## 🔧 **BACKEND SERVICE ARCHITECTURE**

### **Main Services**
```
services/
├── intelligentMatchingService.js    # AI matching orchestration
├── apiFusionService.js              # Multi-AI fusion
├── materialsService.js              # Material management
├── shippingService.js               # Logistics integration
├── heightService.js                 # Project tracking
├── freightosLogisticsService.js     # Freight integration
└── aiEvolutionEngine.js             # AI model evolution
```

### **Key Python AI Files**
```
backend/
├── revolutionary_ai_matching.py (1,621 lines)     # Advanced AI matching
├── ai_listings_generator.py (1,808 lines)         # AI listing generation
├── adaptive_ai_onboarding.py (1,172 lines)        # Smart onboarding
├── world_class_ai_intelligence.py (2,562 lines)   # Core AI system
├── gnn_reasoning_engine.py (36KB)                 # GNN implementation
├── advanced_analytics_engine.py (41KB)            # Analytics engine
├── ai_feedback_orchestrator.py (1,277 lines)      # Feedback system
└── ai_hyperparameter_optimizer.py (39KB)          # Model optimization
```

---

## 🚀 **DEPLOYMENT CONFIGURATION**

### **Environment Variables**
```bash
# Backend (.env)
PORT=5000
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_key
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
STRIPE_SECRET_KEY=your_stripe_key

# Frontend (.env.local)
VITE_API_URL=http://localhost:5000
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_key
```

### **Production URLs**
```
Frontend: https://symbioflows.com
Backend:  https://api.symbioflows.com
Database: https://supabase.com/dashboard/project/your-project
```

---

## 🔒 **SECURITY CONFIGURATION**

### **Authentication Flow**
```
1. User registration → Supabase Auth
2. JWT token generation → Secure storage
3. API requests → Token validation
4. Role-based access → Route protection
5. Session management → Auto-refresh
```

### **Security Headers**
```javascript
// Helmet configuration
app.use(helmet({
  crossOriginResourcePolicy: { policy: "cross-origin" },
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      fontSrc: ["'self'", "data:"],
      connectSrc: ["'self'", "https:", "wss:"],
      frameAncestors: ["'none'"],
    },
  },
}));
```

---

## 📊 **PERFORMANCE METRICS**

### **Target Performance**
```
API Response Time:    < 200ms
AI Processing Time:   < 5s
Frontend Load Time:   < 2s
Database Query Time:  < 100ms
Real-time Updates:    < 1s
```

### **Monitoring Endpoints**
```
GET /api/health                    # Health check
GET /metrics                       # Prometheus metrics
GET /api/performance              # Performance metrics
GET /api/analytics/system         # System analytics
```

---

## 🧠 **AI ALGORITHM QUICK REFERENCE**

### **Core AI Techniques**
```
1. Graph Neural Networks (GNN)
   - PyTorch Geometric
   - Message passing algorithms
   - Multi-hop relationship analysis

2. Federated Learning
   - Privacy-preserving training
   - Secure aggregation
   - Cross-company knowledge sharing

3. Quantum-Inspired Algorithms
   - Optimization techniques
   - Quantum neural networks
   - Exponential speedup

4. Multi-Agent Systems
   - Reinforcement learning
   - Swarm intelligence
   - Coordination protocols
```

### **AI Service Integration**
```
AI Gateway → Route Request → Appropriate AI Service
├── GNN Service → Graph analysis
├── Federated Service → Privacy-preserving learning
├── Analytics Service → Business intelligence
├── Pricing Service → Market analysis
└── Materials Service → Materials science
```

---

## 💼 **BUSINESS LOGIC FLOW**

### **Material Matching Process**
```
1. Company creates listing → AI analyzes properties
2. AI generates content → Database stores listing
3. Matching engine processes → GNN identifies relationships
4. Potential matches found → Notifications sent
5. Companies negotiate → AI assists with pricing
6. Transaction completed → Feedback collected
7. AI models updated → Performance improved
```

### **Revenue Streams**
```
1. Transaction Fees: 2-5% of successful matches
2. Subscription Tiers: Premium features
3. AI Services: Advanced analytics
4. Logistics: Freight and transportation
5. Consulting: Strategic advisory services
```

---

## 🎯 **TROUBLESHOOTING GUIDE**

### **Common Issues**
```
Issue: AI services not responding
Fix: Check Python services on ports 5000-5007

Issue: Database connection errors
Fix: Verify Supabase credentials in .env

Issue: Frontend not loading
Fix: Check VITE_API_URL in .env.local

Issue: Authentication failing
Fix: Verify JWT token and Supabase Auth

Issue: Real-time updates not working
Fix: Check WebSocket connections and Supabase real-time
```

### **Debug Commands**
```bash
# Check service health
curl http://localhost:5000/api/health

# Check AI services
curl http://localhost:5000/api/ai-infer-listings

# Check database
supabase db reset

# Check logs
tail -f backend/logs/app.log
```

---

## 🚀 **DEVELOPMENT WORKFLOW**

### **Feature Development**
```
1. Create feature branch
2. Implement changes
3. Write tests
4. Update documentation
5. Submit pull request
6. Code review
7. Merge to main
8. Deploy to production
```

### **AI Model Updates**
```
1. Collect feedback data
2. Retrain models
3. Validate performance
4. A/B testing
5. Gradual rollout
6. Monitor metrics
7. Full deployment
```

---

## 📚 **LEARNING RESOURCES**

### **Key Documentation**
```
docs/ARCHITECTURE_OVERVIEW.md      # System architecture
docs/PRODUCTION_AI_SYSTEM.md       # AI system details
docs/COMPREHENSIVE_AI_SYSTEM_DOCUMENTATION.md # Complete AI docs
backend/README.md                  # Backend setup
frontend/README.md                 # Frontend setup
```

### **External Resources**
```
- PyTorch Geometric: Graph neural networks
- Supabase: Database and authentication
- Vercel: Frontend deployment
- Railway: Backend deployment
- Stripe: Payment processing
```

---

**Quick Reference Goal**: Master the SymbioFlows codebase efficiently  
**Study Strategy**: Use this guide daily for rapid reference  
**Mastery Timeline**: 16 weeks with 2-3 hours daily study 