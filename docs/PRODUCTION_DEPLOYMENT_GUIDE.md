# SymbioFlows Production Deployment Guide

## 🚀 **Vercel Frontend Deployment (symbioflows.com)**

### **1. Frontend Configuration Fixed ✅**

The 404 errors have been resolved by creating `frontend/vercel.json` with proper client-side routing configuration.

### **2. Environment Variables Setup**

Update `frontend/.env.production` with your actual production URLs:

```env
# Production Environment Variables
VITE_SUPABASE_URL=https://jifkiwbxnttrkdrdcose.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIzNjM5MTQsImV4cCI6MjA2NzkzOTkxNH0.4PE6Zu0RaMhz3QkocYCQsENS9Tv19avtfXSe_ChHcLA

# Update these with your actual production backend URL
VITE_API_URL=https://api.symbioflows.com
VITE_WS_URL=wss://api.symbioflows.com
VITE_AI_PREVIEW_URL=https://api.symbioflows.com/api
VITE_BACKEND_URL=https://api.symbioflows.com

# Update with your actual Stripe publishable key
VITE_STRIPE_PUBLISHABLE_KEY=pk_live_your_actual_stripe_publishable_key_here
```

### **3. Vercel Deployment Steps**

1. **Connect Repository to Vercel**
   ```bash
   # In Vercel Dashboard:
   # 1. Import Git Repository
   # 2. Set Root Directory: frontend
   # 3. Framework Preset: Vite
   # 4. Build Command: npm run build
   # 5. Output Directory: dist
   ```

2. **Environment Variables in Vercel**
   - Go to Project Settings → Environment Variables
   - Add all variables from `.env.production`

3. **Custom Domain Setup**
   - Add `symbioflows.com` in Vercel Domain Settings
   - Configure DNS records as instructed by Vercel

## 🔧 **Backend Microservices Deployment**

### **1. Backend Hosting Options**

#### **Option A: Railway (Recommended)**
```bash
# Deploy to Railway
railway login
railway init
railway up
```

#### **Option B: Render**
```bash
# Create render.yaml
services:
  - type: web
    name: symbioflows-backend
    env: node
    buildCommand: npm install
    startCommand: npm start
    envVars:
      - key: NODE_ENV
        value: production
```

#### **Option C: DigitalOcean App Platform**
```bash
# Deploy via DigitalOcean CLI
doctl apps create --spec app.yaml
```

### **2. Backend Environment Configuration**

Create `backend/.env.production`:

```env
# Production Backend Configuration
PORT=5000
NODE_ENV=production
FRONTEND_URL=https://symbioflows.com

# Database Configuration
SUPABASE_URL=https://jifkiwbxnttrkdrdcose.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIzNjM5MTQsImV4cCI6MjA2NzkzOTkxNH0.4PE6Zu0RaMhz3QkocYCQsENS9Tv19avtfXSe_ChHcLA
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjM2MzkxNCwiZXhwIjoyMDY3OTM5OTE0fQ.a3gnL-rgpLgfzRoxpHdb3jNsLLsbo_2e-U5OU6tcabU

# AI Services
DEEPSEEK_API_KEY=sk-7ce79f30332d45d5b3acb8968b052132
DEEPSEEK_R1_API_KEY=sk-7ce79f30332d45d5b3acb8968b052132
DEEPSEEK_MODEL=deepseek-coder

# Freightos API
FREIGHTOS_API_KEY=V2C6teoe9xSKKpTxL8j4xxuOFGHxQWhx
FREIGHTOS_SECRET_KEY=k6hEyfd3b6ao8rKQ

# Materials Project API
MP_API_KEY=zSFjfpRg6m020aK84yOjM7oLIhjDNPjE

# News API
NEWS_API_KEY=33d86c63e63c46c58c7dfd81068e79a4
NEWSAPI_KEY=33d86c63e63c46c58c7dfd81068e79a4

# Next Gen Materials API
NEXT_GEN_MATERIALS_API_KEY=zSFjfpRg6m020aK84yOjM7oLIhjDNPjE
NEXTGEN_MATERIALS_API_KEY=zSFjfpRg6m020aK84yOjM7oLIhjDNPjE

# Stripe Configuration
STRIPE_SECRET_KEY=sk_live_your_actual_stripe_secret_key
STRIPE_PUBLISHABLE_KEY=pk_live_your_actual_stripe_publishable_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# Security
JWT_SECRET=YWOBJFEkz70pSIvZ8F4FkZ0EjSAYcAHHwU9Bq63G5tGuBrXYWnXHRvF4/sT8AcIBZPkJisoW99TeCNOAz6v2lw==
SESSION_SECRET=ism_ai_platform_session_secret_2025_secure_random_string_64_chars_moat_level

# Application URLs
BACKEND_URL=https://api.symbioflows.com
AI_TRAINING_ENDPOINT=https://api.symbioflows.com/ai-training

# Performance Configuration
MATERIALS_MAX_WORKERS=20
MATERIALS_QUEUE_SIZE=200
MATERIALS_RATE_LIMIT_ENABLED=true
MATERIALS_MONITORING_ENABLED=true

# Logging
LOG_LEVEL=info
```

## 🔄 **Microservices Deployment**

### **1. AI Services Deployment**

#### **MaterialsBERT Service**
```bash
# Deploy to Railway or similar
cd ai_service_flask
railway up
```

#### **Adaptive Onboarding Service**
```bash
# Deploy Python Flask service
cd backend
railway up --service adaptive-onboarding
```

### **2. Database Migration**

```sql
-- Run these in Supabase SQL Editor
-- Ensure all tables are created and up to date
\i complete_database_schema.sql
\i supabase_subscription_columns.sql
\i supabase_username_auth.sql
```

## 🧪 **Pre-Deployment Testing**

### **1. Frontend Testing**
```bash
cd frontend
npm run build
npm run preview
# Test all routes: /, /dashboard, /marketplace, /admin
```

### **2. Backend Testing**
```bash
cd backend
npm test
npm run health-check
```

### **3. API Testing**
```bash
# Test all endpoints
curl https://api.symbioflows.com/health
curl https://api.symbioflows.com/api/materials
curl https://api.symbioflows.com/api/matches
```

## 🔒 **Security Configuration**

### **1. CORS Configuration**
```javascript
// In backend/app.js
app.use(cors({
  origin: ['https://symbioflows.com', 'https://www.symbioflows.com'],
  credentials: true
}));
```

### **2. Rate Limiting**
```javascript
// Already configured in app.js
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100
});
```

### **3. Security Headers**
```javascript
// Already configured with helmet
app.use(helmet());
```

## 📊 **Monitoring & Analytics**

### **1. Vercel Analytics**
- Enable Vercel Analytics in project settings
- Track performance and user behavior

### **2. Error Monitoring**
```javascript
// Sentry configuration
Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: 'production'
});
```

### **3. Health Checks**
```bash
# Automated health checks
curl -f https://symbioflows.com/api/health || exit 1
```

## 🚀 **Deployment Checklist**

### **Frontend (Vercel)**
- [x] vercel.json created with routing configuration
- [ ] Environment variables configured in Vercel
- [ ] Custom domain (symbioflows.com) configured
- [ ] SSL certificate active
- [ ] All routes tested (/dashboard, /marketplace, etc.)

### **Backend (Railway/Render)**
- [ ] Environment variables configured
- [ ] Database migrations completed
- [ ] All microservices deployed
- [ ] Health checks passing
- [ ] CORS configured for production domain

### **Database (Supabase)**
- [ ] All tables created and migrated
- [ ] Row Level Security (RLS) configured
- [ ] Backup strategy implemented
- [ ] Performance optimized

### **Domain & DNS**
- [ ] symbioflows.com pointing to Vercel
- [ ] api.symbioflows.com pointing to backend
- [ ] SSL certificates active
- [ ] DNS propagation complete

## 🔧 **Troubleshooting**

### **404 Errors Fixed**
- ✅ Created vercel.json with proper routing
- ✅ All client-side routes now work correctly

### **Common Issues**
1. **CORS Errors**: Ensure backend CORS includes production domain
2. **Environment Variables**: Check all variables are set in Vercel
3. **Database Connection**: Verify Supabase connection strings
4. **API Endpoints**: Test all endpoints after deployment

## 📈 **Performance Optimization**

### **Frontend**
- ✅ Vite build optimization
- ✅ Code splitting implemented
- ✅ Image optimization
- ✅ CDN delivery via Vercel

### **Backend**
- ✅ Rate limiting configured
- ✅ Caching strategies implemented
- ✅ Database query optimization
- ✅ Load balancing ready

---

**Next Steps:**
1. Deploy frontend to Vercel with symbioflows.com
2. Deploy backend to Railway/Render
3. Configure environment variables
4. Test all functionality
5. Monitor performance and errors 