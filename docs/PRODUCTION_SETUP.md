# 🚀 Production Setup Guide

## 📋 **Current Status**

✅ **Backend files moved to `backend/` directory**  
✅ **Frontend files remain in `frontend/` directory**  
✅ **Root directory cleaned up**  
🔄 **Production deployment configuration needed**

## 🏗️ **Production Architecture**

```
SymbioFlows/
├── frontend/                 # React + TypeScript + Vite
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vercel.json          # Frontend deployment
├── backend/                  # Node.js + Python AI Services
│   ├── app.js               # Main Express server
│   ├── *.py                 # AI services
│   ├── package.json
│   ├── railway.json         # Backend deployment
│   └── .env                 # Backend environment
└── docs/                    # Documentation
```

## 🌐 **Deployment Strategy**

### **Frontend (Vercel)**
- **URL**: `https://symbioflows.com`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Environment**: Production

### **Backend (Railway/Render)**
- **URL**: `https://api.symbioflows.com`
- **Port**: `5000`
- **Environment**: Production
- **AI Services**: Python scripts in backend directory

## 🔧 **Environment Configuration**

### **Frontend Environment (frontend/.env.production)**
```env
VITE_API_URL=https://api.symbioflows.com
VITE_WS_URL=wss://api.symbioflows.com
VITE_AI_PREVIEW_URL=https://api.symbioflows.com
VITE_BACKEND_URL=https://api.symbioflows.com
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_key
```

### **Backend Environment (backend/.env.production)**
```env
PORT=5000
NODE_ENV=production
FRONTEND_URL=https://symbioflows.com
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key
MATERIALS_PROJECT_API_KEY=your_materials_key
NEWS_API_KEY=your_news_key
NEXT_GEN_MATERIALS_API_KEY=your_nextgen_key
STRIPE_SECRET_KEY=your_stripe_key
JWT_SECRET=your_jwt_secret
SESSION_SECRET=your_session_secret
```

## 🔄 **Communication Flow**

### **Frontend → Backend**
```typescript
// Frontend API calls
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// Example API call
const response = await fetch(`${API_BASE_URL}/api/materials`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify(data)
});
```

### **Backend → AI Services**
```javascript
// Backend calls Python scripts
const result = await runPythonScript('revolutionary_ai_matching.py', {
  source_material: materialName,
  source_type: materialType,
  source_company: companyName
});
```

## 🚀 **Deployment Commands**

### **Deploy Frontend to Vercel**
```bash
cd frontend
npm run build
vercel --prod
```

### **Deploy Backend to Railway**
```bash
cd backend
railway login
railway init
railway up
```

### **Deploy Backend to Render**
```bash
cd backend
# Push to GitHub, Render will auto-deploy
```

## 🔒 **Security Configuration**

### **CORS Settings (backend/app.js)**
```javascript
app.use(cors({
  origin: [
    'https://symbioflows.com',
    'https://www.symbioflows.com',
    'http://localhost:5173' // Development
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));
```

### **Rate Limiting**
```javascript
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);
```

## 📊 **Monitoring & Health Checks**

### **Health Check Endpoint**
```javascript
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version || '1.0.0'
  });
});
```

### **Metrics Endpoint**
```javascript
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', client.register.contentType);
  res.end(await client.register.metrics());
});
```

## 🔧 **Next Steps**

1. **Update Environment Variables**
   - Set production API keys
   - Configure database connections
   - Set up monitoring

2. **Test Communication**
   - Verify frontend → backend API calls
   - Test AI service integration
   - Validate authentication flow

3. **Deploy to Production**
   - Deploy backend first
   - Update frontend environment
   - Deploy frontend
   - Test end-to-end functionality

4. **Monitor & Optimize**
   - Set up logging
   - Monitor performance
   - Optimize AI service response times

## 🎯 **Success Criteria**

✅ **Frontend and backend are completely separate**  
✅ **Communication works via HTTP APIs**  
✅ **AI services are properly integrated**  
✅ **Authentication and security are configured**  
✅ **Production environment is stable**  
✅ **Monitoring and health checks are active**

---

**Last Updated**: July 2025  
**Status**: Ready for Production Deployment 