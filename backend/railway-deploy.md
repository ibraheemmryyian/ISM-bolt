# Railway Backend Deployment Guide

## 🚀 **Step-by-Step Railway Deployment**

### **1. Install Railway CLI**
```bash
npm install -g @railway/cli
```

### **2. Login to Railway**
```bash
railway login
```

### **3. Initialize Railway Project**
```bash
cd backend
railway init
```

### **4. Set Environment Variables**
```bash
# Copy all variables from .env.production
railway variables set NODE_ENV=production
railway variables set PORT=5000
railway variables set FRONTEND_URL=https://symbioflows.com
railway variables set SUPABASE_URL=https://jifkiwbxnttrkdrdcose.supabase.co
railway variables set SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIzNjM5MTQsImV4cCI6MjA2NzkzOTkxNH0.4PE6Zu0RaMhz3QkocYCQsENS9Tv19avtfXSe_ChHcLA
railway variables set SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjM2MzkxNCwiZXhwIjoyMDY3OTM5OTE0fQ.a3gnL-rgpLgfzRoxpHdb3jNsLLsbo_2e-U5OU6tcabU
railway variables set DEEPSEEK_API_KEY=sk-7ce79f30332d45d5b3acb8968b052132
railway variables set DEEPSEEK_R1_API_KEY=sk-7ce79f30332d45d5b3acb8968b052132
railway variables set FREIGHTOS_API_KEY=V2C6teoe9xSKKpTxL8j4xxuOFGHxQWhx
railway variables set FREIGHTOS_SECRET_KEY=k6hEyfd3b6ao8rKQ
railway variables set MP_API_KEY=zSFjfpRg6m020aK84yOjM7oLIhjDNPjE
railway variables set NEWS_API_KEY=33d86c63e63c46c58c7dfd81068e79a4
railway variables set NEXT_GEN_MATERIALS_API_KEY=zSFjfpRg6m020aK84yOjM7oLIhjDNPjE
railway variables set JWT_SECRET=YWOBJFEkz70pSIvZ8F4FkZ0EjSAYcAHHwU9Bq63G5tGuBrXYWnXHRvF4/sT8AcIBZPkJisoW99TeCNOAz6v2lw==
railway variables set SESSION_SECRET=ism_ai_platform_session_secret_2025_secure_random_string_64_chars_moat_level
railway variables set BACKEND_URL=https://api.symbioflows.com
```

### **5. Deploy to Railway**
```bash
railway up
```

### **6. Get Your Backend URL**
```bash
railway domain
# This will give you something like: https://your-app-name.railway.app
```

### **7. Update Frontend Environment**
Update `frontend/.env.production` with your Railway URL:
```env
VITE_API_URL=https://your-app-name.railway.app
VITE_WS_URL=wss://your-app-name.railway.app
VITE_AI_PREVIEW_URL=https://your-app-name.railway.app/api
VITE_BACKEND_URL=https://your-app-name.railway.app
```

### **8. Set Custom Domain (Optional)**
```bash
# In Railway Dashboard:
# 1. Go to Settings → Domains
# 2. Add custom domain: api.symbioflows.com
# 3. Configure DNS records as instructed
```

## 🔧 **Railway Dashboard Configuration**

### **Environment Variables**
Go to Railway Dashboard → Your Project → Variables tab and add:
- All variables from `backend/.env.production`
- Make sure to set `NODE_ENV=production`

### **Domain Configuration**
1. Go to Settings → Domains
2. Add custom domain: `api.symbioflows.com`
3. Configure DNS records at your domain registrar

### **Health Checks**
Railway will automatically check `/health` endpoint every 30 seconds

## 📊 **Monitoring**

### **Railway Dashboard**
- View logs in real-time
- Monitor resource usage
- Check deployment status
- View environment variables

### **Health Check Endpoint**
```bash
curl https://your-app-name.railway.app/health
# Should return: {"status":"healthy","timestamp":"2025-01-15T..."}
```

## 🔄 **Deployment Commands**

### **Redeploy After Changes**
```bash
railway up
```

### **View Logs**
```bash
railway logs
```

### **Open Dashboard**
```bash
railway open
```

## 🚨 **Troubleshooting**

### **Common Issues**
1. **Build Failures**: Check `package.json` scripts
2. **Environment Variables**: Ensure all required vars are set
3. **Port Issues**: Railway sets PORT automatically
4. **CORS Errors**: Update CORS origin in backend/app.js

### **Debug Commands**
```bash
# Check deployment status
railway status

# View recent logs
railway logs --tail

# SSH into container (if needed)
railway shell
```

## ✅ **Verification Steps**

1. **Test Health Endpoint**
   ```bash
   curl https://your-app-name.railway.app/health
   ```

2. **Test API Endpoints**
   ```bash
   curl https://your-app-name.railway.app/api/materials
   curl https://your-app-name.railway.app/api/matches
   ```

3. **Test Frontend Connection**
   - Update frontend environment variables
   - Test login/registration
   - Test marketplace functionality

## 🔗 **Next Steps**

1. Deploy backend to Railway
2. Get your Railway URL
3. Update frontend environment variables
4. Test all functionality
5. Set up custom domain (optional)
6. Monitor performance and logs 