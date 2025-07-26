# Deploy to Vercel - Get Your Live Link

## 🚀 **Quick Deployment Steps**

### **Option 1: Vercel Dashboard (Recommended)**

1. **Go to Vercel Dashboard:**
   - Visit: https://vercel.com/dashboard
   - Click "New Project"
   - Import your GitHub repository

2. **Configure Project Settings:**
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`

3. **Add Environment Variables:**
   - Go to Project Settings → Environment Variables
   - Add all variables from `frontend/.env.production`

4. **Deploy:**
   - Click "Deploy"
   - Wait for build to complete
   - Get your live URL!

### **Option 2: Vercel CLI (Alternative)**

```bash
# Navigate to frontend directory
cd frontend

# Deploy to Vercel
npx vercel --prod

# Follow the prompts:
# - Set up and deploy? → Y
# - Which scope? → Select your account
# - Link to existing project? → N
# - Project name? → symbioflows-frontend
# - Directory? → ./
# - Override settings? → N
```

## 🔧 **Environment Variables for Production**

Make sure these are set in Vercel Dashboard:

```env
VITE_SUPABASE_URL=https://jifkiwbxnttrkdrdcose.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIzNjM5MTQsImV4cCI6MjA2NzkzOTkxNH0.4PE6Zu0RaMhz3QkocYCQsENS9Tv19avtfXSe_ChHcLA

# Update these with your actual backend URL after deployment
VITE_API_URL=https://your-backend-url.railway.app
VITE_WS_URL=wss://your-backend-url.railway.app
VITE_AI_PREVIEW_URL=https://your-backend-url.railway.app/api
VITE_BACKEND_URL=https://your-backend-url.railway.app

VITE_STRIPE_PUBLISHABLE_KEY=pk_live_your_stripe_key
```

## 🌐 **Your Live URLs**

After deployment, you'll get:

- **Frontend:** `https://your-project-name.vercel.app`
- **Custom Domain:** `https://symbioflows.com` (after setup)

## 🔗 **Next Steps After Deployment**

1. **Deploy Backend to Railway:**
   ```bash
   cd backend
   railway up
   ```

2. **Update Frontend Environment:**
   - Get your Railway URL: `railway domain`
   - Update Vercel environment variables with Railway URL

3. **Test Everything:**
   - Test login/registration
   - Test marketplace
   - Test AI onboarding
   - Test all routes

## 🚨 **Troubleshooting**

### **404 Errors:**
- ✅ Fixed with updated `vercel.json`
- All routes should work now

### **API Connection Errors:**
- Make sure backend is deployed
- Update environment variables with correct backend URL

### **Build Errors:**
- Check Node.js version compatibility
- Ensure all dependencies are installed

## 📊 **Monitoring**

- **Vercel Dashboard:** Monitor deployments and performance
- **Function Logs:** Check for any runtime errors
- **Analytics:** Track user behavior and performance

## 🔄 **Redeployment**

After making changes:

```bash
# Automatic redeployment via Git push
git push origin main

# Or manual deployment
npx vercel --prod
```

Your live link will be available immediately after deployment! 