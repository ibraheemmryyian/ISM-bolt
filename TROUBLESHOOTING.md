# 🔧 AI Onboarding Troubleshooting Guide

## **Issue: "Backend not available. Please contact support."**

This error occurs when the frontend can't connect to the backend API server.

## **Quick Fix Steps:**

### **1. Start the Backend Server**
```bash
# Navigate to backend directory
cd backend

# Install dependencies (if not done already)
npm install

# Start the server
npm start
```

You should see:
```
🚀 Server running on port 5000
📊 Health check: http://localhost:5000/api/health
🔒 Security: Helmet, CORS, Rate limiting enabled
✅ Input validation: Express-validator enabled
```

### **2. Start the Frontend Server**
```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Install dependencies (if not done already)
npm install

# Start the frontend
npm run dev
```

### **3. Test the Connection**
```bash
# Test if backend is running
curl http://localhost:5000/api/health

# Or use the test script
cd backend
node test-ai.js
```

## **Common Issues & Solutions:**

### **Issue 1: Backend Server Not Running**
**Symptoms:** "Cannot connect to backend" error
**Solution:** 
- Make sure you're in the backend directory
- Run `npm start`
- Check if port 5000 is available

### **Issue 2: Port Already in Use**
**Symptoms:** "EADDRINUSE" error
**Solution:**
```bash
# Find what's using port 5000
netstat -ano | findstr :5000

# Kill the process or change port in backend/app.js
```

### **Issue 3: py Dependencies Missing**
**Symptoms:** "Module not found" errors in backend
**Solution:**
```bash
# Install py dependencies
pip install -r requirements.txt

# Or install individually:
pip install numpy pandas scikit-learn transformers torch
```

### **Issue 4: CORS Issues**
**Symptoms:** "CORS error" in browser console
**Solution:** 
- Backend is configured with CORS
- Frontend proxy is set up in vite.config.ts
- Make sure both servers are running

### **Issue 5: Frontend Proxy Not Working**
**Symptoms:** 404 errors on API calls
**Solution:**
- Restart frontend server after changing vite.config.ts
- Check that proxy configuration is correct
- Try accessing API directly: http://localhost:5000/api/health

## **Verification Steps:**

### **1. Check Backend Health**
```bash
curl http://localhost:5000/api/health
```
Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "version": "1.0.0"
}
```

### **2. Test AI Inference**
```bash
cd backend
node test-ai.js
```

### **3. Check Browser Console**
- Open browser developer tools (F12)
- Go to Console tab
- Look for any error messages
- Check Network tab for failed requests

## **Development Setup:**

### **Required Services:**
1. **Backend Server** (Node.js + Express) - Port 5000
2. **Frontend Server** (Vite + React) - Port 5173
3. **py AI Engine** (for AI inference)
4. **Supabase Database** (for user data)

### **Environment Variables:**
Create `.env` file in backend directory:
```env
PORT=5000
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
```

### **File Structure:**
```
ISM [AI]/
├── backend/
│   ├── app.js              # Main server
│   ├── package.json        # Dependencies
│   └── test-ai.js          # Test script
├── frontend/
│   ├── vite.config.ts      # Proxy config
│   └── src/
└── revolutionary_ai_matching.py  # AI engine
```

## **Debug Mode:**

### **Enable Detailed Logging:**
```javascript
// In backend/app.js, add:
console.log('Request received:', req.body);
console.log('py result:', result);
```

### **Check py AI Engine:**
```bash
# Test py AI directly
py revolutionary_ai_matching.py
```

## **Alternative Solutions:**

### **If Backend Still Won't Start:**
1. Check Node.js version: `node --version` (should be 18+)
2. Clear npm cache: `npm cache clean --force`
3. Delete node_modules and reinstall: `rm -rf node_modules && npm install`

### **If AI Engine Fails:**
1. Check py version: `py --version` (should be 3.8+)
2. Install missing packages: `pip install package-name`
3. Check file paths in revolutionary_ai_matching.py

### **If Frontend Can't Connect:**
1. Check if both servers are running
2. Verify proxy configuration in vite.config.ts
3. Try accessing API directly in browser
4. Check browser console for CORS errors

## **Emergency Fallback:**

If AI onboarding still doesn't work, you can:
1. **Skip AI onboarding** - Go directly to dashboard
2. **Manual listing creation** - Create listings manually
3. **Contact support** - Provide error logs and system info

## **Support Information:**

When contacting support, provide:
- Error message from browser console
- Backend server logs
- System information (OS, Node.js version, py version)
- Steps to reproduce the issue

---

**🎯 Most common solution: Start the backend server with `cd backend && npm start`** 