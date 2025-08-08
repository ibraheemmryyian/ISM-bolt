# 🚀 SymbioFlows Production Deployment Guide

## 📋 Overview

This guide covers deploying SymbioFlows to production with the following architecture:

- **Frontend**: `https://symbioflows.com` (Vercel/Netlify)
- **Backend API**: `https://api.symbioflows.com` (Railway/Render/AWS)
- **AI Services**: `https://api.symbioflows.com/ai/*` (Python/Flask)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │  AI Services    │
│   symbioflows.com│◄──►│api.symbioflows.com│◄──►│api.symbioflows.com│
│   (Vercel)      │    │   (Railway)     │    │  /ai/* (Flask)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Supabase      │    │     Redis       │    │   Monitoring    │
│   Database      │    │     Cache       │    │  (Prometheus)   │
│   (PostgreSQL)  │    │   (Cloud)       │    │  (Grafana)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Environment Configuration

### Frontend Environment (.env)

```bash
# Production API URLs
VITE_API_URL=https://api.symbioflows.com
VITE_AI_SERVICES_URL=https://api.symbioflows.com/ai
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
```

### Backend Environment (.env)

```bash
# Production Configuration
NODE_ENV=production
PORT=3000
FRONTEND_URL=https://symbioflows.com

# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-key

# AI Services
DEEPSEEK_API_KEY=your-deepseek-api-key
OPENAI_API_KEY=your-openai-api-key

# Security
JWT_SECRET=your-super-secret-jwt-key
```

## 🚀 Deployment Steps

### 1. Backend API Deployment (Railway/Render)

#### Option A: Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy backend
cd backend
railway up
```

#### Option B: Render
```bash
# Connect your GitHub repo to Render
# Set environment variables in Render dashboard
# Deploy automatically on push to main branch
```

### 2. AI Services Deployment

```bash
# Deploy AI Gateway to Railway/Render
cd ai_service_flask
railway up

# Or use Docker
docker build -t symbioflows-ai .
docker run -p 5000:5000 symbioflows-ai
```

### 3. Frontend Deployment (Vercel)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy frontend
cd frontend
vercel --prod
```

### 4. Domain Configuration

#### DNS Setup
```
# A Records
api.symbioflows.com → [Railway/Render IP]
symbioflows.com → [Vercel IP]

# CNAME Records
www.symbioflows.com → symbioflows.com
```

#### SSL Certificates
- **Railway/Render**: Automatic SSL
- **Vercel**: Automatic SSL
- **Custom Domain**: Let's Encrypt or Cloudflare

## 🔒 Security Configuration

### CORS Settings
```javascript
// Backend CORS
app.use(cors({
  origin: [
    'https://symbioflows.com',
    'https://www.symbioflows.com',
    'https://symbioflows.vercel.app'
  ],
  credentials: true
}));
```

### Rate Limiting
```javascript
// API Rate Limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
```

## 📊 Monitoring Setup

### Health Checks
- **Backend**: `https://api.symbioflows.com/api/health`
- **AI Services**: `https://api.symbioflows.com/ai/health`
- **Frontend**: `https://symbioflows.com`

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'symbioflows-backend'
    static_configs:
      - targets: ['api.symbioflows.com:3000']

  - job_name: 'symbioflows-ai'
    static_configs:
      - targets: ['api.symbioflows.com:5000']
```

## 🔧 Production Scripts

### Start Production System
```bash
# Start all services
python start_production_demo.py

# Or use Docker Compose
docker-compose -f docker-compose.production.yml up -d
```

### Health Monitoring
```bash
# Check all services
curl https://api.symbioflows.com/api/health
curl https://api.symbioflows.com/ai/health
curl https://symbioflows.com
```

## 🚨 Troubleshooting

### Common Issues

#### 1. CORS Errors
```bash
# Check CORS configuration
curl -H "Origin: https://symbioflows.com" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS https://api.symbioflows.com/api/health
```

#### 2. API Connection Issues
```bash
# Test API connectivity
curl -X GET https://api.symbioflows.com/api/health
curl -X GET https://api.symbioflows.com/ai/health
```

#### 3. Environment Variables
```bash
# Verify environment variables
echo $VITE_API_URL
echo $NODE_ENV
echo $SUPABASE_URL
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export NODE_ENV=development

# Start with debug flags
python start_production_demo.py --debug
```

## 📈 Performance Optimization

### Caching Strategy
- **Redis**: Session storage and API caching
- **CDN**: Static assets (Vercel/Cloudflare)
- **Browser**: Cache headers for assets

### Database Optimization
- **Connection Pooling**: Configure Supabase connection limits
- **Query Optimization**: Use database indexes
- **Read Replicas**: For high-traffic scenarios

### AI Model Optimization
- **Model Quantization**: Reduce model size
- **Batch Processing**: Process multiple requests
- **GPU Acceleration**: For inference workloads

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Railway
        run: |
          railway up --service backend

  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Vercel
        run: |
          vercel --prod
```

## 📞 Support

### Emergency Contacts
- **System Admin**: admin@symbioflows.com
- **DevOps**: devops@symbioflows.com
- **AI Team**: ai-team@symbioflows.com

### Monitoring Alerts
- **Uptime**: UptimeRobot or Pingdom
- **Error Tracking**: Sentry
- **Performance**: New Relic or DataDog

---

**🎉 Your SymbioFlows production system is ready for deployment!**

Visit `https://symbioflows.com` to see your live application.
