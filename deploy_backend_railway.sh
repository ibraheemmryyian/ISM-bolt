#!/bin/bash

# Quick Railway Backend Deployment Script
echo "🚀 Deploying SymbioFlows Backend to Railway..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Navigate to backend directory
cd backend

# Login to Railway (if not already logged in)
echo "🔐 Logging into Railway..."
railway login

# Initialize Railway project (if not already initialized)
if [ ! -f "railway.json" ]; then
    echo "📁 Initializing Railway project..."
    railway init
fi

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

# Get the deployment URL
echo "🔗 Getting deployment URL..."
DEPLOYMENT_URL=$(railway domain)
echo "✅ Backend deployed at: $DEPLOYMENT_URL"

# Update frontend environment
echo "📝 Updating frontend environment..."
cd ../frontend

# Create backup of current .env.production
cp .env.production .env.production.backup

# Update the backend URL in frontend environment
sed -i "s|VITE_API_URL=.*|VITE_API_URL=$DEPLOYMENT_URL|g" .env.production
sed -i "s|VITE_WS_URL=.*|VITE_WS_URL=wss://$(echo $DEPLOYMENT_URL | sed 's|https://||')|g" .env.production
sed -i "s|VITE_AI_PREVIEW_URL=.*|VITE_AI_PREVIEW_URL=$DEPLOYMENT_URL/api|g" .env.production
sed -i "s|VITE_BACKEND_URL=.*|VITE_BACKEND_URL=$DEPLOYMENT_URL|g" .env.production

echo "✅ Frontend environment updated with new backend URL"
echo ""
echo "🔗 Your backend is now live at: $DEPLOYMENT_URL"
echo "📊 Railway Dashboard: https://railway.app/dashboard"
echo ""
echo "🧪 Test your backend:"
echo "curl $DEPLOYMENT_URL/health"
echo ""
echo "🔄 Next steps:"
echo "1. Deploy frontend to Vercel"
echo "2. Test all functionality"
echo "3. Set up custom domain (optional)" 