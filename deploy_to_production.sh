#!/bin/bash

# SymbioFlows Production Deployment Script
# This script helps deploy the entire system to production

set -e  # Exit on any error

echo "🚀 SymbioFlows Production Deployment"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed"
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed"
        exit 1
    fi
    
    if ! command -v git &> /dev/null; then
        print_error "git is not installed"
        exit 1
    fi
    
    print_status "All dependencies are installed ✓"
}

# Build frontend
build_frontend() {
    print_status "Building frontend..."
    cd frontend
    
    # Install dependencies
    npm install
    
    # Build for production
    npm run build
    
    print_status "Frontend build completed ✓"
    cd ..
}

# Test frontend
test_frontend() {
    print_status "Testing frontend build..."
    cd frontend
    
    # Start preview server
    npm run preview &
    PREVIEW_PID=$!
    
    # Wait for server to start
    sleep 5
    
    # Test main routes
    curl -f http://localhost:4173/ > /dev/null && print_status "Home page ✓"
    curl -f http://localhost:4173/dashboard > /dev/null && print_status "Dashboard route ✓"
    curl -f http://localhost:4173/marketplace > /dev/null && print_status "Marketplace route ✓"
    
    # Kill preview server
    kill $PREVIEW_PID
    
    cd ..
    print_status "Frontend tests completed ✓"
}

# Test backend
test_backend() {
    print_status "Testing backend..."
    cd backend
    
    # Install dependencies
    npm install
    
    # Run tests
    npm test
    
    print_status "Backend tests completed ✓"
    cd ..
}

# Check environment files
check_environment() {
    print_status "Checking environment configuration..."
    
    # Check frontend environment
    if [ ! -f "frontend/.env.production" ]; then
        print_error "frontend/.env.production not found"
        exit 1
    fi
    
    # Check backend environment
    if [ ! -f "backend/.env.production" ]; then
        print_error "backend/.env.production not found"
        exit 1
    fi
    
    # Check vercel.json
    if [ ! -f "frontend/vercel.json" ]; then
        print_error "frontend/vercel.json not found"
        exit 1
    fi
    
    print_status "Environment configuration checked ✓"
}

# Deploy to Vercel
deploy_frontend() {
    print_status "Deploying frontend to Vercel..."
    
    if ! command -v vercel &> /dev/null; then
        print_warning "Vercel CLI not installed. Please install it first:"
        echo "npm i -g vercel"
        echo "Then run: vercel login"
        return 1
    fi
    
    cd frontend
    
    # Deploy to Vercel
    vercel --prod
    
    cd ..
    print_status "Frontend deployment completed ✓"
}

# Deploy backend (Railway example)
deploy_backend() {
    print_status "Deploying backend to Railway..."
    
    if ! command -v railway &> /dev/null; then
        print_warning "Railway CLI not installed. Please install it first:"
        echo "npm i -g @railway/cli"
        echo "Then run: railway login"
        return 1
    fi
    
    cd backend
    
    # Deploy to Railway
    railway up
    
    cd ..
    print_status "Backend deployment completed ✓"
}

# Health check
health_check() {
    print_status "Performing health checks..."
    
    # Wait for deployment to complete
    sleep 30
    
    # Test frontend
    if curl -f https://symbioflows.com/ > /dev/null; then
        print_status "Frontend health check passed ✓"
    else
        print_error "Frontend health check failed"
    fi
    
    # Test backend (if deployed)
    if curl -f https://api.symbioflows.com/health > /dev/null; then
        print_status "Backend health check passed ✓"
    else
        print_warning "Backend health check failed (may not be deployed yet)"
    fi
}

# Main deployment process
main() {
    echo "Starting deployment process..."
    
    check_dependencies
    check_environment
    build_frontend
    test_frontend
    test_backend
    
    echo ""
    print_status "Pre-deployment checks completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Deploy frontend: ./deploy_to_production.sh --frontend"
    echo "2. Deploy backend: ./deploy_to_production.sh --backend"
    echo "3. Deploy all: ./deploy_to_production.sh --all"
    echo ""
}

# Handle command line arguments
case "${1:-}" in
    --frontend)
        deploy_frontend
        health_check
        ;;
    --backend)
        deploy_backend
        health_check
        ;;
    --all)
        deploy_frontend
        deploy_backend
        health_check
        ;;
    --test)
        check_dependencies
        check_environment
        build_frontend
        test_frontend
        test_backend
        ;;
    *)
        main
        ;;
esac

echo ""
print_status "Deployment script completed!"
echo ""
echo "🔗 Production URLs:"
echo "Frontend: https://symbioflows.com"
echo "Backend: https://api.symbioflows.com"
echo ""
echo "📊 Monitoring:"
echo "Vercel Dashboard: https://vercel.com/dashboard"
echo "Railway Dashboard: https://railway.app/dashboard"
echo "" 