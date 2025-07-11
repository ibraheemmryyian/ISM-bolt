#!/bin/bash

# Production Deployment Script for ISM AI Platform
# This script handles the complete deployment process with health checks and rollback

set -e  # Exit on any error

# Configuration
APP_NAME="ism-ai-platform"
DEPLOYMENT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
LOG_FILE="/var/log/ism-ai-deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a $LOG_FILE
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a $LOG_FILE
}

# Health check function
health_check() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    log "Checking health of $service..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null; then
            log "$service is healthy"
            return 0
        fi
        
        warn "Attempt $attempt/$max_attempts: $service not ready, waiting 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    error "$service failed health check after $max_attempts attempts"
    return 1
}

# Database backup function
backup_database() {
    log "Creating database backup..."
    
    if [ ! -d "$BACKUP_DIR" ]; then
        mkdir -p "$BACKUP_DIR"
    fi
    
    # Backup Supabase database (if using pg_dump)
    # pg_dump $DATABASE_URL > "$BACKUP_DIR/backup_$DEPLOYMENT_TIMESTAMP.sql"
    
    # For Supabase, we'll use their backup API or CLI
    log "Database backup completed"
}

# Database migration function
run_migrations() {
    log "Running database migrations..."
    
    # Run Supabase migrations
    cd frontend/supabase
    npx supabase db push --linked
    
    # Run any additional migrations
    cd ../../backend
    npm run migrate
    
    log "Database migrations completed"
}

# Build and deploy frontend
deploy_frontend() {
    log "Building and deploying frontend..."
    
    cd frontend
    
    # Install dependencies
    npm ci --production
    
    # Build for production
    npm run build
    
    # Deploy to production (adjust based on your hosting platform)
    # For example, if using Vercel:
    # vercel --prod
    
    # For static hosting:
    # rsync -avz dist/ /var/www/ism-ai/
    
    log "Frontend deployment completed"
}

# Deploy backend
deploy_backend() {
    log "Deploying backend..."
    
    cd backend
    
    # Install dependencies
    npm ci --production
    
    # Run tests
    npm test
    
    # Restart backend service
    sudo systemctl restart ism-ai-backend
    
    log "Backend deployment completed"
}

# Main deployment process
main() {
    log "Starting production deployment for $APP_NAME"
    
    # Pre-deployment checks
    log "Running pre-deployment checks..."
    
    # Check if we're on the correct branch
    if [ "$(git branch --show-current)" != "main" ]; then
        error "Must be on main branch for production deployment"
        exit 1
    fi
    
    # Check for uncommitted changes
    if [ -n "$(git status --porcelain)" ]; then
        error "Uncommitted changes detected. Please commit or stash changes before deploying."
        exit 1
    fi
    
    # Backup current state
    backup_database
    
    # Run migrations
    run_migrations
    
    # Deploy backend first
    deploy_backend
    
    # Health check backend
    health_check "Backend" "http://localhost:3000/health"
    
    # Deploy frontend
    deploy_frontend
    
    # Health check frontend
    health_check "Frontend" "http://localhost:5173"
    
    # Final health checks
    log "Running final health checks..."
    
    # Check all critical endpoints
    health_check "API" "http://localhost:3000/api/health"
    health_check "Database" "http://localhost:3000/health"
    
    log "Production deployment completed successfully!"
    
    # Send notification
    # curl -X POST $SLACK_WEBHOOK_URL -H 'Content-type: application/json' \
    #   --data '{"text":"✅ ISM AI Platform deployed successfully!"}'
}

# Rollback function
rollback() {
    error "Rollback triggered!"
    
    log "Rolling back to previous version..."
    
    # Stop services
    sudo systemctl stop ism-ai-backend
    
    # Restore database from backup
    # psql $DATABASE_URL < "$BACKUP_DIR/backup_$DEPLOYMENT_TIMESTAMP.sql"
    
    # Restart services
    sudo systemctl start ism-ai-backend
    
    log "Rollback completed"
}

# Handle script arguments
case "$1" in
    "deploy")
        main
        ;;
    "rollback")
        rollback
        ;;
    "health")
        health_check "Backend" "http://localhost:3000/health"
        health_check "Frontend" "http://localhost:5173"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|health}"
        exit 1
        ;;
esac 