#!/bin/bash

# ISM Platform Environment Setup Script
# This script sets up the production environment with all necessary configurations

set -euo pipefail

# Configuration
NAMESPACE="ism-platform"
ENVIRONMENT="production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl is not configured or cluster is not accessible"
        exit 1
    fi
    
    # Check if helm is installed
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        kubectl create namespace $NAMESPACE
        log_success "Namespace $NAMESPACE created"
    else
        log_warning "Namespace $NAMESPACE already exists"
    fi
}

# Setup storage classes
setup_storage() {
    log_info "Setting up storage classes..."
    
    # Check if default storage class exists
    if ! kubectl get storageclass standard &> /dev/null; then
        log_warning "Default storage class 'standard' not found"
        log_info "Please ensure your cluster has a default storage class configured"
    else
        log_success "Storage class 'standard' found"
    fi
}

# Setup ingress controller
setup_ingress() {
    log_info "Setting up NGINX Ingress Controller..."
    
    # Check if ingress controller is already installed
    if kubectl get pods -n ingress-nginx &> /dev/null; then
        log_warning "NGINX Ingress Controller already installed"
        return
    fi
    
    # Add helm repository
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo update
    
    # Install ingress controller
    helm install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --set controller.service.type=LoadBalancer \
        --set controller.resources.requests.cpu=100m \
        --set controller.resources.requests.memory=128Mi \
        --set controller.resources.limits.cpu=200m \
        --set controller.resources.limits.memory=256Mi
    
    log_success "NGINX Ingress Controller installed"
}

# Setup cert-manager
setup_cert_manager() {
    log_info "Setting up cert-manager for SSL certificates..."
    
    # Check if cert-manager is already installed
    if kubectl get pods -n cert-manager &> /dev/null; then
        log_warning "cert-manager already installed"
        return
    fi
    
    # Install cert-manager
    helm repo add jetstack https://charts.jetstack.io
    helm repo update
    
    helm install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --create-namespace \
        --version v1.13.0 \
        --set installCRDs=true
    
    # Wait for cert-manager to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=300s
    
    log_success "cert-manager installed"
}

# Create cluster issuer
create_cluster_issuer() {
    log_info "Creating Let's Encrypt cluster issuer..."
    
    cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
    
    log_success "Cluster issuer created"
}

# Setup monitoring stack
setup_monitoring() {
    log_info "Setting up monitoring stack..."
    
    # Add Prometheus Operator repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus Operator
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=10Gi \
        --set grafana.adminPassword=admin \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=5Gi
    
    log_success "Monitoring stack installed"
}

# Create secrets
create_secrets() {
    log_info "Creating Kubernetes secrets..."
    
    # Check if secrets already exist
    if kubectl get secret ism-secrets -n $NAMESPACE &> /dev/null; then
        log_warning "Secret 'ism-secrets' already exists"
        return
    fi
    
    # Create secrets (you should replace these with actual values)
    kubectl create secret generic ism-secrets \
        --namespace $NAMESPACE \
        --from-literal=SUPABASE_URL="https://your-project.supabase.co" \
        --from-literal=SUPABASE_ANON_KEY="your-anon-key" \
        --from-literal=SUPABASE_SERVICE_ROLE_KEY="your-service-role-key" \
        --from-literal=JWT_SECRET="your-jwt-secret" \
        --from-literal=REDIS_PASSWORD="changeme" \
        --from-literal=GRAFANA_PASSWORD="admin" \
        --from-literal=DEEPSEEK_API_KEY="your-deepseek-api-key" \
        --from-literal=AWS_ACCESS_KEY_ID="your-aws-access-key" \
        --from-literal=AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
    
    log_success "Secrets created"
}

# Create configmap
create_configmap() {
    log_info "Creating ConfigMap..."
    
    # Check if configmap already exists
    if kubectl get configmap ism-config -n $NAMESPACE &> /dev/null; then
        log_warning "ConfigMap 'ism-config' already exists"
        return
    fi
    
    # Apply configmap
    kubectl apply -f k8s/configmap.yaml
    
    log_success "ConfigMap created"
}

# Setup backup storage
setup_backup_storage() {
    log_info "Setting up backup storage..."
    
    # Create S3 bucket for backups (if using AWS)
    log_info "Please ensure you have an S3 bucket named 'ism-backups' for storing backups"
    log_info "You can create it using: aws s3 mb s3://ism-backups"
    
    log_success "Backup storage setup instructions provided"
}

# Validate setup
validate_setup() {
    log_info "Validating environment setup..."
    
    # Check namespace
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        log_success "Namespace $NAMESPACE exists"
    else
        log_error "Namespace $NAMESPACE not found"
        return 1
    fi
    
    # Check secrets
    if kubectl get secret ism-secrets -n $NAMESPACE &> /dev/null; then
        log_success "Secret 'ism-secrets' exists"
    else
        log_error "Secret 'ism-secrets' not found"
        return 1
    fi
    
    # Check configmap
    if kubectl get configmap ism-config -n $NAMESPACE &> /dev/null; then
        log_success "ConfigMap 'ism-config' exists"
    else
        log_error "ConfigMap 'ism-config' not found"
        return 1
    fi
    
    # Check ingress controller
    if kubectl get pods -n ingress-nginx &> /dev/null; then
        log_success "NGINX Ingress Controller is running"
    else
        log_error "NGINX Ingress Controller not found"
        return 1
    fi
    
    # Check cert-manager
    if kubectl get pods -n cert-manager &> /dev/null; then
        log_success "cert-manager is running"
    else
        log_error "cert-manager not found"
        return 1
    fi
    
    # Check monitoring
    if kubectl get pods -n monitoring &> /dev/null; then
        log_success "Monitoring stack is running"
    else
        log_error "Monitoring stack not found"
        return 1
    fi
    
    log_success "Environment setup validation passed"
}

# Display next steps
display_next_steps() {
    log_success "Environment setup completed!"
    echo
    log_info "Next steps:"
    echo "1. Update the secrets with your actual values:"
    echo "   kubectl edit secret ism-secrets -n $NAMESPACE"
    echo
    echo "2. Update the ConfigMap with your domain:"
    echo "   kubectl edit configmap ism-config -n $NAMESPACE"
    echo
    echo "3. Deploy the application:"
    echo "   ./scripts/deploy-production.sh"
    echo
    echo "4. Access your application:"
    echo "   Frontend: https://ism.yourdomain.com"
    echo "   API: https://api.ism.yourdomain.com"
    echo "   Monitoring: https://grafana.ism.yourdomain.com"
    echo
    log_warning "Remember to:"
    echo "- Update DNS records to point to your ingress controller"
    echo "- Configure your domain in the ingress configuration"
    echo "- Set up proper monitoring alerts"
    echo "- Configure backup retention policies"
}

# Main setup function
main() {
    log_info "Starting ISM Platform environment setup for $ENVIRONMENT"
    
    # Check prerequisites
    check_prerequisites
    
    # Create namespace
    create_namespace
    
    # Setup storage
    setup_storage
    
    # Setup ingress controller
    setup_ingress
    
    # Setup cert-manager
    setup_cert_manager
    
    # Create cluster issuer
    create_cluster_issuer
    
    # Setup monitoring
    setup_monitoring
    
    # Create secrets
    create_secrets
    
    # Create configmap
    create_configmap
    
    # Setup backup storage
    setup_backup_storage
    
    # Validate setup
    if validate_setup; then
        display_next_steps
    else
        log_error "Environment setup validation failed"
        exit 1
    fi
}

# Run main function
main "$@" 