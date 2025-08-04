# SymbioFlows Developer Quick-Start Guide

## 1. Introduction

Welcome to the SymbioFlows codebase! This guide will help you quickly understand the project structure, set up your development environment, and start contributing to the platform.

SymbioFlows is an advanced AI-powered industrial symbiosis platform that connects companies to optimize material flows in a circular economy. The platform leverages cutting-edge AI technologies including quantum-inspired algorithms, graph neural networks, and multi-agent systems.

## 2. Repository Structure

The repository is organized into several key directories:

```
SymbioFlows/
├── frontend/                 # React Frontend Application
├── backend/                  # Node.js + Python Backend
├── ai_service_flask/         # Flask-based AI services
├── ai_service/               # Core AI service modules
├── model_storage/            # Model storage and versioning
├── docs/                     # Documentation
├── data/                     # Data files
├── infrastructure/           # Infrastructure setup
├── k8s/                      # Kubernetes configuration
└── monitoring/               # Monitoring and alerting
```

## 3. Development Environment Setup

### 3.1 Prerequisites

- Node.js 18+
- Python 3.8+
- Docker (optional, for containerized development)
- Git

### 3.2 Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at http://localhost:5173

### 3.3 Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt

# Start backend server
npm start
```

The backend will be available at http://localhost:5000

### 3.4 AI Services Setup

```bash
# Navigate to AI service directory
cd ai_service_flask

# Install dependencies
pip install -r requirements.txt

# Start AI Gateway service
python ai_gateway.py
```

The AI Gateway will be available at http://localhost:8000

## 4. Key Files to Understand

### 4.1 Configuration Files

- `frontend/.env.local` - Frontend environment variables
- `backend/.env` - Backend environment variables
- `ai_service_flask/.env` - AI services environment variables

### 4.2 Entry Points

- `frontend/src/main.tsx` - Frontend entry point
- `backend/app.js` - Backend entry point
- `ai_service_flask/ai_gateway.py` - AI Gateway entry point

### 4.3 Core Components

- `frontend/src/components/` - React components
- `backend/ml_core/` - Machine learning core components
- `ai_service_flask/multi_hop_symbiosis_service.py` - Multi-hop symbiosis service

## 5. Development Workflow

### 5.1 Feature Development

1. Create a feature branch from `main`
2. Implement your changes
3. Write tests for your changes
4. Submit a pull request

### 5.2 Testing

- Frontend: `cd frontend && npm test`
- Backend: `cd backend && npm test`
- AI Services: `cd ai_service_flask && python -m pytest`

### 5.3 Code Style

- Frontend: ESLint and Prettier
- Backend: ESLint
- AI Services: PEP 8

## 6. AI Components Overview

### 6.1 AI Gateway

The AI Gateway (`ai_service_flask/ai_gateway.py`) serves as the central entry point for all AI services. It provides:

- Model inference endpoints
- Explainability features
- Hyperparameter optimization
- Health monitoring

### 6.2 Multi-Hop Symbiosis Service

The Multi-Hop Symbiosis Service (`ai_service_flask/multi_hop_symbiosis_service.py`) provides advanced network analysis for circular economy. It includes:

- Graph-based symbiosis detection
- Pattern recognition for industrial relationships
- Optimal path finding algorithms
- Impact and feasibility assessment

### 6.3 Other AI Services

- Advanced Analytics Service
- GNN Inference Service
- Federated Learning Service
- AI Pricing Service
- Logistics Service

## 7. Working with Models

### 7.1 Model Registry

The Model Registry manages AI model versions and metadata. To use it:

```python
from ml_core.utils import ModelRegistry

# Initialize registry
model_registry = ModelRegistry()

# Get model
model_info = model_registry.get_model('model_id')
```

### 7.2 Model Inference

To perform inference with a model:

```python
import torch

# Load model
model = model_info['model_class'](**model_info['model_params'])
model.load_state_dict(torch.load(model_info['model_path']))
model.eval()

# Perform inference
with torch.no_grad():
    output = model(input_tensor)
```

## 8. API Documentation

### 8.1 Backend API

The backend API documentation is available at:
- Development: http://localhost:5000/api/docs
- Production: https://api.symbioflows.com/api/docs

### 8.2 AI Gateway API

The AI Gateway API documentation is available at:
- Development: http://localhost:8000/docs
- Production: https://ai.symbioflows.com/docs

## 9. Deployment

### 9.1 Frontend Deployment

```bash
# Build frontend
cd frontend
npm run build

# Deploy to Vercel
vercel --prod
```

### 9.2 Backend Deployment

```bash
# Deploy to Railway
cd backend
railway up
```

### 9.3 AI Services Deployment

```bash
# Build Docker image
cd ai_service_flask
docker build -t symbioflows/ai-gateway .

# Deploy to container service
docker push symbioflows/ai-gateway
```

## 10. Additional Resources

- [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)
- [AI System Documentation](docs/COMPREHENSIVE_AI_SYSTEM_DOCUMENTATION.md)
- [API Documentation](docs/API.md)
- [Production Setup](docs/PRODUCTION_SETUP.md)

## 11. Getting Help

If you need help with the codebase, please:

1. Check the documentation in the `docs/` directory
2. Look for comments in the code
3. Contact the development team

Happy coding!