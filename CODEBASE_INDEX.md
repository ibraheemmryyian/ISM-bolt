# SymbioFlows Codebase Index

## 1. Project Overview

SymbioFlows is an advanced AI-powered industrial symbiosis platform designed to connect companies and optimize material flows in a circular economy. The platform uses cutting-edge AI technologies including quantum-inspired algorithms, graph neural networks, and multi-agent systems to identify optimal material matches and symbiosis opportunities.

## 2. Core Architecture

### 2.1 Frontend Layer
- **Technology**: React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui
- **Key Components**: 
  - User Interface Components (`frontend/src/components/`)
  - State Management with Zustand
  - Routing with React Router
  - Real-time Updates via WebSocket

### 2.2 Backend Layer
- **Technology**: Node.js, Express, Python AI Services, Supabase
- **Key Components**:
  - API Gateway & Authentication
  - Business Logic Services
  - Database Operations
  - External API Integrations

### 2.3 AI Services Layer
- **Technology**: Python, PyTorch, Transformers, Flask
- **Key Components**:
  - Adaptive AI Onboarding
  - AI Listings Generator
  - AI Matching Engine
  - Advanced Analytics Engine
  - Materials Analysis Engine

### 2.4 Data Layer
- **Technology**: Supabase (PostgreSQL)
- **Key Components**:
  - User Management
  - Company Profiles
  - Materials Database
  - Matches & Transactions
  - AI Insights & Analytics

## 3. AI Services

### 3.1 AI Gateway (`ai_service_flask/ai_gateway.py`)
- **Purpose**: Central entry point for AI services
- **Features**:
  - Model inference endpoints
  - Explainability features
  - Hyperparameter optimization
  - Health monitoring

### 3.2 Multi-Hop Symbiosis Service (`ai_service_flask/multi_hop_symbiosis_service.py`)
- **Purpose**: Advanced network analysis for circular economy
- **Features**:
  - Graph-based symbiosis detection
  - Pattern recognition for industrial relationships
  - Optimal path finding algorithms
  - Impact and feasibility assessment

### 3.3 Advanced Analytics Service (`ai_service_flask/advanced_analytics_service.py`)
- **Purpose**: Business intelligence and insights
- **Features**:
  - Trend analysis
  - Predictive modeling
  - Performance metrics

### 3.4 GNN Inference Service (`ai_service_flask/gnn_inference_service.py`)
- **Purpose**: Graph Neural Network inference
- **Features**:
  - Material flow network modeling
  - Node and edge feature processing
  - Graph-based predictions

### 3.5 Federated Learning Service (`ai_service_flask/federated_learning_service.py`)
- **Purpose**: Distributed AI model training
- **Features**:
  - Privacy-preserving learning
  - Model aggregation
  - Distributed training

## 4. Backend Services

### 4.1 Revolutionary AI Matching Intelligence
- **Purpose**: Advanced material matching
- **Components**:
  - Quantum-inspired matching engine
  - Brain-inspired pattern matching
  - Evolutionary optimization
  - Swarm intelligence
  - Neuro-symbolic reasoning

### 4.2 AI Pricing Services
- **Purpose**: Dynamic pricing for materials
- **Components**:
  - Market analysis
  - Value assessment
  - Price optimization

### 4.3 Logistics Services
- **Purpose**: Transportation and logistics optimization
- **Components**:
  - Route optimization
  - Cost calculation
  - Carbon footprint assessment

## 5. Frontend Components

### 5.1 Core UI Components
- Adaptive AI Onboarding
- Material Management
- Match Discovery
- Transaction Processing
- Analytics Dashboard

### 5.2 Advanced Features
- Real-time notifications
- Interactive visualizations
- Multi-step workflows
- Responsive design

## 6. Database Architecture

### 6.1 Core Tables
- Users & Authentication
- Companies & Profiles
- Materials & Listings
- Matches & Transactions
- AI Insights & Analytics

### 6.2 Relationships
- One-to-one and one-to-many relationships
- Complex material flow connections
- Transaction history tracking

## 7. Infrastructure & DevOps

### 7.1 Deployment
- Frontend: Vercel
- Backend: Railway/Render
- Database: Supabase
- Containerization: Docker

### 7.2 Monitoring & Maintenance
- Health checks
- Performance monitoring
- Error tracking
- Backup strategy

## 8. AI Models & Algorithms

### 8.1 Neural Architecture Components
- Multi-Head Attention System
- Transformer-XL System
- Advanced GNN System

### 8.2 Quantum-Inspired Systems
- Quantum-Inspired Optimizer
- Quantum-Inspired Search

### 8.3 Brain-Inspired Processing
- Cortical Column Model
- Hippocampal Memory System
- Basal Ganglia System

### 8.4 Advanced AI Techniques
- Evolutionary Neural Systems
- Multi-Agent Swarm Intelligence
- Neuro-Symbolic Reasoning
- Advanced Meta-Learning
- Hyperdimensional Computing

## 9. Security & Compliance

### 9.1 Authentication & Authorization
- Multi-factor Authentication
- JWT Tokens
- Role-based Access Control

### 9.2 Data Protection
- Encryption
- API Security
- HTTPS
- Environment Variables

### 9.3 Privacy & Compliance
- GDPR Compliance
- Data Minimization
- Audit Logging
- Data Retention

## 10. Key Files & Directories

### 10.1 AI Services
- `ai_service_flask/` - Flask-based AI services
- `ai_service/` - Core AI service modules
- `model_storage/` - Model storage and versioning

### 10.2 Backend
- `backend/` - Main backend services
- `backend/ml_core/` - Machine learning core components
- `backend/utils/` - Utility functions

### 10.3 Frontend
- `frontend/src/components/` - React components
- `frontend/src/lib/` - Frontend libraries and services

### 10.4 Infrastructure
- `k8s/` - Kubernetes configuration
- `infrastructure/` - Infrastructure setup
- `monitoring/` - Monitoring and alerting

### 10.5 Documentation
- `docs/` - Project documentation
- `docs/ARCHITECTURE_OVERVIEW.md` - Architecture documentation
- `docs/COMPREHENSIVE_AI_SYSTEM_DOCUMENTATION.md` - AI system documentation