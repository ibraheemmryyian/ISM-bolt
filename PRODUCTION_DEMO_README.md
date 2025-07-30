# SymbioFlows Production Demo System

## 🚀 Complete User Flow Demo

This production-ready demo system showcases the complete SymbioFlows user journey:

1. **User Signup & Authentication**
2. **AI-Powered Onboarding**
3. **Material Listings Generation**
4. **Intelligent Matchmaking**
5. **Real-time Analytics & Insights**

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │  AI Services    │
│   (React/Vite)  │◄──►│   (Node.js)     │◄──►│  (Python/Flask) │
│   Port: 5173    │    │   Port: 3000    │    │  Port: 5000-6   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Supabase      │    │     Redis       │    │   Monitoring    │
│   Database      │    │     Cache       │    │  (Prometheus)   │
│   (PostgreSQL)  │    │   Port: 6379    │    │  Port: 9090     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 AI Microservices

### Core AI Services (Ports 5000-5006)
- **AI Gateway** (5000): Main orchestration and routing
- **GNN Inference** (5001): Graph Neural Network analysis
- **Federated Learning** (5002): Distributed model training
- **Multi-Hop Symbiosis** (5003): Complex network analysis
- **Advanced Analytics** (5004): Predictive modeling
- **AI Pricing** (5005): Dynamic pricing intelligence
- **Logistics Service** (5006): Supply chain optimization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- Docker (optional, for containerized deployment)
- Git

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd SymbioFlows

# Copy environment template
cp .env.example .env

# Edit .env with your actual values
# Required: SUPABASE_URL, SUPABASE_ANON_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY
```

### 2. Start Production Demo

#### Option A: Windows (Recommended)
```cmd
# Run the production demo script
start_production_demo.bat
```

#### Option B: Python Script
```bash
# Install dependencies and start all services
python start_production_demo.py
```

#### Option C: Docker Compose
```bash
# Start all services in containers
docker-compose -f docker-compose.production.yml up -d
```

### 3. Access the System

Once all services are running, access the system at:

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3000
- **AI Services**: http://localhost:5000
- **API Documentation**: http://localhost:3000/api-docs
- **Monitoring**: http://localhost:9090 (Prometheus)
- **Dashboard**: http://localhost:3001 (Grafana)

## 🎭 Demo User Flow

### Step 1: User Signup
1. Visit http://localhost:5173
2. Click "Get Started"
3. Fill out company information:
   - Company Name: EcoTech Manufacturing
   - Industry: Electronics Manufacturing
   - Waste Streams: Electronic waste, plastic components
   - Sustainability Goals: Zero waste by 2030

### Step 2: AI Onboarding
1. Complete the AI-powered questionnaire
2. Answer questions about:
   - Company size and location
   - Production processes
   - Current waste management
   - Material requirements

### Step 3: Material Listings Generation
1. AI automatically generates material listings
2. Review AI-generated materials
3. Customize descriptions and quantities

### Step 4: Matchmaking
1. AI finds potential partners
2. View match recommendations
3. Explore symbiosis opportunities

## 🔧 Configuration

### Environment Variables

#### Required
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key
DEEPSEEK_API_KEY=your-deepseek-api-key
OPENAI_API_KEY=your-openai-api-key
```

#### Optional
```bash
JWT_SECRET=your-jwt-secret
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

### Service Configuration

#### Backend API (Node.js)
- Port: 3000
- Environment: Production
- Database: Supabase PostgreSQL
- Cache: Redis

#### AI Services (Python/Flask)
- Gateway Port: 5000
- Individual Services: 5001-5006
- Model Storage: Local + Cloud
- Monitoring: Prometheus metrics

#### Frontend (React/Vite)
- Port: 5173
- Build Tool: Vite
- UI Framework: React + Tailwind CSS
- State Management: React Context + Supabase

## 📊 Monitoring & Analytics

### Prometheus Metrics
- Request rates and latencies
- Error rates and types
- AI model performance
- Resource utilization

### Grafana Dashboards
- System health overview
- AI service performance
- User activity metrics
- Business intelligence

### Health Checks
All services include health check endpoints:
- Backend: `/api/health`
- AI Gateway: `/health`
- Frontend: `/health`

## 🔒 Security Features

### Authentication & Authorization
- Supabase Auth integration
- JWT token management
- Role-based access control
- Session management

### API Security
- Rate limiting
- CORS configuration
- Input validation
- SQL injection prevention

### Data Protection
- Encrypted data transmission
- Secure API keys management
- Audit logging
- GDPR compliance

## 🚀 Production Deployment

### Docker Deployment
```bash
# Build and deploy all services
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Scale services
docker-compose -f docker-compose.production.yml up -d --scale ai-gateway=3
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n symbioflows

# Access services
kubectl port-forward svc/frontend 5173:5173
```

### Cloud Deployment
- **Railway**: `railway up`
- **Heroku**: `heroku container:push web`
- **AWS ECS**: Use provided CloudFormation templates
- **Google Cloud Run**: `gcloud run deploy`

## 🧪 Testing

### Automated Tests
```bash
# Run backend tests
cd backend && npm test

# Run frontend tests
cd frontend && npm test

# Run AI services tests
cd ai_service_flask && python -m pytest
```

### Manual Testing
1. **User Flow Test**: Complete signup → onboarding → listings → matches
2. **API Test**: Use Postman collection in `/docs/`
3. **Performance Test**: Load test with Artillery
4. **Security Test**: OWASP ZAP scan

## 📈 Performance Optimization

### Caching Strategy
- Redis for session storage
- CDN for static assets
- Browser caching headers
- API response caching

### Database Optimization
- Connection pooling
- Query optimization
- Indexing strategy
- Read replicas

### AI Model Optimization
- Model quantization
- Batch processing
- GPU acceleration
- Model serving optimization

## 🔧 Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check logs
tail -f logs/production_demo.log

# Check ports
netstat -tulpn | grep :3000
netstat -tulpn | grep :5000
netstat -tulpn | grep :5173
```

#### Database Connection Issues
```bash
# Test Supabase connection
curl -H "apikey: $SUPABASE_ANON_KEY" \
     -H "Authorization: Bearer $SUPABASE_ANON_KEY" \
     "$SUPABASE_URL/rest/v1/"
```

#### AI Services Issues
```bash
# Check AI service health
curl http://localhost:5000/health

# Check model availability
ls -la models/
ls -la model_storage/
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export NODE_ENV=development

# Start with debug flags
python start_production_demo.py --debug
```

## 📚 Documentation

### API Documentation
- Swagger UI: http://localhost:3000/api-docs
- OpenAPI Spec: http://localhost:3000/api-docs.json
- Postman Collection: `/docs/SymbioFlows_API.postman_collection.json`

### Architecture Documentation
- System Architecture: `/docs/ARCHITECTURE.md`
- AI System Design: `/docs/AI_SYSTEM_DESIGN.md`
- Database Schema: `/docs/DATABASE_SCHEMA.md`

### User Guides
- User Manual: `/docs/USER_MANUAL.md`
- Admin Guide: `/docs/ADMIN_GUIDE.md`
- API Reference: `/docs/API_REFERENCE.md`

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt
npm install --dev

# Run development server
npm run dev:backend
npm run dev:frontend
python ai_service_flask/ai_gateway.py --dev
```

### Code Standards
- Python: Black, Flake8, MyPy
- JavaScript: ESLint, Prettier
- TypeScript: Strict mode enabled
- Git: Conventional commits

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: Check `/docs/` directory
- **Issues**: Create GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: support@symbioflows.com

### Emergency Contacts
- **System Admin**: admin@symbioflows.com
- **AI Team**: ai-team@symbioflows.com
- **DevOps**: devops@symbioflows.com

---

**🎉 Ready to revolutionize the circular economy with AI-powered symbiosis!** 