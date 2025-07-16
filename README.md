# ISM [AI] - Industrial Symbiosis Marketplace

Advanced AI-powered platform for industrial symbiosis, connecting companies to create sustainable waste-to-resource networks.

## 🌟 Features

### 🤖 AI-Powered Matching
- **Revolutionary AI Engine**: Advanced machine learning for buyer-seller compatibility
- **Graph-Based Symbiosis**: Network analysis for industrial symbiosis opportunities
- **Explainable AI**: Transparent AI decision-making with detailed explanations
- **Active Learning**: Continuous improvement from user feedback
- **Real-time Recommendations**: Dynamic AI-powered suggestions

### 🏭 Industrial Focus
- **Waste-to-Resource Matching**: Connect waste producers with resource consumers
- **Carbon Footprint Analysis**: Comprehensive environmental impact assessment
- **Cost Optimization**: Logistics and financial analysis for symbiosis projects
- **Regulatory Compliance**: Built-in compliance checking and reporting

### 🔒 Enterprise Security
- **Multi-layer Authentication**: Secure user and company authentication
- **Role-based Access Control**: Granular permissions and admin controls
- **Data Encryption**: End-to-end encryption for sensitive data
- **Audit Trails**: Comprehensive logging and monitoring

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.8+
- Supabase account
- Git

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ISM-AI-Platform
```

### 2. Install Dependencies
```bash
# Backend
cd backend
npm install

# Frontend
cd ../frontend
npm install

# Python dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
```bash
# Run automated environment setup
.\scripts\audit-env.ps1 fix

# Or manually configure
cp backend/env.example backend/.env
cp frontend/.env.example frontend/.env
# Edit .env files with your configuration
```

### 4. Database Setup
```bash
# Run database migrations
.\scripts\migrate-database.ps1

# Import sample data (optional)
python import_companies.py
```

### 5. Start Development Servers
```bash
# Backend (Terminal 1)
cd backend
npm run dev

# Frontend (Terminal 2)
cd frontend
npm run dev
```

### 6. Access the Platform
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/health
- **Admin Access**: http://localhost:5173/admin (Password: NA10EN)

## 🧪 Testing

### Backend Tests
```bash
cd backend

# Unit tests
npm test

# Integration tests
npm run test:integration

# Test coverage
npm run test:coverage

# Health check
npm run health-check
```

### Frontend Tests
```bash
cd frontend

# Unit tests
npm test

# E2E tests
npm run test:e2e
```

### Security Audit
```bash
# Environment variable audit
.\scripts\audit-env.ps1 check

# Generate security report
.\scripts\audit-env.ps1 report

# Fix common issues
.\scripts\audit-env.ps1 fix
```

## 🏗️ Architecture

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS + shadcn/ui
- **State Management**: React Context + Zustand
- **Build Tool**: Vite
- **Testing**: Vitest + React Testing Library

### Backend (Node.js + Express)
- **Framework**: Express.js with TypeScript
- **Database**: Supabase (PostgreSQL)
- **Authentication**: Supabase Auth + JWT
- **AI Integration**: Python subprocess + REST APIs
- **Testing**: Jest + Supertest

### AI Services (Python)
- **Matching Engine**: Advanced ML algorithms
- **Carbon Calculator**: Environmental impact analysis
- **Waste Tracking**: Material flow optimization
- **Analytics**: Business intelligence and insights

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes manifests
- **Monitoring**: Prometheus + Grafana
- **CI/CD**: GitHub Actions

## 📚 Documentation

### Setup Guides
- [Environment Configuration](docs/environment-setup.md)
- [Database Migration](docs/database-migration.md)
- [Deployment Guide](docs/deployment.md)

### API Documentation
- [Backend API Reference](backend/README.md)
- [AI Service Endpoints](ai_service_flask/README.md)
- [Health Check Endpoints](docs/health-checks.md)

### Development
- [Contributing Guidelines](docs/contributing.md)
- [Testing Strategy](docs/testing.md)
- [Security Best Practices](docs/security.md)

## 🔧 Scripts

### Automation Scripts
```bash
# Database migration
.\scripts\migrate-database.ps1 [status|validate|backup|rollback]

# Environment audit
.\scripts\audit-env.ps1 [check|fix|report]

# Production deployment
.\scripts\deploy-production.sh

# Environment setup
.\scripts\setup-environment.sh
```

### Development Scripts
```bash
# Backend
npm run dev          # Development server
npm run build        # Production build
npm run test         # Run tests
npm run lint         # Code linting

# Frontend
npm run dev          # Development server
npm run build        # Production build
npm run preview      # Preview build
npm run test         # Run tests
```

## 🚀 Deployment

### Production Checklist
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] Security audit passed
- [ ] Tests passing
- [ ] Health checks verified
- [ ] Monitoring configured
- [ ] Backup strategy implemented

### 🌐 Environment Variables Reference

| Variable Name                  | Description                                      | Example/Default Value                | Used By                |
|-------------------------------|--------------------------------------------------|--------------------------------------|------------------------|
| SUPABASE_URL                  | Supabase project URL                             | https://xxxx.supabase.co             | Backend, Frontend      |
| SUPABASE_KEY / SUPABASE_ANON_KEY | Supabase anon/public key                      | (long string)                        | Backend, Frontend      |
| SUPABASE_SERVICE_ROLE_KEY     | Supabase service role key (secure)               | (long string)                        | Backend                |
| DEEPSEEK_API_KEY              | DeepSeek AI API key                              | sk-xxxx...                           | Backend, Frontend      |
| DEEPSEEK_R1_API_KEY           | DeepSeek R1 advanced AI key                      | sk-xxxx...                           | Backend                |
| MP_API_KEY                    | Materials Project API key                        | zSFjfpRg6m...                        | Backend                |
| NEXT_GEN_MATERIALS_API_KEY    | Next Gen Materials API key                       | zSFjfpRg6m...                        | Backend                |
| MATERIALSBERT_ENABLED         | Enable MaterialsBERT service                     | true/false                           | Backend                |
| JWT_SECRET                    | JWT signing secret                               | (secure random string)               | Backend                |
| SESSION_SECRET                | Session secret                                   | (secure random string)               | Backend                |
| FREIGHTOS_API_KEY             | Freightos logistics API key                      | (string)                             | Backend                |
| FREIGHTOS_SECRET_KEY          | Freightos secret key                             | (string)                             | Backend                |
| NEWS_API_KEY                  | News API key for market intelligence             | (string)                             | Backend                |
| VITE_SUPABASE_URL             | Supabase URL for frontend                        | https://xxxx.supabase.co             | Frontend               |
| VITE_SUPABASE_ANON_KEY        | Supabase anon/public key for frontend            | (long string)                        | Frontend               |
| VITE_API_URL                  | Backend API base URL for frontend                | http://localhost:3000                | Frontend               |
| VITE_WS_URL                   | WebSocket URL for frontend                       | ws://localhost:3000                  | Frontend               |
| VITE_AI_PREVIEW_URL           | AI preview API URL for frontend                  | http://localhost:5001/api            | Frontend               |
| LOG_LEVEL                     | Logging level                                    | info                                 | Backend                |
| LOG_FILE                      | Log file path                                    | logs/app.log                         | Backend                |
| RATE_LIMIT_WINDOW_MS          | Rate limit window (ms)                           | 900000                               | Backend                |
| RATE_LIMIT_MAX_REQUESTS       | Max requests per window                          | 100                                  | Backend                |
| BACKEND_URL                   | Backend service URL                              | http://localhost:5001                | Backend                |
| FRONTEND_URL                  | Frontend service URL                             | http://localhost:5173                | Backend                |

> **Note:** Never commit your actual `.env` files or secrets to version control. Always use secure, unique values for all secrets and API keys.

---

### Deployment Options

#### Docker Compose
```bash
docker-compose -f docker-compose.prod.yml up -d
```

#### Kubernetes
```bash
kubectl apply -f k8s/
```

#### Cloud Platforms
- **Vercel**: Frontend deployment
- **Railway/Heroku**: Backend deployment
- **Supabase**: Database and authentication

## 🔒 Security

### Security Features
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: Protection against abuse
- **CORS Configuration**: Secure cross-origin requests
- **Helmet**: Security headers
- **JWT Tokens**: Secure authentication
- **Environment Auditing**: Automated security checks

### Security Audit
```bash
# Run security audit
.\scripts\audit-env.ps1 check

# View security report
.\scripts\audit-env.ps1 report

# Fix security issues
.\scripts\audit-env.ps1 fix
```

## 📊 Monitoring

### Health Checks
- **Backend Health**: `/health` - Database connectivity
- **API Health**: `/api/health` - Service status
- **Frontend Health**: Frontend status monitoring

### Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and alerting
- **Sentry**: Error tracking and performance monitoring

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Run the test suite
6. Submit a pull request

### Code Standards
- **TypeScript**: Strict type checking
- **ESLint**: Code quality enforcement
- **Prettier**: Code formatting
- **Conventional Commits**: Commit message standards

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: Check the docs folder
- **Issues**: Create an issue in the repository
- **Discussions**: Use GitHub Discussions
- **Security**: Report security issues privately

### Common Issues
- **Environment Setup**: Run `.\scripts\audit-env.ps1 fix`
- **Database Issues**: Run `.\scripts\migrate-database.ps1`
- **Build Problems**: Check Node.js and Python versions
- **Authentication**: Verify Supabase configuration

## 🎯 Roadmap

### Upcoming Features
- [ ] Advanced AI matching algorithms
- [ ] Real-time collaboration tools
- [ ] Mobile application
- [ ] Blockchain integration
- [ ] Advanced analytics dashboard
- [ ] Multi-language support

### Performance Improvements
- [ ] Database optimization
- [ ] Caching strategies
- [ ] CDN integration
- [ ] Load balancing
- [ ] Microservices architecture

---

**Built with ❤️ for sustainable industrial symbiosis**