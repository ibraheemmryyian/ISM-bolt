# Industrial AI Marketplace Backend

Advanced Node.js backend for the Industrial AI Marketplace with comprehensive AI features, security, and monitoring.

## 🚀 Features

- **AI-Powered Matching**: Revolutionary AI engine for buyer-seller compatibility
- **Graph-Based Symbiosis**: Network analysis for industrial symbiosis opportunities
- **Explainable AI**: Transparent AI decision-making with detailed explanations
- **Active Learning**: Continuous improvement from user feedback
- **Real-time Recommendations**: Dynamic AI-powered suggestions
- **Security**: Helmet, CORS, rate limiting, input validation
- **Error Monitoring**: Sentry integration for production monitoring
- **Comprehensive Testing**: Jest framework with full API coverage

## 📋 Prerequisites

- Node.js 18+ 
- py 3.8+ with required packages
- npm or yarn

## 🛠️ Installation

1. **Clone and navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Install py dependencies**
   ```bash
   pip install -r ../requirements.txt
   ```

## 🚀 Running the Server

### Development
```bash
npm run dev
```

### Production
```bash
npm start
```

### Testing
```bash
npm test
npm run test:watch
```

## 📚 API Documentation

### Health Check
```http
GET /api/health
```

### AI Inference for Onboarding
```http
POST /api/ai-infer-listings
Content-Type: application/json

{
  "companyName": "Steel Corp",
  "industry": "Manufacturing",
  "products": "Steel products",
      "location": "Amman",
  "productionVolume": "1000 tons/year",
  "mainMaterials": "Iron ore, coal",
  "processDescription": "Steel manufacturing process"
}
```

### AI Matching
```http
POST /api/match
Content-Type: application/json

{
  "buyer": {
    "id": "buyer123",
    "industry": "Automotive",
    "needs": ["steel", "aluminum"]
  },
  "seller": {
    "id": "seller456",
    "industry": "Metallurgy",
    "products": ["steel", "aluminum"]
  }
}
```

### User Feedback
```http
POST /api/feedback
Content-Type: application/json

{
  "matchId": "match123",
  "userId": "user456",
  "rating": 5,
  "feedback": "Excellent match quality",
  "categories": ["quality", "delivery"]
}
```

### Symbiosis Network Analysis
```http
POST /api/symbiosis-network
Content-Type: application/json

{
  "participants": [
    {
      "id": "company1",
      "industry": "Steel Manufacturing",
      "annual_waste": 1000,
      "carbon_footprint": 500,
      "waste_type": "Slag",
      "location": "Amman"
    }
  ]
}
```

### Explainable AI
```http
POST /api/explain-match
Content-Type: application/json

{
  "buyer": {
    "id": "buyer123",
    "industry": "Automotive"
  },
  "seller": {
    "id": "seller456",
    "industry": "Metallurgy"
  }
}
```

### AI Chat
```http
POST /api/ai-chat
Content-Type: application/json

{
  "message": "I need help with steel procurement",
  "context": {
    "userId": "user123"
  }
}
```

### Real-time Recommendations
```http
POST /api/real-time-recommendations
Content-Type: application/json

{
  "userId": "user123",
  "userProfile": {
    "industry": "Automotive",
    "preferences": ["sustainability", "cost-effectiveness"]
  }
}
```

## 🔒 Security Features

- **Helmet**: Security headers
- **CORS**: Cross-origin resource sharing
- **Rate Limiting**: 100 requests per 15 minutes per IP
- **Input Validation**: Express-validator for all endpoints
- **Error Handling**: Comprehensive error management

## 📊 Monitoring

### Sentry Integration
Configure `SENTRY_DSN` in your environment variables to enable error monitoring and performance tracking.

### Health Check
Monitor server health at `/api/health`

## 🧪 Testing

### Run Tests
```bash
npm test
```

### Test Coverage
```bash
npm test -- --coverage
```

### Test Structure
- **Unit Tests**: Individual function testing
- **Integration Tests**: API endpoint testing
- **Validation Tests**: Input validation testing
- **Error Handling Tests**: Error scenario testing

## 📁 Project Structure

```
backend/
├── app.js                 # Main application file
├── package.json          # Dependencies and scripts
├── jest.config.js        # Jest configuration
├── sentry.config.js      # Sentry error monitoring
├── env.example           # Environment variables template
├── tests/
│   ├── setup.js          # Test setup configuration
│   └── app.test.js       # API endpoint tests
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 5000 |
| `NODE_ENV` | Environment | development |
| `FRONTEND_URL` | Frontend URL for CORS | http://localhost:5173 |
| `SENTRY_DSN` | Sentry DSN for error monitoring | - |
| `RATE_LIMIT_MAX_REQUESTS` | Rate limit requests | 100 |
| `RATE_LIMIT_WINDOW_MS` | Rate limit window | 900000 |

## 🚀 Deployment

### Production Checklist
- [ ] Set `NODE_ENV=production`
- [ ] Configure `SENTRY_DSN`
- [ ] Set up proper CORS origins
- [ ] Configure rate limiting
- [ ] Set up logging
- [ ] Run security audit: `npm audit`

### Docker Deployment
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 5000
CMD ["npm", "start"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Run the test suite
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation
- Review the test cases for usage examples 