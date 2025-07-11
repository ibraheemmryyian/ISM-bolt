# Advanced AI Services for Industrial Symbiosis Marketplace

## Overview

This directory contains a comprehensive suite of advanced AI services designed to power the Industrial Symbiosis Marketplace (ISM) platform. These services provide cutting-edge AI capabilities for customer support, predictive analytics, and circular economy optimization.

## 🚀 Services Architecture

### Core AI Services

1. **AI Gateway** (`ai_gateway.py`) - Port 5000
   - Intelligent request routing and load balancing
   - Service health monitoring and circuit breakers
   - Rate limiting and caching
   - Orchestration of complex AI workflows

2. **GNN Reasoning Engine** (`gnn_reasoning_service.py`) - Port 5001
   - Graph Neural Network inference for industrial networks
   - Multi-hop relationship analysis
   - Real-time pattern recognition
   - PyTorch Geometric integration

3. **Federated Learning Service** (`federated_learning_service.py`) - Port 5002
   - Privacy-preserving distributed learning
   - Multi-party model training
   - Secure aggregation protocols
   - Cross-company knowledge sharing

4. **Multi-Hop Symbiosis Detection** (`multi_hop_symbiosis_service.py`) - Port 5003
   - Advanced circular economy network analysis
   - Multi-hop opportunity detection
   - Pattern recognition and clustering
   - Feasibility assessment

5. **Advanced Analytics Engine** (`advanced_analytics_service.py`) - Port 5004
   - Predictive modeling and forecasting
   - Anomaly detection
   - Trend analysis
   - Impact assessment

## 🛠️ Installation & Setup

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Required Dependencies

```bash
pip install torch torch-geometric
pip install networkx scikit-learn
pip install pandas numpy scipy
pip install aiohttp flask redis
pip install xgboost lightgbm prophet
pip install matplotlib seaborn
```

### Environment Variables

Create a `.env` file:

```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Service Configuration
GATEWAY_PORT=5000
GNN_PORT=5001
FEDERATED_PORT=5002
SYMBIOSIS_PORT=5003
ANALYTICS_PORT=5004

# AI Model Configuration
MODEL_CACHE_DIR=./models
LOG_LEVEL=INFO
```

## 🚀 Quick Start

### 1. Start All Services

```bash
# Terminal 1 - AI Gateway
python ai_gateway.py

# Terminal 2 - GNN Reasoning
python gnn_reasoning_service.py

# Terminal 3 - Federated Learning
python federated_learning_service.py

# Terminal 4 - Multi-Hop Symbiosis
python multi_hop_symbiosis_service.py

# Terminal 5 - Advanced Analytics
python advanced_analytics_service.py
```

### 2. Health Check

```bash
curl http://localhost:5000/health
```

### 3. Service Status

```bash
curl http://localhost:5000/status
```

## 📡 API Reference

### AI Gateway Endpoints

#### Health Check
```http
GET /health
```

#### Service Status
```http
GET /status
```

#### Route to GNN Service
```http
POST /gnn/{endpoint}
Content-Type: application/json
X-Client-ID: your-client-id

{
  "data": {...}
}
```

#### Route to Federated Learning
```http
POST /federated/{endpoint}
Content-Type: application/json
X-Client-ID: your-client-id

{
  "data": {...}
}
```

#### Route to Symbiosis Detection
```http
POST /symbiosis/{endpoint}
Content-Type: application/json
X-Client-ID: your-client-id

{
  "data": {...}
}
```

#### Route to Analytics
```http
POST /analytics/{endpoint}
Content-Type: application/json
X-Client-ID: your-client-id

{
  "data": {...}
}
```

#### Orchestrate Multiple Services
```http
POST /orchestrate
Content-Type: application/json
X-Client-ID: your-client-id

{
  "workflow": [
    {
      "service": "gnn_reasoning",
      "endpoint": "analyze_network",
      "data": {...}
    },
    {
      "service": "analytics",
      "endpoint": "predict_impact",
      "data": {...}
    }
  ]
}
```

### GNN Reasoning Service

#### Health Check
```http
GET /health
```

#### Network Analysis
```http
POST /analyze_network
{
  "nodes": [...],
  "edges": [...],
  "node_features": {...}
}
```

#### Multi-Hop Reasoning
```http
POST /multi_hop_reasoning
{
  "source_node": "company_a",
  "target_node": "company_b",
  "max_hops": 3
}
```

### Federated Learning Service

#### Health Check
```http
GET /health
```

#### Start Training
```http
POST /start_training
{
  "participants": ["company_a", "company_b"],
  "model_config": {...}
}
```

#### Get Training Status
```http
GET /training_status
```

### Multi-Hop Symbiosis Service

#### Health Check
```http
GET /health
```

#### Add Company
```http
POST /add_company
{
  "company_id": "company_a",
  "company_data": {
    "industry": "steel",
    "location": {...},
    "capabilities": [...]
  }
}
```

#### Find Symbiosis
```http
POST /find_symbiosis
{
  "source": "company_a",
  "target": "company_b",
  "max_hops": 3
}
```

### Advanced Analytics Service

#### Health Check
```http
GET /health
```

#### Analyze Data
```http
POST /analyze
{
  "data": [...]
}
```

#### Train Models
```http
POST /train
{
  "features": [...],
  "target": [...]
}
```

#### Make Predictions
```http
POST /predict
{
  "features": [...]
}
```

## 🔧 Configuration

### AI Gateway Configuration

```python
@dataclass
class AIGatewayConfig:
    services: Dict[str, str] = {
        'gnn_reasoning': 'http://localhost:5001',
        'federated_learning': 'http://localhost:5002',
        'multi_hop_symbiosis': 'http://localhost:5003',
        'advanced_analytics': 'http://localhost:5004'
    }
    load_balancing: bool = True
    health_check_interval: int = 30
    request_timeout: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    cache_enabled: bool = True
    rate_limiting: bool = True
    max_requests_per_minute: int = 100
```

### Service-Specific Configuration

Each service has its own configuration class that can be customized:

- `GNNConfig` - GNN model parameters, inference settings
- `FederatedConfig` - Learning rates, aggregation methods
- `MultiHopConfig` - Graph algorithms, pattern recognition
- `AnalyticsConfig` - Model selection, forecasting horizons

## 📊 Monitoring & Metrics

### Health Monitoring

All services provide health endpoints that return:

```json
{
  "status": "healthy",
  "service": "service_name",
  "timestamp": "2024-01-01T00:00:00Z",
  "metrics": {
    "response_time": 0.123,
    "requests_per_second": 10.5,
    "error_rate": 0.01
  }
}
```

### Gateway Status

The gateway provides comprehensive status information:

```json
{
  "services": {
    "gnn_reasoning": {
      "status": "healthy",
      "available": true,
      "metrics": {...}
    }
  },
  "cache_stats": {
    "hits": 1000,
    "misses": 100,
    "hit_rate": 0.91
  },
  "load_balancing": {
    "enabled": true,
    "service_weights": {...}
  }
}
```

## 🔒 Security Features

### Authentication

- API key validation
- Client ID tracking
- Rate limiting per client

### Rate Limiting

- Configurable requests per minute
- Per-client tracking
- Automatic throttling

### Circuit Breakers

- Automatic service isolation
- Configurable failure thresholds
- Automatic recovery

## 🚀 Performance Optimization

### Caching

- Intelligent result caching
- Configurable TTL
- Automatic cache cleanup

### Load Balancing

- Health-based service selection
- Weighted round-robin
- Response time optimization

### Parallel Processing

- Async request handling
- Multi-threaded processing
- Configurable worker pools

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific service tests
python -m pytest tests/test_gnn_service.py
python -m pytest tests/test_analytics_service.py
```

### Integration Tests

```bash
# Test service communication
python -m pytest tests/test_integration.py

# Test gateway orchestration
python -m pytest tests/test_gateway.py
```

### Load Testing

```bash
# Test with multiple concurrent requests
python tests/load_test.py
```

## 📈 Scaling

### Horizontal Scaling

Services can be scaled horizontally by:

1. Running multiple instances of each service
2. Updating the gateway configuration with new service URLs
3. Using a load balancer in front of the gateway

### Vertical Scaling

Increase performance by:

1. Adding more CPU cores
2. Increasing memory allocation
3. Using GPU acceleration for ML models

## 🔧 Troubleshooting

### Common Issues

1. **Service Not Responding**
   - Check if service is running
   - Verify port configuration
   - Check firewall settings

2. **High Response Times**
   - Monitor service health
   - Check cache hit rates
   - Review load balancer configuration

3. **Memory Issues**
   - Monitor memory usage
   - Adjust cache sizes
   - Review model configurations

### Logs

All services log to stdout with configurable levels:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# View logs
tail -f logs/ai_gateway.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information
4. Contact the development team

---

**Built with ❤️ for the Circular Economy Revolution** 