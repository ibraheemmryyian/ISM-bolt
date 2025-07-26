# SymbioFlows API Documentation

## 📚 **API Overview**

The SymbioFlows API provides comprehensive access to industrial symbiosis functionality through RESTful endpoints. The API is designed for high performance, scalability, and ease of integration.

## 🔗 **Base URLs**

- **Development**: `http://localhost:3000/api`
- **Staging**: `https://staging-api.symbioflows.com/api`
- **Production**: `https://api.symbioflows.com/api`

## 🔐 **Authentication**

### **JWT Token Authentication**

All API requests require authentication using JWT tokens.

```http
Authorization: Bearer <your-jwt-token>
```

### **Getting a Token**

```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "your-password"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": "uuid",
      "email": "user@example.com",
      "role": "user",
      "company_id": "uuid"
    }
  }
}
```

## 📋 **API Endpoints**

### **Authentication Endpoints**

#### **Login**
```http
POST /api/auth/login
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "your-password"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "token": "jwt-token",
    "refresh_token": "refresh-token",
    "user": {
      "id": "uuid",
      "email": "user@example.com",
      "role": "user",
      "company_id": "uuid"
    }
  }
}
```

#### **Register**
```http
POST /api/auth/register
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "your-password",
  "company_name": "Example Corp",
  "industry": "manufacturing"
}
```

#### **Logout**
```http
POST /api/auth/logout
Authorization: Bearer <token>
```

#### **Refresh Token**
```http
POST /api/auth/refresh
```

**Request Body:**
```json
{
  "refresh_token": "your-refresh-token"
}
```

#### **Get Profile**
```http
GET /api/auth/profile
Authorization: Bearer <token>
```

### **Companies Endpoints**

#### **Get All Companies**
```http
GET /api/companies
Authorization: Bearer <token>
```

**Query Parameters:**
- `page` (number): Page number for pagination
- `limit` (number): Number of items per page
- `industry` (string): Filter by industry
- `location` (string): Filter by location
- `search` (string): Search in company name

**Response:**
```json
{
  "success": true,
  "data": {
    "companies": [
      {
        "id": "uuid",
        "name": "Example Corp",
        "industry": "manufacturing",
        "location": {
          "city": "New York",
          "country": "USA"
        },
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 100,
      "pages": 10
    }
  }
}
```

#### **Get Company by ID**
```http
GET /api/companies/{id}
Authorization: Bearer <token>
```

#### **Create Company**
```http
POST /api/companies
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "New Company",
  "industry": "manufacturing",
  "location": {
    "city": "New York",
    "country": "USA"
  },
  "description": "Company description"
}
```

#### **Update Company**
```http
PUT /api/companies/{id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Updated Company Name",
  "industry": "technology"
}
```

#### **Delete Company**
```http
DELETE /api/companies/{id}
Authorization: Bearer <token>
```

### **Materials Endpoints**

#### **Get All Materials**
```http
GET /api/materials
Authorization: Bearer <token>
```

**Query Parameters:**
- `page` (number): Page number for pagination
- `limit` (number): Number of items per page
- `company_id` (string): Filter by company
- `type` (string): Filter by material type
- `search` (string): Search in material name

**Response:**
```json
{
  "success": true,
  "data": {
    "materials": [
      {
        "id": "uuid",
        "company_id": "uuid",
        "name": "Steel Scrap",
        "type": "metal",
        "quantity": 1000,
        "unit": "kg",
        "price": 500,
        "description": "High-quality steel scrap",
        "properties": {
          "purity": 95,
          "grade": "A"
        },
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "pagination": {
      "page": 1,
      "limit": 10,
      "total": 50,
      "pages": 5
    }
  }
}
```

#### **Get Material by ID**
```http
GET /api/materials/{id}
Authorization: Bearer <token>
```

#### **Create Material**
```http
POST /api/materials
Authorization: Bearer <token>
Content-Type: application/json

{
  "company_id": "uuid",
  "name": "Plastic Waste",
  "type": "polymer",
  "quantity": 500,
  "unit": "kg",
  "price": 200,
  "description": "Recyclable plastic waste",
  "properties": {
    "type": "PET",
    "color": "clear"
  }
}
```

#### **Update Material**
```http
PUT /api/materials/{id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "quantity": 750,
  "price": 250
}
```

#### **Delete Material**
```http
DELETE /api/materials/{id}
Authorization: Bearer <token>
```

### **AI Matching Endpoints**

#### **Generate Matches**
```http
POST /api/ai/matching
Authorization: Bearer <token>
Content-Type: application/json

{
  "material_id": "uuid",
  "algorithm": "advanced", // "basic", "advanced", "gnn"
  "filters": {
    "max_distance": 100,
    "min_confidence": 0.8,
    "industries": ["manufacturing", "construction"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "matches": [
      {
        "id": "uuid",
        "material_id": "uuid",
        "matched_material_id": "uuid",
        "confidence": 0.95,
        "score": {
          "compatibility": 0.9,
          "economic": 0.85,
          "environmental": 0.95,
          "logistics": 0.8
        },
        "analysis": {
          "compatibility_reason": "Both materials are steel-based",
          "economic_benefit": "Cost savings of $500/ton",
          "environmental_impact": "CO2 reduction of 2.5 tons",
          "logistics_notes": "Same city, low transport cost"
        },
        "created_at": "2024-01-01T00:00:00Z"
      }
    ],
    "processing_time": 2.5,
    "algorithm_used": "advanced"
  }
}
```

#### **Get Match by ID**
```http
GET /api/ai/matches/{id}
Authorization: Bearer <token>
```

#### **Get Match History**
```http
GET /api/ai/matches
Authorization: Bearer <token>
```

**Query Parameters:**
- `material_id` (string): Filter by material
- `date_from` (string): Start date (ISO format)
- `date_to` (string): End date (ISO format)
- `min_confidence` (number): Minimum confidence score

#### **Analyze Material**
```http
POST /api/ai/analyze
Authorization: Bearer <token>
Content-Type: application/json

{
  "material_id": "uuid",
  "analysis_type": "comprehensive" // "basic", "comprehensive", "detailed"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "analysis": {
      "material_properties": {
        "composition": "Steel (95%), Carbon (3%), Other (2%)",
        "grade": "A",
        "purity": 95
      },
      "market_analysis": {
        "current_price": 500,
        "price_trend": "increasing",
        "demand": "high",
        "supply": "moderate"
      },
      "environmental_impact": {
        "co2_emissions": 2.5,
        "energy_consumption": 1500,
        "recyclability": 0.9
      },
      "recommendations": [
        "Consider processing for higher value",
        "Market timing is optimal for sale",
        "Environmental benefits are significant"
      ]
    }
  }
}
```

#### **Get AI Recommendations**
```http
GET /api/ai/recommendations
Authorization: Bearer <token>
```

**Query Parameters:**
- `company_id` (string): Company ID
- `type` (string): Recommendation type
- `limit` (number): Number of recommendations

### **Analytics Endpoints**

#### **Get Impact Analytics**
```http
GET /api/analytics/impact
Authorization: Bearer <token>
```

**Query Parameters:**
- `company_id` (string): Company ID
- `date_from` (string): Start date
- `date_to` (string): End date
- `metric` (string): Metric type

**Response:**
```json
{
  "success": true,
  "data": {
    "environmental_impact": {
      "co2_reduction": 1250.5,
      "waste_diverted": 5000,
      "energy_saved": 7500,
      "water_saved": 2500
    },
    "economic_impact": {
      "cost_savings": 15000,
      "revenue_generated": 25000,
      "roi": 0.67
    },
    "social_impact": {
      "jobs_created": 5,
      "communities_impacted": 3
    }
  }
}
```

#### **Get Performance Analytics**
```http
GET /api/analytics/performance
Authorization: Bearer <token>
```

#### **Get Trend Analytics**
```http
GET /api/analytics/trends
Authorization: Bearer <token>
```

#### **Generate Forecast**
```http
POST /api/analytics/forecast
Authorization: Bearer <token>
Content-Type: application/json

{
  "material_id": "uuid",
  "forecast_period": 30, // days
  "forecast_type": "price" // "price", "demand", "supply"
}
```

### **Logistics Endpoints**

#### **Calculate Shipping Cost**
```http
POST /api/logistics/shipping
Authorization: Bearer <token>
Content-Type: application/json

{
  "origin": {
    "city": "New York",
    "country": "USA"
  },
  "destination": {
    "city": "Los Angeles",
    "country": "USA"
  },
  "material": {
    "weight": 1000,
    "volume": 2.5,
    "type": "metal"
  },
  "transport_mode": "truck" // "truck", "rail", "ship", "air"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "cost_breakdown": {
      "transport": 1500,
      "handling": 200,
      "insurance": 50,
      "total": 1750
    },
    "route": {
      "distance": 2800,
      "duration": "3 days",
      "waypoints": ["NYC", "Chicago", "Denver", "LA"]
    },
    "environmental_impact": {
      "co2_emissions": 2.8,
      "fuel_consumption": 400
    }
  }
}
```

#### **Optimize Route**
```http
POST /api/logistics/optimize
Authorization: Bearer <token>
Content-Type: application/json

{
  "stops": [
    {"city": "New York", "country": "USA"},
    {"city": "Chicago", "country": "USA"},
    {"city": "Los Angeles", "country": "USA"}
  ],
  "constraints": {
    "max_distance": 5000,
    "time_limit": "5 days"
  }
}
```

### **User Management Endpoints**

#### **Get Users**
```http
GET /api/users
Authorization: Bearer <token>
```

#### **Get User by ID**
```http
GET /api/users/{id}
Authorization: Bearer <token>
```

#### **Update User**
```http
PUT /api/users/{id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "John Doe",
  "role": "admin"
}
```

#### **Delete User**
```http
DELETE /api/users/{id}
Authorization: Bearer <token>
```

### **System Endpoints**

#### **Health Check**
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "ai_services": "healthy",
    "cache": "healthy"
  }
}
```

#### **Get System Status**
```http
GET /api/system/status
Authorization: Bearer <token>
```

#### **Get API Documentation**
```http
GET /api/docs
```

## 🔄 **WebSocket Endpoints**

### **Real-time Updates**

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:3000/ws');

ws.onopen = function() {
  console.log('Connected to WebSocket');
  
  // Subscribe to updates
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'matches',
    company_id: 'your-company-id'
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Real-time update:', data);
};
```

### **WebSocket Events**

#### **Match Updates**
```json
{
  "type": "match_update",
  "data": {
    "match_id": "uuid",
    "status": "new",
    "confidence": 0.95,
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

#### **System Alerts**
```json
{
  "type": "system_alert",
  "data": {
    "level": "warning",
    "message": "High system load detected",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

## 📊 **Error Handling**

### **Error Response Format**

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### **Common Error Codes**

- `AUTHENTICATION_ERROR`: Invalid or missing authentication
- `AUTHORIZATION_ERROR`: Insufficient permissions
- `VALIDATION_ERROR`: Invalid input data
- `NOT_FOUND`: Resource not found
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_SERVER_ERROR`: Server error
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

### **HTTP Status Codes**

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## 📈 **Rate Limiting**

API requests are rate-limited to ensure fair usage:

- **Authenticated users**: 1000 requests per hour
- **Guest users**: 100 requests per hour
- **AI endpoints**: 100 requests per hour (higher computational cost)

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## 🔧 **SDK Examples**

### **JavaScript/TypeScript**

```javascript
import { SymbioFlowsAPI } from '@symbioflows/sdk';

const api = new SymbioFlowsAPI({
  baseURL: 'https://api.symbioflows.com',
  token: 'your-jwt-token'
});

// Get companies
const companies = await api.companies.getAll({
  page: 1,
  limit: 10,
  industry: 'manufacturing'
});

// Generate matches
const matches = await api.ai.generateMatches({
  material_id: 'uuid',
  algorithm: 'advanced'
});

// Get analytics
const analytics = await api.analytics.getImpact({
  company_id: 'uuid',
  date_from: '2024-01-01',
  date_to: '2024-01-31'
});
```

### **Python**

```python
from symbioflows import SymbioFlowsAPI

api = SymbioFlowsAPI(
    base_url='https://api.symbioflows.com',
    token='your-jwt-token'
)

# Get companies
companies = api.companies.get_all(
    page=1,
    limit=10,
    industry='manufacturing'
)

# Generate matches
matches = api.ai.generate_matches(
    material_id='uuid',
    algorithm='advanced'
)

# Get analytics
analytics = api.analytics.get_impact(
    company_id='uuid',
    date_from='2024-01-01',
    date_to='2024-01-31'
)
```

## 📚 **Additional Resources**

- [API Changelog](./CHANGELOG.md)
- [SDK Documentation](./SDK.md)
- [Webhook Documentation](./WEBHOOKS.md)
- [Testing Guide](./TESTING.md)

For additional support, contact:
- **Email**: api-support@symbioflows.com
- **Documentation**: https://docs.symbioflows.com/api
- **Status Page**: https://status.symbioflows.com 