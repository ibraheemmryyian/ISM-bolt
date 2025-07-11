#!/usr/bin/env python3
"""
List all available endpoints in the ISM AI backend
"""

import requests
import json

def list_all_endpoints():
    """List all available endpoints"""
    
    print("🚀 ISM AI Backend - All Available Endpoints")
    print("=" * 60)
    
    # Backend URL
    BASE_URL = "http://localhost:5001"
    
    # Test if backend is running
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend is not responding properly")
            return
    except:
        print("❌ Backend is not running on port 5001")
        print("Please start the backend with: start_backend.bat")
        return
    
    print("✅ Backend is running on port 5001")
    print()
    
    # Define all endpoint categories
    endpoints = {
        "🔧 System & Health": [
            ("GET", "/api/health", "Backend health check"),
            ("GET", "/health", "Alternative health check"),
            ("GET", "/api/monitoring/health", "Detailed health status"),
            ("GET", "/api/monitoring/metrics", "System metrics"),
        ],
        
        "🏢 Company Management": [
            ("POST", "/api/companies", "Create new company"),
            ("GET", "/api/companies", "List all companies"),
            ("GET", "/api/companies/:id", "Get company by ID"),
            ("GET", "/api/companies/current", "Get current company"),
        ],
        
        "🤖 AI Portfolio & Listings": [
            ("POST", "/api/ai-portfolio-generation", "Generate AI portfolio"),
            ("POST", "/api/generate-ai-portfolio", "Alternative portfolio generation"),
            ("POST", "/api/ai/portfolio/generate", "Advanced portfolio generation"),
            ("POST", "/api/v1/companies/:id/generate-listings", "Generate listings for company"),
            ("POST", "/api/ai/generate-listings/:companyId", "Generate AI listings"),
            ("POST", "/api/ai/generate-all-listings", "Generate all listings"),
            ("GET", "/api/ai/listings-stats", "Get listings statistics"),
        ],
        
        "🎯 AI Matching & Analysis": [
            ("POST", "/api/ai-matchmaking", "Basic AI matchmaking"),
            ("POST", "/api/ai-match", "AI matching"),
            ("POST", "/api/ai-pipeline", "Full AI pipeline"),
            ("POST", "/api/enhanced-matching", "Enhanced matching"),
            ("POST", "/api/intelligent-matching", "Intelligent matching"),
            ("POST", "/api/ai/revolutionary/match", "Revolutionary AI matching"),
            ("POST", "/api/ai/revolutionary/predict-compatibility", "Predict compatibility"),
            ("POST", "/api/ai/revolutionary/feedback", "Record matching feedback"),
        ],
        
        "🧠 GNN & Graph Analysis": [
            ("POST", "/api/ai/gnn/create-graph", "Create industrial graph"),
            ("POST", "/api/ai/gnn/train-model", "Train GNN model"),
            ("POST", "/api/ai/gnn/infer", "GNN inference"),
            ("GET", "/api/ai/gnn/models", "List GNN models"),
        ],
        
        "📊 Knowledge Graph": [
            ("POST", "/api/ai/knowledge-graph/build", "Build knowledge graph"),
            ("POST", "/api/ai/knowledge-graph/query", "Query knowledge graph"),
        ],
        
        "🔄 Federated Learning": [
            ("POST", "/api/ai/federated/initialize", "Initialize federated learning"),
            ("POST", "/api/ai/federated/train", "Train federated round"),
        ],
        
        "🌐 Multi-hop Symbiosis": [
            ("POST", "/api/ai/multi-hop/analyze", "Analyze multi-hop network"),
            ("POST", "/api/ai/multi-hop/optimize", "Optimize network"),
        ],
        
        "🔬 Advanced AI Integration": [
            ("POST", "/api/ai/integration/comprehensive-analysis", "Comprehensive AI analysis"),
        ],
        
        "📈 Analytics & Insights": [
            ("POST", "/api/analytics", "General analytics"),
            ("POST", "/api/ai-insights", "AI insights"),
            ("GET", "/api/ai-insights", "Get AI insights"),
            ("POST", "/api/comprehensive-match-analysis", "Comprehensive match analysis"),
            ("POST", "/api/refinement-analysis", "Refinement analysis"),
            ("POST", "/api/financial-analysis", "Financial analysis"),
            ("POST", "/api/ai-analysis", "AI analysis"),
            ("POST", "/api/ai-feedback", "AI feedback"),
            ("GET", "/api/ai-feedback-stats", "AI feedback statistics"),
        ],
        
        "🌱 Sustainability & Environment": [
            ("POST", "/api/carbon-calculate", "Calculate carbon footprint"),
            ("POST", "/api/waste-calculate", "Calculate waste impact"),
            ("POST", "/api/sustainability-initiatives", "Sustainability initiatives"),
            ("POST", "/api/environmental-analysis", "Environmental analysis"),
            ("POST", "/api/sustainability/assess", "Assess sustainability"),
        ],
        
        "🚚 Shipping & Logistics": [
            ("POST", "/api/shipping/calculate-rates", "Calculate shipping rates"),
            ("POST", "/api/shipping/create-exchange", "Create shipping exchange"),
            ("GET", "/api/shipping/track/:trackingNumber", "Track shipment"),
            ("POST", "/api/shipping/validate-address", "Validate address"),
            ("GET", "/api/shipping/history/:companyId", "Shipping history"),
            ("PUT", "/api/shipping/exchange/:exchangeId/status", "Update exchange status"),
            ("POST", "/api/shipping/rates", "Get shipping rates"),
            ("POST", "/api/shipping/label", "Generate shipping label"),
            ("POST", "/api/shipping/optimize", "Optimize shipping"),
            ("POST", "/api/freightos/rates", "Freightos rates"),
            ("POST", "/api/freightos/emissions", "Freightos emissions"),
            ("GET", "/api/freightos/network/:region?", "Freightos network"),
        ],
        
        "💳 Payments & Subscriptions": [
            ("POST", "/api/payments/create-order", "Create payment order"),
            ("POST", "/api/payments/capture/:orderId", "Capture payment"),
            ("POST", "/api/payments/create-subscription", "Create subscription"),
            ("POST", "/api/payments/refund", "Process refund"),
            ("GET", "/api/payments/analytics/:companyId", "Payment analytics"),
            ("POST", "/api/payments/webhook", "Payment webhook"),
        ],
        
        "📋 Materials & Portfolio": [
            ("GET", "/api/materials/:materialName/data", "Get material data"),
            ("GET", "/api/materials/:materialId/properties", "Get material properties"),
            ("GET", "/api/materials/:materialId/alternatives", "Get material alternatives"),
            ("POST", "/api/materials/impact/calculate", "Calculate material impact"),
            ("GET", "/api/materials/:materialId/circular-opportunities", "Circular opportunities"),
            ("POST", "/api/materials/matching/scientific", "Scientific material matching"),
            ("GET", "/api/materials/:materialId/supply-chain", "Supply chain analysis"),
            ("POST", "/api/materials/translate-shipping", "Translate shipping materials"),
            ("POST", "/api/materials/analyze", "Analyze materials"),
            ("POST", "/api/portfolio/scientific", "Scientific portfolio"),
            ("POST", "/api/portfolio/create", "Create portfolio"),
            ("GET", "/api/portfolio", "Get portfolio"),
        ],
        
        "🎓 Onboarding & AI Questions": [
            ("POST", "/api/onboarding-flow", "Onboarding flow"),
            ("POST", "/api/ai-onboarding/initial-questions", "Initial AI questions"),
            ("POST", "/api/ai-onboarding/questions", "AI onboarding questions"),
            ("POST", "/api/ai-onboarding/complete", "Complete AI onboarding"),
            ("POST", "/api/ai-onboarding/scientific-complete", "Complete scientific onboarding"),
            ("POST", "/api/adaptive-onboarding/start", "Start adaptive onboarding"),
            ("POST", "/api/adaptive-onboarding/respond", "Respond to adaptive onboarding"),
            ("POST", "/api/adaptive-onboarding/complete", "Complete adaptive onboarding"),
            ("GET", "/api/adaptive-onboarding/status/:session_id", "Get onboarding status"),
            ("GET", "/api/adaptive-onboarding/questions/:session_id", "Get onboarding questions"),
        ],
        
        "🔧 Height Integration": [
            ("POST", "/api/height/create-exchange-tracking", "Create Height exchange tracking"),
            ("POST", "/api/height/create-sustainability-tracking", "Create sustainability tracking"),
            ("GET", "/api/height/project/:projectId", "Get Height project"),
            ("POST", "/api/height/task/:taskId/comment", "Add task comment"),
            ("GET", "/api/height/workspace/members", "Get workspace members"),
        ],
        
        "📊 Real Data & Import": [
            ("POST", "/api/real-data/process-company", "Process real company data"),
            ("POST", "/api/real-data/bulk-import", "Bulk import companies"),
            ("GET", "/api/real-data/import-status/:importId", "Get import status"),
            ("GET", "/api/real-data/high-value-targets", "Get high-value targets"),
            ("GET", "/api/real-data/symbiosis-network", "Get symbiosis network"),
            ("GET", "/api/real-data/market-analysis", "Get market analysis"),
        ],
        
        "💰 Cost & Financial": [
            ("POST", "/api/cost-breakdown", "Cost breakdown"),
            ("POST", "/api/equipment-recommendations", "Equipment recommendations"),
        ],
        
        "🔌 Plugins & Extensions": [
            ("GET", "/api/plugins", "List plugins"),
            ("POST", "/api/plugins", "Install plugin"),
        ],
        
        "💬 Chat & Communication": [
            ("POST", "/api/chat", "Chat interface"),
            ("GET", "/api/v1/companies/:id/notifications/stream", "Notifications stream"),
        ],
        
        "🌐 Network & Symbiosis": [
            ("POST", "/api/symbiosis-network", "Symbiosis network analysis"),
        ],
        
        "🔧 AI Services Management": [
            ("GET", "/api/ai/services/status", "AI services status"),
            ("POST", "/api/ai/services/restart", "Restart AI service"),
        ],
        
        "👨‍💼 Admin & Management": [
            ("GET", "/api/admin/users", "List admin users"),
            ("POST", "/api/admin/upgrade-user", "Upgrade user"),
        ],
    }
    
    # Print all endpoints by category
    total_endpoints = 0
    for category, category_endpoints in endpoints.items():
        print(f"\n{category}")
        print("-" * len(category))
        for method, path, description in category_endpoints:
            print(f"  {method:6} {path:<50} {description}")
            total_endpoints += 1
    
    print(f"\n" + "=" * 60)
    print(f"📊 Total Endpoints: {total_endpoints}")
    print(f"🌐 Base URL: {BASE_URL}")
    print(f"📚 Documentation: PYTHON_AI_ENDPOINTS.md")
    print(f"🧪 Test Script: test_python_ai_endpoints.py")
    print("=" * 60)

if __name__ == "__main__":
    list_all_endpoints() 