@echo off
REM Start all major backend AI services and frontend in separate terminals

REM Start Node.js backend
start cmd /k "cd backend && npm run dev"

REM Start MaterialsBERT Advanced Service
start cmd /k "python backend/materials_bert_service_advanced.py"

REM Start Adaptive AI Onboarding Service
start cmd /k "python backend/adaptive_onboarding_server.py"

REM Start Advanced Analytics Service
start cmd /k "python ai_service_flask/advanced_analytics_service.py"

REM Start Federated Learning Service
start cmd /k "python ai_service_flask/federated_learning_service.py"

REM Start GNN Inference Service
start cmd /k "python ai_service_flask/gnn_inference_service.py"

REM Start Multi-Hop Symbiosis Service
start cmd /k "python ai_service_flask/multi_hop_symbiosis_service.py"

REM Start AI Pricing Service
start cmd /k "python backend/ai_pricing_service.py"

REM Start Logistics Cost Service
start cmd /k "python logistics_cost_service.py"

REM Start Frontend React Dev Server
start cmd /k "cd frontend && npm run dev"

@echo All services started in separate windows. 