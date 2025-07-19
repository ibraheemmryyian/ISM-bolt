@echo off
setlocal EnableDelayedExpansion

REM ========================================
REM Starting SymbioFlows Advanced AI System
REM ========================================

REM Count the number of microservices (count 'start' commands for services only)
set SERVICE_COUNT=0

REM List of microservices (update this block as you add/remove services)
REM Each 'start' command below for a backend service increments SERVICE_COUNT

REM 🚀 Starting Node.js Backend Server...
start "Backend API" cmd /k "npm run dev"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Adaptive AI Onboarding Server on port 5003...
start "Adaptive Onboarding" cmd /k "python adaptive_onboarding_server.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Production AI System...
start "Production AI System" cmd /k "python start_production_ai_system.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting System Health Monitor...
start "System Health Monitor" cmd /k "python system_health_monitor.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Monitoring Dashboard...
start "AI Monitoring Dashboard" cmd /k "python ai_monitoring_dashboard.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Pricing Service...
start "AI Pricing Service" cmd /k "python ai_pricing_service.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Pricing Orchestrator...
start "AI Pricing Orchestrator" cmd /k "python ai_pricing_orchestrator.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Meta-Learning Orchestrator...
start "Meta-Learning Orchestrator" cmd /k "python meta_learning_orchestrator.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Enhanced Materials System...
start "Enhanced Materials System" cmd /k "python start_enhanced_materials_system.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Production AI System Fixed...
start "Production AI System Fixed" cmd /k "python start_production_ai_system_fixed.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting MaterialsBERT Advanced Service...
start "MaterialsBERT Advanced" cmd /k "python materials_bert_service_advanced.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Listings Generator...
start "AI Listings Generator" cmd /k "python ai_listings_generator.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Matchmaking Service...
start "AI Matchmaking Service" cmd /k "python ai_matchmaking_service.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Gateway...
start "AI Gateway" cmd /k "cd ../ai_service_flask && python ai_gateway.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Advanced Analytics Service...
start "Advanced Analytics" cmd /k "cd ../ai_service_flask && python advanced_analytics_service.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Federated Learning Service...
start "Federated Learning" cmd /k "cd ../ai_service_flask && python federated_learning_service.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting GNN Inference Service...
start "GNN Inference" cmd /k "cd ../ai_service_flask && python gnn_inference_service.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Multi-Hop Symbiosis Service...
start "Multi-Hop Symbiosis" cmd /k "cd ../ai_service_flask && python multi_hop_symbiosis_service.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Logistics Cost Service...
start "Logistics Cost Service" cmd /k "cd .. && python logistics_cost_service.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Feedback Orchestrator...
start "AI Feedback Orchestrator" cmd /k "python ai_feedback_orchestrator.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Proactive Opportunity Engine...
start "Proactive Opportunity Engine" cmd /k "python proactive_opportunity_engine.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting GNN Reasoning Engine...
start "GNN Reasoning Engine" cmd /k "python gnn_reasoning_engine.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Dynamic Materials Integration Service...
start "Dynamic Materials Integration" cmd /k "python dynamic_materials_integration_service.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Production Orchestrator...
start "AI Production Orchestrator" cmd /k "python ai_production_orchestrator.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Service Integration...
start "AI Service Integration" cmd /k "python ai_service_integration.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Retraining Pipeline...
start "AI Retraining Pipeline" cmd /k "python ai_retraining_pipeline.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Hyperparameter Optimizer...
start "AI Hyperparameter Optimizer" cmd /k "python ai_hyperparameter_optimizer.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting AI Fusion Layer...
start "AI Fusion Layer" cmd /k "python ai_fusion_layer.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Advanced AI Prompts Service...
start "Advanced AI Prompts Service" cmd /k "python advanced_ai_prompts_service.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Industrial Intelligence Engine...
start "Industrial Intelligence Engine" cmd /k "python industrial_intelligence_engine.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Impact Forecasting...
start "Impact Forecasting" cmd /k "python impact_forecasting.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting DeepSeek R1 Semantic Service...
start "DeepSeek R1 Semantic Service" cmd /k "python deepseek_r1_semantic_service.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Revolutionary AI Matching...
start "Revolutionary AI Matching" cmd /k "python revolutionary_ai_matching.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Multi-Hop Symbiosis Network...
start "Multi-Hop Symbiosis Network" cmd /k "python multi_hop_symbiosis_network.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Workflow Orchestrator...
start "Workflow Orchestrator" cmd /k "python workflow_orchestrator.py"
set /a SERVICE_COUNT+=1

REM 🚀 Starting Complete System Integration...
start "Complete System Integration" cmd /k "python complete_system_integration.py"
set /a SERVICE_COUNT+=1

REM (Frontend and health checks are not counted as microservices)

REM Print the actual number of microservices started
set /a SERVICE_COUNT=!SERVICE_COUNT!
echo.
echo ========================================
echo 🎉 ALL !SERVICE_COUNT! MICROSERVICES STARTED!
echo ========================================
echo.
REM (Optionally, print the list of services here)
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:3000
echo 🧠 AI Onboarding: http://localhost:5003
echo 📊 AI Dashboard: http://localhost:5004
echo 🤖 AI Gateway: http://localhost:5000
echo 💰 AI Pricing: http://localhost:5005
echo 🏭 Production AI: http://localhost:5016
echo 🏥 Health Monitor: http://localhost:5018
echo 📈 Analytics: http://localhost:5001
echo 🔗 Federated Learning: http://localhost:5002
echo 🧠 GNN Inference: http://localhost:5006
echo 🌐 Multi-Hop: http://localhost:5007
echo 📦 Logistics: http://localhost:5008
echo.
echo Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped.
endlocal 