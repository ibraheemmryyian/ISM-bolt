@echo off
echo ========================================
echo Starting SymbioFlows Advanced AI System
echo ========================================
echo 🚀 Starting ALL 18 Microservices...
echo ========================================

REM Suppress warnings globally
set PYTHONWARNINGS=ignore
set PYTHONPATH=%cd%;%PYTHONPATH%

echo.
echo 🚀 Starting Node.js Backend Server...
start "Backend API" cmd /k "npm run dev"

REM Wait for backend to start
echo ⏳ Waiting for backend server to start...
timeout /t 3 /nobreak >nul

echo.
echo 🚀 Starting Adaptive AI Onboarding Server on port 5003...
start "Adaptive Onboarding" cmd /k "python adaptive_onboarding_server.py"

REM Wait a moment for the server to start
echo ⏳ Waiting for adaptive onboarding server to start...
timeout /t 5 /nobreak >nul

REM Test if the adaptive onboarding server is running
echo 🔍 Testing adaptive onboarding server...
python test_adaptive_onboarding.py
if %errorlevel% neq 0 (
    echo ⚠️ Warning: Adaptive onboarding server may not be running properly
    echo Continuing with main system startup...
)

echo.
echo 🚀 Starting Production AI System...
start "Production AI System" cmd /k "python start_production_ai_system.py"

echo.
echo 🚀 Starting System Health Monitor...
start "System Health Monitor" cmd /k "python system_health_monitor.py"

echo.
echo 🚀 Starting AI Monitoring Dashboard...
start "AI Monitoring Dashboard" cmd /k "python ai_monitoring_dashboard.py"

echo.
echo 🚀 Starting AI Pricing Service...
start "AI Pricing Service" cmd /k "python ai_pricing_service.py"

echo.
echo 🚀 Starting AI Pricing Orchestrator...
start "AI Pricing Orchestrator" cmd /k "python ai_pricing_orchestrator.py"

echo.
echo 🚀 Starting Meta-Learning Orchestrator...
start "Meta-Learning Orchestrator" cmd /k "python meta_learning_orchestrator.py"

echo.
echo 🚀 Starting Enhanced Materials System...
start "Enhanced Materials System" cmd /k "python start_enhanced_materials_system.py"

echo.
echo 🚀 Starting Production AI System Fixed...
start "Production AI System Fixed" cmd /k "python start_production_ai_system_fixed.py"

echo.
echo 🚀 Starting MaterialsBERT Advanced Service...
start "MaterialsBERT Advanced" cmd /k "python materials_bert_service_advanced.py"

echo.
echo 🚀 Starting AI Listings Generator...
start "AI Listings Generator" cmd /k "python ai_listings_generator.py"

echo.
echo 🚀 Starting AI Matchmaking Service...
start "AI Matchmaking Service" cmd /k "python ai_matchmaking_service.py"

echo.
echo 🚀 Starting AI Gateway...
start "AI Gateway" cmd /k "cd ../ai_service_flask && python ai_gateway.py"

echo.
echo 🚀 Starting Advanced Analytics Service...
start "Advanced Analytics" cmd /k "cd ../ai_service_flask && python advanced_analytics_service.py"

echo.
echo 🚀 Starting Federated Learning Service...
start "Federated Learning" cmd /k "cd ../ai_service_flask && python federated_learning_service.py"

echo.
echo 🚀 Starting GNN Inference Service...
start "GNN Inference" cmd /k "cd ../ai_service_flask && python gnn_inference_service.py"

echo.
echo 🚀 Starting Multi-Hop Symbiosis Service...
start "Multi-Hop Symbiosis" cmd /k "cd ../ai_service_flask && python multi_hop_symbiosis_service.py"

echo.
echo 🚀 Starting Logistics Cost Service...
start "Logistics Cost Service" cmd /k "cd .. && python logistics_cost_service.py"

echo.
echo 🚀 Starting Frontend Development Server...
start "Frontend" cmd /k "cd ../frontend && npm run dev"

REM Wait for frontend to start
echo ⏳ Waiting for frontend to start...
timeout /t 5 /nobreak >nul

echo.
echo 🔍 Running Final Health Check...
python test_all_services.py

echo.
echo ========================================
echo 🎉 ALL 18 MICROSERVICES STARTED!
echo ========================================
echo.
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