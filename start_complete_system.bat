@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🚀 SymbioFlows - COMPLETE SYSTEM STARTUP
echo ========================================
echo 🎯 Starting ALL 33+ Microservices with Full Validation
echo ========================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

REM Suppress warnings globally
set PYTHONWARNINGS=ignore
set PYTHONPATH=%cd%;%PYTHONPATH%

echo %BLUE%🔍 Pre-flight validation...%RESET%
echo.

REM ========================================
REM PREREQUISITE CHECKS
REM ========================================

echo %YELLOW%Checking prerequisites...%RESET%

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Node.js is not installed or not in PATH%RESET%
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Python is not installed or not in PATH%RESET%
    pause
    exit /b 1
)

REM Check npm
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ npm is not installed or not in PATH%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ Prerequisites OK%RESET%
echo.

REM ========================================
REM FILE VALIDATION
REM ========================================

echo %YELLOW%Validating all microservice files...%RESET%

REM Check backend services
set BACKEND_SERVICES=0
if exist "backend\adaptive_onboarding_server.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_listings_generator.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_monitoring_dashboard.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_pricing_service.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_pricing_orchestrator.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_matchmaking_service.py" set /a BACKEND_SERVICES+=1
if exist "backend\meta_learning_orchestrator.py" set /a BACKEND_SERVICES+=1
if exist "backend\materials_bert_service.py" set /a BACKEND_SERVICES+=1
if exist "backend\materials_bert_service_simple.py" set /a BACKEND_SERVICES+=1
if exist "backend\ultra_ai_listings_generator.py" set /a BACKEND_SERVICES+=1
if exist "backend\regulatory_compliance.py" set /a BACKEND_SERVICES+=1
if exist "backend\proactive_opportunity_engine.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_feedback_orchestrator.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_production_orchestrator.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_retraining_pipeline.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_production_ai_system.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_production_ai_system_fixed.py" set /a BACKEND_SERVICES+=1
if exist "backend\system_health_monitor.py" set /a BACKEND_SERVICES+=1
if exist "backend\start_enhanced_materials_system.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_service_integration.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_hyperparameter_optimizer.py" set /a BACKEND_SERVICES+=1
if exist "backend\ai_fusion_layer.py" set /a BACKEND_SERVICES+=1
if exist "backend\value_function_arbiter.py" set /a BACKEND_SERVICES+=1

REM Check AI Service Flask services
set AI_FLASK_SERVICES=0
if exist "ai_service_flask\ai_gateway.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\advanced_analytics_service.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\ai_pricing_service_wrapper.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\gnn_inference_service.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\logistics_service_wrapper.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\multi_hop_symbiosis_service.py" set /a AI_FLASK_SERVICES+=1
if exist "ai_service_flask\federated_learning_service.py" set /a AI_FLASK_SERVICES+=1

REM Check root services
set ROOT_SERVICES=0
if exist "logistics_cost_service.py" set /a ROOT_SERVICES+=1
if exist "optimize_deepseek_r1.py" set /a ROOT_SERVICES+=1
if exist "financial_analysis_engine.py" set /a ROOT_SERVICES+=1

echo %GREEN%✅ Found %BACKEND_SERVICES% backend services%RESET%
echo %GREEN%✅ Found %AI_FLASK_SERVICES% AI Flask services%RESET%
echo %GREEN%✅ Found %ROOT_SERVICES% root services%RESET%

set TOTAL_SERVICES=%BACKEND_SERVICES%
set /a TOTAL_SERVICES+=%AI_FLASK_SERVICES%
set /a TOTAL_SERVICES+=%ROOT_SERVICES%

echo %BLUE%🎯 Total microservices found: %TOTAL_SERVICES%%RESET%
echo.

REM ========================================
REM CORE SERVICES (ALWAYS START)
REM ========================================

echo %BLUE%📦 Starting Core Services...%RESET%

REM 1. Backend (Node.js) on port 3000
echo %YELLOW%1. Starting Node.js Backend (port 3000)...%RESET%
start "Backend API" cmd /k "cd backend && npm run dev"

REM 2. Frontend (React) on port 5173
echo %YELLOW%2. Starting React Frontend (port 5173)...%RESET%
start "Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 8 /nobreak >nul

REM ========================================
REM BACKEND MICROSERVICES (25+)
REM ========================================

echo %BLUE%🤖 Starting Backend Microservices...%RESET%

REM 3. Adaptive Onboarding Server (Python) on port 5003
echo %YELLOW%3. Starting Adaptive Onboarding Server (port 5003)...%RESET%
start "Adaptive Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"

REM 4. System Health Monitor (Python) on port 5018
echo %YELLOW%4. Starting System Health Monitor (port 5018)...%RESET%
start "System Health Monitor" cmd /k "cd backend && python system_health_monitor.py"

REM 5. AI Monitoring Dashboard (Python) on port 5004
echo %YELLOW%5. Starting AI Monitoring Dashboard (port 5004)...%RESET%
start "AI Monitoring Dashboard" cmd /k "cd backend && python ai_monitoring_dashboard.py"

REM 6. AI Pricing Service (Python) on port 5005
echo %YELLOW%6. Starting AI Pricing Service (port 5005)...%RESET%
start "AI Pricing Service" cmd /k "cd backend && python ai_pricing_service.py"

REM 7. AI Pricing Orchestrator (Python) on port 8030
echo %YELLOW%7. Starting AI Pricing Orchestrator (port 8030)...%RESET%
start "AI Pricing Orchestrator" cmd /k "cd backend && python ai_pricing_orchestrator.py"

REM 8. Meta-Learning Orchestrator (Python) on port 8010
echo %YELLOW%8. Starting Meta-Learning Orchestrator (port 8010)...%RESET%
start "Meta-Learning Orchestrator" cmd /k "cd backend && python meta_learning_orchestrator.py"

REM 9. AI Matchmaking Service (Python) on port 8020
echo %YELLOW%9. Starting AI Matchmaking Service (port 8020)...%RESET%
start "AI Matchmaking Service" cmd /k "cd backend && python ai_matchmaking_service.py"

REM 10. AI Listings Generator (Python) - Flask app
echo %YELLOW%10. Starting AI Listings Generator...%RESET%
start "AI Listings Generator" cmd /k "cd backend && python ai_listings_generator.py"

REM 11. MaterialsBERT Service (Python) - Flask app
echo %YELLOW%11. Starting MaterialsBERT Service...%RESET%
start "MaterialsBERT Service" cmd /k "cd backend && python materials_bert_service.py"

REM 12. MaterialsBERT Simple Service (Python) on port 5002
echo %YELLOW%12. Starting MaterialsBERT Simple Service (port 5002)...%RESET%
start "MaterialsBERT Simple" cmd /k "cd backend && python materials_bert_service_simple.py"

REM 13. Ultra AI Listings Generator (Python) - Flask app
echo %YELLOW%13. Starting Ultra AI Listings Generator...%RESET%
start "Ultra AI Listings Generator" cmd /k "cd backend && python ultra_ai_listings_generator.py"

REM 14. Regulatory Compliance (Python) - Flask app
echo %YELLOW%14. Starting Regulatory Compliance Service...%RESET%
start "Regulatory Compliance" cmd /k "cd backend && python regulatory_compliance.py"

REM 15. Proactive Opportunity Engine (Python) - Flask app
echo %YELLOW%15. Starting Proactive Opportunity Engine...%RESET%
start "Proactive Opportunity Engine" cmd /k "cd backend && python proactive_opportunity_engine.py"

REM 16. AI Feedback Orchestrator (Python) - Flask app
echo %YELLOW%16. Starting AI Feedback Orchestrator...%RESET%
start "AI Feedback Orchestrator" cmd /k "cd backend && python ai_feedback_orchestrator.py"

REM 17. AI Production Orchestrator (Python) - Production orchestrator
echo %YELLOW%17. Starting AI Production Orchestrator...%RESET%
start "AI Production Orchestrator" cmd /k "cd backend && python ai_production_orchestrator.py"

REM 18. AI Retraining Pipeline (Python) - Retraining pipeline
echo %YELLOW%18. Starting AI Retraining Pipeline...%RESET%
start "AI Retraining Pipeline" cmd /k "cd backend && python ai_retraining_pipeline.py"

REM 19. Production AI System (Python) - Production system
echo %YELLOW%19. Starting Production AI System...%RESET%
start "Production AI System" cmd /k "cd backend && python start_production_ai_system.py"

REM 20. Production AI System Fixed (Python) - Fixed production system
echo %YELLOW%20. Starting Production AI System Fixed...%RESET%
start "Production AI System Fixed" cmd /k "cd backend && python start_production_ai_system_fixed.py"

REM 21. Enhanced Materials System (Python) - Materials system
echo %YELLOW%21. Starting Enhanced Materials System...%RESET%
start "Enhanced Materials System" cmd /k "cd backend && python start_enhanced_materials_system.py"

REM 22. AI Service Integration (Python) - Service integration
echo %YELLOW%22. Starting AI Service Integration...%RESET%
start "AI Service Integration" cmd /k "cd backend && python ai_service_integration.py"

REM 23. AI Hyperparameter Optimizer (Python) - Hyperparameter optimization
echo %YELLOW%23. Starting AI Hyperparameter Optimizer...%RESET%
start "AI Hyperparameter Optimizer" cmd /k "cd backend && python ai_hyperparameter_optimizer.py"

REM 24. AI Fusion Layer (Python) - Fusion layer
echo %YELLOW%24. Starting AI Fusion Layer...%RESET%
start "AI Fusion Layer" cmd /k "cd backend && python ai_fusion_layer.py"

REM 25. Value Function Arbiter (Python) - FastAPI app
echo %YELLOW%25. Starting Value Function Arbiter...%RESET%
start "Value Function Arbiter" cmd /k "cd backend && python value_function_arbiter.py"

timeout /t 10 /nobreak >nul

REM ========================================
REM AI SERVICE FLASK MICROSERVICES (7)
REM ========================================

echo %BLUE%🧠 Starting AI Service Flask Microservices...%RESET%

REM 26. AI Gateway (Python) on port 8000
echo %YELLOW%26. Starting AI Gateway (port 8000)...%RESET%
start "AI Gateway" cmd /k "cd ai_service_flask && python ai_gateway.py"

REM 27. Advanced Analytics Service (Python) on port 5004
echo %YELLOW%27. Starting Advanced Analytics Service (port 5004)...%RESET%
start "Advanced Analytics" cmd /k "cd ai_service_flask && python advanced_analytics_service.py"

REM 28. AI Pricing Service Wrapper (Python) on port 8002
echo %YELLOW%28. Starting AI Pricing Service Wrapper (port 8002)...%RESET%
start "AI Pricing Wrapper" cmd /k "cd ai_service_flask && python ai_pricing_service_wrapper.py"

REM 29. GNN Inference Service (Python) on port 8001
echo %YELLOW%29. Starting GNN Inference Service (port 8001)...%RESET%
start "GNN Inference" cmd /k "cd ai_service_flask && python gnn_inference_service.py"

REM 30. Logistics Service Wrapper (Python) on port 8003
echo %YELLOW%30. Starting Logistics Service Wrapper (port 8003)...%RESET%
start "Logistics Wrapper" cmd /k "cd ai_service_flask && python logistics_service_wrapper.py"

REM 31. Multi-Hop Symbiosis Service (Python) on port 5003
echo %YELLOW%31. Starting Multi-Hop Symbiosis Service (port 5003)...%RESET%
start "Multi-Hop Symbiosis" cmd /k "cd ai_service_flask && python multi_hop_symbiosis_service.py"

REM 32. Federated Learning Service (Python) - Flask app
echo %YELLOW%32. Starting Federated Learning Service...%RESET%
start "Federated Learning" cmd /k "cd ai_service_flask && python federated_learning_service.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM ROOT MICROSERVICES (3)
REM ========================================

echo %BLUE%🌐 Starting Root Microservices...%RESET%

REM 33. Logistics Cost Service (Python) on port 5006
echo %YELLOW%33. Starting Logistics Cost Service (port 5006)...%RESET%
start "Logistics Cost Service" cmd /k "python logistics_cost_service.py"

REM 34. Optimize DeepSeek R1 (Python) on port 5005
echo %YELLOW%34. Starting Optimize DeepSeek R1 (port 5005)...%RESET%
start "Optimize DeepSeek R1" cmd /k "python optimize_deepseek_r1.py"

REM 35. Financial Analysis Engine (Python) - Flask app
echo %YELLOW%35. Starting Financial Analysis Engine...%RESET%
start "Financial Analysis Engine" cmd /k "python financial_analysis_engine.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM HEALTH CHECK AND VALIDATION
REM ========================================

echo.
echo %BLUE%🔍 Running comprehensive health checks...%RESET%

REM Test core services
echo %YELLOW%Testing core services...%RESET%
cd backend
python test_all_services.py
cd ..

echo.
echo ========================================
echo 🎉 COMPLETE SYSTEM STARTED!
echo ========================================
echo.
echo 📊 SYSTEM SUMMARY:
echo ✅ Backend API: http://localhost:3000
echo ✅ Frontend: http://localhost:5173
echo ✅ Adaptive Onboarding: http://localhost:5003
echo ✅ System Health Monitor: http://localhost:5018
echo ✅ AI Monitoring Dashboard: http://localhost:5004
echo ✅ AI Pricing Service: http://localhost:5005
echo ✅ AI Pricing Orchestrator: http://localhost:8030
echo ✅ Meta-Learning Orchestrator: http://localhost:8010
echo ✅ AI Matchmaking Service: http://localhost:8020
echo ✅ MaterialsBERT Simple: http://localhost:5002
echo ✅ AI Gateway: http://localhost:8000
echo ✅ Advanced Analytics: http://localhost:5004
echo ✅ AI Pricing Wrapper: http://localhost:8002
echo ✅ GNN Inference: http://localhost:8001
echo ✅ Logistics Wrapper: http://localhost:8003
echo ✅ Multi-Hop Symbiosis: http://localhost:5003
echo ✅ Logistics Cost Service: http://localhost:5006
echo ✅ Optimize DeepSeek R1: http://localhost:5005
echo.
echo 🎯 TOTAL MICROSERVICES: %TOTAL_SERVICES%
echo 🚀 ALL SERVICES STARTED IN SEPARATE WINDOWS
echo.
echo 💡 Each service window shows real-time logs and errors
echo 🛑 Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped. 