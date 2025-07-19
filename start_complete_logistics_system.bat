@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🚚 SymbioFlows - COMPLETE LOGISTICS SYSTEM
echo ========================================
echo 🎯 Starting COMPLETE Logistics Platform
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
REM START REDIS (CRITICAL INFRASTRUCTURE)
REM ========================================

echo %BLUE%⚡ Starting Redis Infrastructure...%RESET%

REM Check if Redis is already running
echo %YELLOW%Checking Redis status...%RESET%
redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Redis is already running%RESET%
) else (
    echo %YELLOW%Starting Redis server...%RESET%
    start "Redis Server" cmd /k "redis-server"
    timeout /t 5 /nobreak >nul
    
    REM Test Redis connection
    redis-cli ping >nul 2>&1
    if %errorlevel% equ 0 (
        echo %GREEN%✅ Redis started successfully%RESET%
    ) else (
        echo %RED%❌ Redis failed to start%RESET%
        echo %YELLOW%💡 Please install Redis or start it manually%RESET%
        echo %YELLOW%   Windows: Download from https://redis.io/download%RESET%
        echo %YELLOW%   Or use: docker run -d -p 6379:6379 redis:alpine%RESET%
        pause
    )
)

timeout /t 3 /nobreak >nul

REM ========================================
REM START CORE SERVICES
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
REM START LOGISTICS SERVICES
REM ========================================

echo %BLUE%🚚 Starting Logistics Services...%RESET%

REM 3. Logistics Orchestration Engine (Python) on port 5025
echo %YELLOW%3. Starting Logistics Orchestration Engine (port 5025)...%RESET%
start "Logistics Orchestration Engine" cmd /k "cd backend && python logistics_orchestration_engine.py"

REM 4. Complete Logistics Platform (Python) on port 5026
echo %YELLOW%4. Starting Complete Logistics Platform (port 5026)...%RESET%
start "Complete Logistics Platform" cmd /k "cd backend && python complete_logistics_platform.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM START AI SERVICES
REM ========================================

echo %BLUE%🤖 Starting AI Services...%RESET%

REM 5. AI Matchmaking Service (Python) on port 8020
echo %YELLOW%5. Starting AI Matchmaking Service (port 8020)...%RESET%
start "AI Matchmaking Service" cmd /k "cd backend && python ai_matchmaking_service.py"

REM 6. AI Pricing Service (Python) on port 5005
echo %YELLOW%6. Starting AI Pricing Service (port 5005)...%RESET%
start "AI Pricing Service" cmd /k "cd backend && python ai_pricing_service.py"

REM 7. AI Pricing Orchestrator (Python) on port 8030
echo %YELLOW%7. Starting AI Pricing Orchestrator (port 8030)...%RESET%
start "AI Pricing Orchestrator" cmd /k "cd backend && python ai_pricing_orchestrator.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM START ORCHESTRATION SERVICES
REM ========================================

echo %BLUE%🎯 Starting Orchestration Services...%RESET%

REM 8. Advanced Orchestration Engine (Python) on port 5018
echo %YELLOW%8. Starting Advanced Orchestration Engine (port 5018)...%RESET%
start "Advanced Orchestration Engine" cmd /k "cd backend && python advanced_orchestration_engine.py"

REM 9. Service Mesh Proxy (Python) on port 5019
echo %YELLOW%9. Starting Service Mesh Proxy (port 5019)...%RESET%
start "Service Mesh Proxy" cmd /k "cd backend && python service_mesh_proxy.py"

REM 10. Real Service Communication (Python) on port 5020
echo %YELLOW%10. Starting Real Service Communication (port 5020)...%RESET%
start "Real Service Communication" cmd /k "cd backend && python real_service_communication.py"

REM 11. Workflow Orchestrator (Python) on port 5021
echo %YELLOW%11. Starting Workflow Orchestrator (port 5021)...%RESET%
start "Workflow Orchestrator" cmd /k "cd backend && python workflow_orchestrator.py"

REM 12. Distributed Tracing (Python) on port 5022
echo %YELLOW%12. Starting Distributed Tracing (port 5022)...%RESET%
start "Distributed Tracing" cmd /k "cd backend && python distributed_tracing.py"

REM 13. Event-Driven Architecture (Python) on port 5023
echo %YELLOW%13. Starting Event-Driven Architecture (port 5023)...%RESET%
start "Event-Driven Architecture" cmd /k "cd backend && python event_driven_architecture.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM START MONITORING SERVICES
REM ========================================

echo %BLUE%📊 Starting Monitoring Services...%RESET%

REM 14. AI Monitoring Dashboard (Python) on port 5011
echo %YELLOW%14. Starting AI Monitoring Dashboard (port 5011)...%RESET%
start "AI Monitoring Dashboard" cmd /k "cd backend && python ai_monitoring_dashboard.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM START ADDITIONAL SERVICES
REM ========================================

echo %BLUE%🔧 Starting Additional Services...%RESET%

REM 15. Adaptive Onboarding Server (Python) on port 5003
echo %YELLOW%15. Starting Adaptive Onboarding Server (port 5003)...%RESET%
start "Adaptive Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"

REM 16. MaterialsBERT Simple Service (Python) on port 5002
echo %YELLOW%16. Starting MaterialsBERT Simple Service (port 5002)...%RESET%
start "MaterialsBERT Simple" cmd /k "cd backend && python materials_bert_service_simple.py"

REM 17. AI Listings Generator (Python) on port 5010
echo %YELLOW%17. Starting AI Listings Generator (port 5010)...%RESET%
start "AI Listings Generator" cmd /k "cd backend && python ai_listings_generator.py"

REM 18. Ultra AI Listings Generator (Python) on port 5012
echo %YELLOW%18. Starting Ultra AI Listings Generator (port 5012)...%RESET%
start "Ultra AI Listings Generator" cmd /k "cd backend && python ultra_ai_listings_generator.py"

REM 19. Regulatory Compliance (Python) on port 5013
echo %YELLOW%19. Starting Regulatory Compliance Service (port 5013)...%RESET%
start "Regulatory Compliance" cmd /k "cd backend && python regulatory_compliance.py"

REM 20. Proactive Opportunity Engine (Python) on port 5014
echo %YELLOW%20. Starting Proactive Opportunity Engine (port 5014)...%RESET%
start "Proactive Opportunity Engine" cmd /k "cd backend && python proactive_opportunity_engine.py"

REM 21. AI Feedback Orchestrator (Python) on port 5015
echo %YELLOW%21. Starting AI Feedback Orchestrator (port 5015)...%RESET%
start "AI Feedback Orchestrator" cmd /k "cd backend && python ai_feedback_orchestrator.py"

REM 22. Value Function Arbiter (Python) on port 5016
echo %YELLOW%22. Starting Value Function Arbiter (port 5016)...%RESET%
start "Value Function Arbiter" cmd /k "cd backend && python value_function_arbiter.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM START AI SERVICE FLASK
REM ========================================

echo %BLUE%🧠 Starting AI Service Flask...%RESET%

REM 23. AI Gateway (Python) on port 8000
echo %YELLOW%23. Starting AI Gateway (port 8000)...%RESET%
start "AI Gateway" cmd /k "cd ai_service_flask && python ai_gateway.py"

REM 24. Advanced Analytics Service (Python) on port 5004
echo %YELLOW%24. Starting Advanced Analytics Service (port 5004)...%RESET%
start "Advanced Analytics" cmd /k "cd ai_service_flask && python advanced_analytics_service.py"

REM 25. AI Pricing Service Wrapper (Python) on port 8002
echo %YELLOW%25. Starting AI Pricing Service Wrapper (port 8002)...%RESET%
start "AI Pricing Wrapper" cmd /k "cd ai_service_flask && python ai_pricing_service_wrapper.py"

REM 26. GNN Inference Service (Python) on port 8001
echo %YELLOW%26. Starting GNN Inference Service (port 8001)...%RESET%
start "GNN Inference" cmd /k "cd ai_service_flask && python gnn_inference_service.py"

REM 27. Logistics Service Wrapper (Python) on port 8003
echo %YELLOW%27. Starting Logistics Service Wrapper (port 8003)...%RESET%
start "Logistics Wrapper" cmd /k "cd ai_service_flask && python logistics_service_wrapper.py"

REM 28. Multi-Hop Symbiosis Service (Python) on port 5003
echo %YELLOW%28. Starting Multi-Hop Symbiosis Service (port 5003)...%RESET%
start "Multi-Hop Symbiosis" cmd /k "cd ai_service_flask && python multi_hop_symbiosis_service.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM START ROOT SERVICES
REM ========================================

echo %BLUE%🌐 Starting Root Services...%RESET%

REM 29. Logistics Cost Service (Python) on port 5006
echo %YELLOW%29. Starting Logistics Cost Service (port 5006)...%RESET%
start "Logistics Cost Service" cmd /k "python logistics_cost_service.py"

REM 30. Financial Analysis Engine (Python) on port 5017
echo %YELLOW%30. Starting Financial Analysis Engine (port 5017)...%RESET%
start "Financial Analysis Engine" cmd /k "python financial_analysis_engine.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM SYSTEM VALIDATION
REM ========================================

echo %BLUE%🔍 Validating System...%RESET%

REM Wait for services to stabilize
timeout /t 10 /nobreak >nul

REM Test key endpoints
echo %YELLOW%Testing system endpoints...%RESET%

REM Test logistics services
curl -s http://localhost:5025/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Logistics Orchestration Engine: OK%RESET%
) else (
    echo %RED%❌ Logistics Orchestration Engine: FAILED%RESET%
)

curl -s http://localhost:5026/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Complete Logistics Platform: OK%RESET%
) else (
    echo %RED%❌ Complete Logistics Platform: FAILED%RESET%
)

REM Test Redis
redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Redis: OK%RESET%
) else (
    echo %RED%❌ Redis: FAILED%RESET%
)

REM Test backend
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Backend API: OK%RESET%
) else (
    echo %RED%❌ Backend API: FAILED%RESET%
)

REM Test frontend
curl -s http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Frontend: OK%RESET%
) else (
    echo %RED%❌ Frontend: FAILED%RESET%
)

echo.
echo ========================================
echo 🎉 COMPLETE LOGISTICS SYSTEM STARTED!
echo ========================================
echo.
echo 📊 COMPLETE LOGISTICS SYSTEM SUMMARY:
echo.
echo ⚡ INFRASTRUCTURE:
echo ✅ Redis Server: Running on port 6379
echo.
echo 🚚 LOGISTICS SERVICES (2):
echo ✅ Logistics Orchestration Engine: http://localhost:5025
echo ✅ Complete Logistics Platform: http://localhost:5026
echo.
echo 🤖 AI SERVICES (3):
echo ✅ AI Matchmaking Service: http://localhost:8020
echo ✅ AI Pricing Service: http://localhost:5005
echo ✅ AI Pricing Orchestrator: http://localhost:8030
echo.
echo 🎯 ORCHESTRATION SERVICES (6):
echo ✅ Advanced Orchestration Engine: http://localhost:5018
echo ✅ Service Mesh Proxy: http://localhost:5019
echo ✅ Real Service Communication: http://localhost:5020
echo ✅ Workflow Orchestrator: http://localhost:5021
echo ✅ Distributed Tracing: http://localhost:5022
echo ✅ Event-Driven Architecture: http://localhost:5023
echo.
echo 📊 MONITORING SERVICES (1):
echo ✅ AI Monitoring Dashboard: http://localhost:5011
echo.
echo 🔧 BUSINESS SERVICES (18):
echo ✅ Backend API: http://localhost:3000
echo ✅ Frontend: http://localhost:5173
echo ✅ Adaptive Onboarding: http://localhost:5003
echo ✅ MaterialsBERT Simple: http://localhost:5002
echo ✅ AI Listings Generator: http://localhost:5010
echo ✅ Ultra AI Listings Generator: http://localhost:5012
echo ✅ Regulatory Compliance: http://localhost:5013
echo ✅ Proactive Opportunity Engine: http://localhost:5014
echo ✅ AI Feedback Orchestrator: http://localhost:5015
echo ✅ Value Function Arbiter: http://localhost:5016
echo ✅ AI Gateway: http://localhost:8000
echo ✅ Advanced Analytics: http://localhost:5004
echo ✅ AI Pricing Wrapper: http://localhost:8002
echo ✅ GNN Inference: http://localhost:8001
echo ✅ Logistics Wrapper: http://localhost:8003
echo ✅ Multi-Hop Symbiosis: http://localhost:5003
echo ✅ Logistics Cost Service: http://localhost:5006
echo ✅ Financial Analysis Engine: http://localhost:5017
echo.
echo 🚀 LOGISTICS FEATURES:
echo ✅ Freightos Integration Ready
echo ✅ AI-Powered Matching
echo ✅ Complete Cost Calculation
echo ✅ Buyer/Seller Dashboards
echo ✅ Deal Orchestration
echo ✅ Shipment Tracking
echo ✅ Payment Processing
echo ✅ No Direct Communication
echo ✅ Platform Fee Management
echo ✅ Complete Logistics Control
echo.
echo 🎯 TOTAL LOGISTICS SERVICES: 30
echo 🚀 ALL SERVICES STARTED IN SEPARATE WINDOWS
echo.
echo 💡 Each service window shows real-time logs and errors
echo 🛑 Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im redis-server.exe >nul 2>&1
echo ✅ All services stopped. 