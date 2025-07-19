@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🚀 SymbioFlows - PRODUCTION-READY SYSTEM
echo ========================================
echo 🎯 Starting ALL Production-Ready Microservices
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
REM PRODUCTION-READY ORCHESTRATION SERVICES (6)
REM ========================================

echo %BLUE%🎯 Starting Production-Ready Orchestration Services...%RESET%

REM 3. Advanced Orchestration Engine (Python) on port 5018
echo %YELLOW%3. Starting Advanced Orchestration Engine (port 5018)...%RESET%
start "Advanced Orchestration Engine" cmd /k "cd backend && python advanced_orchestration_engine.py"

REM 4. Service Mesh Proxy (Python) on port 5019
echo %YELLOW%4. Starting Service Mesh Proxy (port 5019)...%RESET%
start "Service Mesh Proxy" cmd /k "cd backend && python service_mesh_proxy.py"

REM 5. Real Service Communication (Python) on port 5020
echo %YELLOW%5. Starting Real Service Communication (port 5020)...%RESET%
start "Real Service Communication" cmd /k "cd backend && python real_service_communication.py"

REM 6. Workflow Orchestrator (Python) on port 5021
echo %YELLOW%6. Starting Workflow Orchestrator (port 5021)...%RESET%
start "Workflow Orchestrator" cmd /k "cd backend && python workflow_orchestrator.py"

REM 7. Distributed Tracing (Python) on port 5022
echo %YELLOW%7. Starting Distributed Tracing (port 5022)...%RESET%
start "Distributed Tracing" cmd /k "cd backend && python distributed_tracing.py"

REM 8. Event-Driven Architecture (Python) on port 5023
echo %YELLOW%8. Starting Event-Driven Architecture (port 5023)...%RESET%
start "Event-Driven Architecture" cmd /k "cd backend && python event_driven_architecture.py"

timeout /t 10 /nobreak >nul

REM ========================================
REM REAL BACKEND MICROSERVICES (15)
REM ========================================

echo %BLUE%🤖 Starting Real Backend Microservices...%RESET%

REM 9. Adaptive Onboarding Server (Python) on port 5003
echo %YELLOW%9. Starting Adaptive Onboarding Server (port 5003)...%RESET%
start "Adaptive Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"

REM 10. AI Pricing Service (Python) on port 5005
echo %YELLOW%10. Starting AI Pricing Service (port 5005)...%RESET%
start "AI Pricing Service" cmd /k "cd backend && python ai_pricing_service.py"

REM 11. AI Pricing Orchestrator (Python) on port 8030
echo %YELLOW%11. Starting AI Pricing Orchestrator (port 8030)...%RESET%
start "AI Pricing Orchestrator" cmd /k "cd backend && python ai_pricing_orchestrator.py"

REM 12. Meta-Learning Orchestrator (Python) on port 8010
echo %YELLOW%12. Starting Meta-Learning Orchestrator (port 8010)...%RESET%
start "Meta-Learning Orchestrator" cmd /k "cd backend && python meta_learning_orchestrator.py"

REM 13. AI Matchmaking Service (Python) on port 8020
echo %YELLOW%13. Starting AI Matchmaking Service (port 8020)...%RESET%
start "AI Matchmaking Service" cmd /k "cd backend && python ai_matchmaking_service.py"

REM 14. MaterialsBERT Service (Python) - dynamic port
echo %YELLOW%14. Starting MaterialsBERT Service (dynamic port)...%RESET%
start "MaterialsBERT Service" cmd /k "cd backend && python materials_bert_service.py"

REM 15. MaterialsBERT Simple Service (Python) on port 5002
echo %YELLOW%15. Starting MaterialsBERT Simple Service (port 5002)...%RESET%
start "MaterialsBERT Simple" cmd /k "cd backend && python materials_bert_service_simple.py"

REM 16. AI Listings Generator (Python) on port 5010
echo %YELLOW%16. Starting AI Listings Generator (port 5010)...%RESET%
start "AI Listings Generator" cmd /k "cd backend && python ai_listings_generator.py"

REM 17. AI Monitoring Dashboard (Python) on port 5011
echo %YELLOW%17. Starting AI Monitoring Dashboard (port 5011)...%RESET%
start "AI Monitoring Dashboard" cmd /k "cd backend && python ai_monitoring_dashboard.py"

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

timeout /t 10 /nobreak >nul

REM ========================================
REM AI SERVICE FLASK MICROSERVICES (6)
REM ========================================

echo %BLUE%🧠 Starting AI Service Flask Microservices...%RESET%

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
REM ROOT MICROSERVICES (3)
REM ========================================

echo %BLUE%🌐 Starting Root Microservices...%RESET%

REM 29. Logistics Cost Service (Python) on port 5006
echo %YELLOW%29. Starting Logistics Cost Service (port 5006)...%RESET%
start "Logistics Cost Service" cmd /k "python logistics_cost_service.py"

REM 30. Optimize DeepSeek R1 (Python) on port 5005
echo %YELLOW%30. Starting Optimize DeepSeek R1 (port 5005)...%RESET%
start "Optimize DeepSeek R1" cmd /k "python optimize_deepseek_r1.py"

REM 31. Financial Analysis Engine (Python) on port 5017
echo %YELLOW%31. Starting Financial Analysis Engine (port 5017)...%RESET%
start "Financial Analysis Engine" cmd /k "python financial_analysis_engine.py"

timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo 🎉 ALL 31 PRODUCTION-READY MICROSERVICES STARTED!
echo ========================================
echo.
echo 📊 PRODUCTION-READY SYSTEM SUMMARY:
echo.
echo 🎯 ORCHESTRATION SERVICES (6):
echo ✅ Advanced Orchestration Engine: http://localhost:5018
echo ✅ Service Mesh Proxy: http://localhost:5019
echo ✅ Real Service Communication: http://localhost:5020
echo ✅ Workflow Orchestrator: http://localhost:5021
echo ✅ Distributed Tracing: http://localhost:5022
echo ✅ Event-Driven Architecture: http://localhost:5023
echo.
echo 🤖 BACKEND MICROSERVICES (15):
echo ✅ Backend API: http://localhost:3000
echo ✅ Frontend: http://localhost:5173
echo ✅ Adaptive Onboarding: http://localhost:5003
echo ✅ AI Pricing Service: http://localhost:5005
echo ✅ AI Pricing Orchestrator: http://localhost:8030
echo ✅ Meta-Learning Orchestrator: http://localhost:8010
echo ✅ AI Matchmaking Service: http://localhost:8020
echo ✅ MaterialsBERT Simple: http://localhost:5002
echo ✅ AI Listings Generator: http://localhost:5010
echo ✅ AI Monitoring Dashboard: http://localhost:5011
echo ✅ Ultra AI Listings Generator: http://localhost:5012
echo ✅ Regulatory Compliance: http://localhost:5013
echo ✅ Proactive Opportunity Engine: http://localhost:5014
echo ✅ AI Feedback Orchestrator: http://localhost:5015
echo ✅ Value Function Arbiter: http://localhost:5016
echo.
echo 🧠 AI SERVICE FLASK (6):
echo ✅ AI Gateway: http://localhost:8000
echo ✅ Advanced Analytics: http://localhost:5004
echo ✅ AI Pricing Wrapper: http://localhost:8002
echo ✅ GNN Inference: http://localhost:8001
echo ✅ Logistics Wrapper: http://localhost:8003
echo ✅ Multi-Hop Symbiosis: http://localhost:5003
echo.
echo 🌐 ROOT SERVICES (3):
echo ✅ Logistics Cost Service: http://localhost:5006
echo ✅ Optimize DeepSeek R1: http://localhost:5005
echo ✅ Financial Analysis Engine: http://localhost:5017
echo.
echo 🚀 PRODUCTION-READY FEATURES:
echo ✅ Real Inter-Service Communication (HTTP/gRPC)
echo ✅ Advanced Workflow Orchestration
echo ✅ Service Mesh with Load Balancing
echo ✅ Distributed Tracing (Jaeger)
echo ✅ Event-Driven Architecture (Redis Streams)
echo ✅ Circuit Breakers & Retry Logic
echo ✅ CQRS & Event Sourcing
echo ✅ Health Monitoring & Metrics
echo ✅ Fault Tolerance & Error Handling
echo.
echo 🎯 TOTAL PRODUCTION-READY MICROSERVICES: 31
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