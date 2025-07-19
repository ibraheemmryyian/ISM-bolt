@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🚀 SymbioFlows - REAL MICROSERVICES STARTUP
echo ========================================
echo 🎯 Starting ALL REAL Microservices (with app.run/uvicorn.run)
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
REM REAL BACKEND MICROSERVICES (15)
REM ========================================

echo %BLUE%🤖 Starting Real Backend Microservices...%RESET%

REM 3. Adaptive Onboarding Server (Python) on port 5003
echo %YELLOW%3. Starting Adaptive Onboarding Server (port 5003)...%RESET%
start "Adaptive Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"

REM 4. AI Pricing Service (Python) on port 5005
echo %YELLOW%4. Starting AI Pricing Service (port 5005)...%RESET%
start "AI Pricing Service" cmd /k "cd backend && python ai_pricing_service.py"

REM 5. AI Pricing Orchestrator (Python) on port 8030
echo %YELLOW%5. Starting AI Pricing Orchestrator (port 8030)...%RESET%
start "AI Pricing Orchestrator" cmd /k "cd backend && python ai_pricing_orchestrator.py"

REM 6. Meta-Learning Orchestrator (Python) on port 8010
echo %YELLOW%6. Starting Meta-Learning Orchestrator (port 8010)...%RESET%
start "Meta-Learning Orchestrator" cmd /k "cd backend && python meta_learning_orchestrator.py"

REM 7. AI Matchmaking Service (Python) on port 8020
echo %YELLOW%7. Starting AI Matchmaking Service (port 8020)...%RESET%
start "AI Matchmaking Service" cmd /k "cd backend && python ai_matchmaking_service.py"

REM 8. MaterialsBERT Service (Python) - dynamic port
echo %YELLOW%8. Starting MaterialsBERT Service (dynamic port)...%RESET%
start "MaterialsBERT Service" cmd /k "cd backend && python materials_bert_service.py"

REM 9. MaterialsBERT Simple Service (Python) on port 5002
echo %YELLOW%9. Starting MaterialsBERT Simple Service (port 5002)...%RESET%
start "MaterialsBERT Simple" cmd /k "cd backend && python materials_bert_service_simple.py"

REM 10. AI Listings Generator (Python) on port 5010
echo %YELLOW%10. Starting AI Listings Generator (port 5010)...%RESET%
start "AI Listings Generator" cmd /k "cd backend && python ai_listings_generator.py"

REM 11. AI Monitoring Dashboard (Python) on port 5011
echo %YELLOW%11. Starting AI Monitoring Dashboard (port 5011)...%RESET%
start "AI Monitoring Dashboard" cmd /k "cd backend && python ai_monitoring_dashboard.py"

REM 12. Ultra AI Listings Generator (Python) on port 5012
echo %YELLOW%12. Starting Ultra AI Listings Generator (port 5012)...%RESET%
start "Ultra AI Listings Generator" cmd /k "cd backend && python ultra_ai_listings_generator.py"

REM 13. Regulatory Compliance (Python) on port 5013
echo %YELLOW%13. Starting Regulatory Compliance Service (port 5013)...%RESET%
start "Regulatory Compliance" cmd /k "cd backend && python regulatory_compliance.py"

REM 14. Proactive Opportunity Engine (Python) on port 5014
echo %YELLOW%14. Starting Proactive Opportunity Engine (port 5014)...%RESET%
start "Proactive Opportunity Engine" cmd /k "cd backend && python proactive_opportunity_engine.py"

REM 15. AI Feedback Orchestrator (Python) on port 5015
echo %YELLOW%15. Starting AI Feedback Orchestrator (port 5015)...%RESET%
start "AI Feedback Orchestrator" cmd /k "cd backend && python ai_feedback_orchestrator.py"

REM 16. Value Function Arbiter (Python) on port 5016
echo %YELLOW%16. Starting Value Function Arbiter (port 5016)...%RESET%
start "Value Function Arbiter" cmd /k "cd backend && python value_function_arbiter.py"

timeout /t 10 /nobreak >nul

REM ========================================
REM AI SERVICE FLASK MICROSERVICES (5)
REM ========================================

echo %BLUE%🧠 Starting AI Service Flask Microservices...%RESET%

REM 17. AI Gateway (Python) on port 8000
echo %YELLOW%17. Starting AI Gateway (port 8000)...%RESET%
start "AI Gateway" cmd /k "cd ai_service_flask && python ai_gateway.py"

REM 18. Advanced Analytics Service (Python) on port 5004
echo %YELLOW%18. Starting Advanced Analytics Service (port 5004)...%RESET%
start "Advanced Analytics" cmd /k "cd ai_service_flask && python advanced_analytics_service.py"

REM 19. AI Pricing Service Wrapper (Python) on port 8002
echo %YELLOW%19. Starting AI Pricing Service Wrapper (port 8002)...%RESET%
start "AI Pricing Wrapper" cmd /k "cd ai_service_flask && python ai_pricing_service_wrapper.py"

REM 20. GNN Inference Service (Python) on port 8001
echo %YELLOW%20. Starting GNN Inference Service (port 8001)...%RESET%
start "GNN Inference" cmd /k "cd ai_service_flask && python gnn_inference_service.py"

REM 21. Logistics Service Wrapper (Python) on port 8003
echo %YELLOW%21. Starting Logistics Service Wrapper (port 8003)...%RESET%
start "Logistics Wrapper" cmd /k "cd ai_service_flask && python logistics_service_wrapper.py"

REM 22. Multi-Hop Symbiosis Service (Python) on port 5003
echo %YELLOW%22. Starting Multi-Hop Symbiosis Service (port 5003)...%RESET%
start "Multi-Hop Symbiosis" cmd /k "cd ai_service_flask && python multi_hop_symbiosis_service.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM ROOT MICROSERVICES (3)
REM ========================================

echo %BLUE%🌐 Starting Root Microservices...%RESET%

REM 23. Logistics Cost Service (Python) on port 5006
echo %YELLOW%23. Starting Logistics Cost Service (port 5006)...%RESET%
start "Logistics Cost Service" cmd /k "python logistics_cost_service.py"

REM 24. Optimize DeepSeek R1 (Python) on port 5005
echo %YELLOW%24. Starting Optimize DeepSeek R1 (port 5005)...%RESET%
start "Optimize DeepSeek R1" cmd /k "python optimize_deepseek_r1.py"

REM 25. Financial Analysis Engine (Python) on port 5017
echo %YELLOW%25. Starting Financial Analysis Engine (port 5017)...%RESET%
start "Financial Analysis Engine" cmd /k "python financial_analysis_engine.py"

timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo 🎉 ALL 25 REAL MICROSERVICES STARTED!
echo ========================================
echo.
echo 📊 SYSTEM SUMMARY:
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
echo ✅ AI Gateway: http://localhost:8000
echo ✅ Advanced Analytics: http://localhost:5004
echo ✅ AI Pricing Wrapper: http://localhost:8002
echo ✅ GNN Inference: http://localhost:8001
echo ✅ Logistics Wrapper: http://localhost:8003
echo ✅ Multi-Hop Symbiosis: http://localhost:5003
echo ✅ Logistics Cost Service: http://localhost:5006
echo ✅ Optimize DeepSeek R1: http://localhost:5005
echo ✅ Financial Analysis Engine: http://localhost:5017
echo.
echo 🎯 TOTAL REAL MICROSERVICES: 25
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