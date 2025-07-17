@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 ISM AI Platform - Optimized AI System Startup
echo ================================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Starting optimized AI system with intelligent orchestration...%RESET%
echo.

REM Set Python path to include backend directory
set PYTHONPATH=%CD%\backend;%PYTHONPATH%

REM Check prerequisites
echo %YELLOW%Checking prerequisites...%RESET%
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Node.js is not installed or not in PATH%RESET%
    pause
    exit /b 1
)

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Python is not installed or not in PATH%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ Prerequisites OK%RESET%
echo.

REM ========================================
REM PHASE 1: CORE INFRASTRUCTURE
REM ========================================

echo %BLUE%🏗️ Phase 1: Starting Core Infrastructure...%RESET%

REM 1. Backend (Node.js) - Main API Gateway
echo %YELLOW%1. Starting Backend API Gateway (port 3000)...%RESET%
start "Backend API Gateway" cmd /k "cd backend && npm run dev"

REM 2. Frontend (React) - User Interface
echo %YELLOW%2. Starting Frontend (port 5173)...%RESET%
start "Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 10 /nobreak >nul

REM ========================================
REM PHASE 2: AI GATEWAY & ORCHESTRATION
REM ========================================

echo %BLUE%🤖 Phase 2: Starting AI Gateway and Orchestration...%RESET%

REM 3. AI Gateway (Python) - Main AI Service Router
echo %YELLOW%3. Starting AI Gateway (port 5000)...%RESET%
start "AI Gateway" cmd /k "cd ai_service_flask && python ai_gateway.py"

REM 4. AI Production Orchestrator (Python) - Main Coordinator
echo %YELLOW%4. Starting AI Production Orchestrator (port 5016)...%RESET%
start "AI Production Orchestrator" cmd /k "set PYTHONPATH=%CD%\backend;%PYTHONPATH% && python backend/ai_production_orchestrator.py"

timeout /t 8 /nobreak >nul

REM ========================================
REM PHASE 3: CORE AI SERVICES
REM ========================================

echo %BLUE%🧠 Phase 3: Starting Core AI Services...%RESET%

REM 5. AI Pricing Service (Python) - Pricing Intelligence
echo %YELLOW%5. Starting AI Pricing Service (port 5005)...%RESET%
start "AI Pricing Service" cmd /k "cd ai_service_flask && python ai_pricing_service_wrapper.py"

REM 6. Logistics Service (Python) - Supply Chain Intelligence
echo %YELLOW%6. Starting Logistics Service (port 5006)...%RESET%
start "Logistics Service" cmd /k "cd ai_service_flask && python logistics_service_wrapper.py"

REM 7. GNN Reasoning Engine (Python) - Graph Neural Networks
echo %YELLOW%7. Starting GNN Reasoning Engine (port 5012)...%RESET%
start "GNN Reasoning Engine" cmd /k "set PYTHONPATH=%CD%\backend;%PYTHONPATH% && python backend/gnn_reasoning_engine.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM PHASE 4: ADVANCED AI MODULES
REM ========================================

echo %BLUE%🎯 Phase 4: Starting Advanced AI Modules...%RESET%

REM 8. Multi-Hop Symbiosis Service (Python) - Complex Matching
echo %YELLOW%8. Starting Multi-Hop Symbiosis Service (port 5008)...%RESET%
start "Multi-Hop Symbiosis" cmd /k "cd ai_service_flask && python multi_hop_symbiosis_service.py"

REM 9. Proactive Opportunity Engine (Python) - Predictive Analytics
echo %YELLOW%9. Starting Proactive Opportunity Engine (port 5011)...%RESET%
start "Proactive Engine" cmd /k "set PYTHONPATH=%CD%\backend;%PYTHONPATH% && python backend/proactive_opportunity_engine.py"

REM 10. Materials BERT Service (Python) - Materials Intelligence
echo %YELLOW%10. Starting Materials BERT Service (port 5014)...%RESET%
start "Materials BERT" cmd /k "set PYTHONPATH=%CD%\backend;%PYTHONPATH% && python backend/materials_bert_service_advanced.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM PHASE 5: MONITORING & OPTIMIZATION
REM ========================================

echo %BLUE%📊 Phase 5: Starting Monitoring and Optimization...%RESET%

REM 11. System Health Monitor (Python) - System Monitoring
echo %YELLOW%11. Starting System Health Monitor (port 5018)...%RESET%
start "System Health Monitor" cmd /k "set PYTHONPATH=%CD%\backend;%PYTHONPATH% && python backend/system_health_monitor.py"

REM 12. AI Monitoring Dashboard (Python) - AI Performance Monitoring
echo %YELLOW%12. Starting AI Monitoring Dashboard (port 5017)...%RESET%
start "AI Monitoring Dashboard" cmd /k "set PYTHONPATH=%CD%\backend;%PYTHONPATH% && python backend/ai_monitoring_dashboard.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM PHASE 6: ONBOARDING & FEEDBACK
REM ========================================

echo %BLUE%👤 Phase 6: Starting Onboarding and Feedback...%RESET%

REM 13. Adaptive Onboarding Server (Python) - User Onboarding
echo %YELLOW%13. Starting Adaptive Onboarding Server (port 5019)...%RESET%
start "Adaptive Onboarding" cmd /k "set PYTHONPATH=%CD%\backend;%PYTHONPATH% && python backend/adaptive_onboarding_server.py"

REM 14. AI Feedback Orchestrator (Python) - Feedback Processing
echo %YELLOW%14. Starting AI Feedback Orchestrator (port 5020)...%RESET%
start "AI Feedback Orchestrator" cmd /k "set PYTHONPATH=%CD%\backend;%PYTHONPATH% && python backend/ai_feedback_orchestrator.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM WAIT FOR ORCHESTRATION
REM ========================================

echo %YELLOW%⏳ Waiting for system orchestration...%RESET%
timeout /t 20 /nobreak >nul

REM ========================================
REM HEALTH CHECKS
REM ========================================

echo %YELLOW%🔍 Checking system health...%RESET%

REM Core Infrastructure
curl -s http://localhost:3000/api/health >nul 2>&1
if %errorlevel% equ 0 (echo %GREEN%✅ Backend API Gateway (3000): Running%RESET%) else (echo %RED%❌ Backend API Gateway (3000): Not running%RESET%)

curl -s http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (echo %GREEN%✅ Frontend (5173): Running%RESET%) else (echo %RED%❌ Frontend (5173): Not running%RESET%)

REM AI Gateway & Orchestration
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (echo %GREEN%✅ AI Gateway (5000): Running%RESET%) else (echo %RED%❌ AI Gateway (5000): Not running%RESET%)

curl -s http://localhost:5016/health >nul 2>&1
if %errorlevel% equ 0 (echo %GREEN%✅ AI Production Orchestrator (5016): Running%RESET%) else (echo %RED%❌ AI Production Orchestrator (5016): Not running%RESET%)

REM Core AI Services
curl -s http://localhost:5005/health >nul 2>&1
if %errorlevel% equ 0 (echo %GREEN%✅ AI Pricing Service (5005): Running%RESET%) else (echo %RED%❌ AI Pricing Service (5005): Not running%RESET%)

curl -s http://localhost:5006/health >nul 2>&1
if %errorlevel% equ 0 (echo %GREEN%✅ Logistics Service (5006): Running%RESET%) else (echo %RED%❌ Logistics Service (5006): Not running%RESET%)

echo.
echo %GREEN%🎉 Optimized AI System Started!%RESET%
echo.
echo %BLUE%📊 Service Architecture:%RESET%
echo %GREEN%• Backend API Gateway: http://localhost:3000%RESET%
echo %GREEN%• Frontend: http://localhost:5173%RESET%
echo %GREEN%• AI Gateway: http://localhost:5000%RESET%
echo %GREEN%• AI Production Orchestrator: http://localhost:5016%RESET%
echo %GREEN%• AI Pricing Service: http://localhost:5005%RESET%
echo %GREEN%• Logistics Service: http://localhost:5006%RESET%
echo %GREEN%• GNN Reasoning Engine: http://localhost:5012%RESET%
echo %GREEN%• Multi-Hop Symbiosis: http://localhost:5008%RESET%
echo %GREEN%• Proactive Engine: http://localhost:5011%RESET%
echo %GREEN%• Materials BERT: http://localhost:5014%RESET%
echo %GREEN%• System Health Monitor: http://localhost:5018%RESET%
echo %GREEN%• AI Monitoring Dashboard: http://localhost:5017%RESET%
echo %GREEN%• Adaptive Onboarding: http://localhost:5019%RESET%
echo %GREEN%• AI Feedback Orchestrator: http://localhost:5020%RESET%
echo.
echo %BLUE%🧪 Ready for comprehensive testing!%RESET%
echo %YELLOW%Run: python run_system_tests.py --test-type all%RESET%
echo.
echo %BLUE%📋 Optimization Features:%RESET%
echo %GREEN%• Phased startup for optimal resource usage%RESET%
echo %GREEN%• Intelligent orchestration through AI Gateway%RESET%
echo %GREEN%• Centralized monitoring and health checks%RESET%
echo %GREEN%• Reduced from 19 to 14 essential services%RESET%
echo %GREEN%• Better service communication patterns%RESET%
echo.
echo %BLUE%📋 Total Services Started: 14 (Optimized)%RESET%
echo %BLUE%📋 Ports Used: 3000, 5173, 5000, 5005-5006, 5008, 5011-5012, 5014, 5016-5020%RESET%
echo.
pause 