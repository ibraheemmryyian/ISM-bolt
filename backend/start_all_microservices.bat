@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 SymbioFlows - ALL 19 MICRO-SERVICES STARTUP
echo ================================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Starting ALL 19 AI micro-services in separate windows...%RESET%
echo.

REM Verify all files exist first
echo %YELLOW%Verifying all required files exist...%RESET%
call verify_files.bat
if %errorlevel% neq 0 (
    echo %RED%❌ File verification failed. Cannot start services.%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ All files verified successfully!%RESET%
echo.

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
REM CORE SERVICES
REM ========================================

echo %BLUE%📦 Starting Core Services...%RESET%

REM 1. Backend (Node.js) on port 3000
echo %YELLOW%1. Starting Node.js Backend (port 3000)...%RESET%
start "Backend API" cmd /k "npm run dev"

REM 2. Frontend (React) on port 5173
echo %YELLOW%2. Starting React Frontend (port 5173)...%RESET%
start "Frontend" cmd /k "cd ..\frontend && npm run dev"

timeout /t 8 /nobreak >nul

REM ========================================
REM AI GATEWAY & CORE AI SERVICES
REM ========================================

echo %BLUE%🤖 Starting AI Gateway & Core AI Services...%RESET%

REM 3. AI Gateway (Python) on port 5000
echo %YELLOW%3. Starting AI Gateway (port 5000)...%RESET%
start "AI Gateway" cmd /k "cd ..\ai_service_flask && python ai_gateway.py"

REM 4. AI Pricing Service (Python) on port 5005
echo %YELLOW%4. Starting AI Pricing Service (port 5005)...%RESET%
start "AI Pricing Service" cmd /k "cd ..\ai_service_flask && python ai_pricing_service_wrapper.py"

REM 5. Logistics Service (Python) on port 5006
echo %YELLOW%5. Starting Logistics Service (port 5006)...%RESET%
start "Logistics Service" cmd /k "cd ..\ai_service_flask && python logistics_service_wrapper.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM ADVANCED AI MODULES
REM ========================================

echo %BLUE%🧠 Starting Advanced AI Modules...%RESET%

REM 6. GNN Inference Service (Python) on port 5007
echo %YELLOW%6. Starting GNN Inference Service (port 5007)...%RESET%
start "GNN Service" cmd /k "cd ..\ai_service_flask && python gnn_inference_service.py"

REM 7. Multi-Hop Symbiosis Service (Python) on port 5008
echo %YELLOW%7. Starting Multi-Hop Symbiosis Service (port 5008)...%RESET%
start "Multi-Hop Symbiosis" cmd /k "cd ..\ai_service_flask && python multi_hop_symbiosis_service.py"

REM 8. Advanced Analytics Service (Python) on port 5009
echo %YELLOW%8. Starting Advanced Analytics Service (port 5009)...%RESET%
start "Advanced Analytics" cmd /k "cd ..\ai_service_flask && python advanced_analytics_service.py"

REM 9. Federated Learning Service (Python) on port 5010
echo %YELLOW%9. Starting Federated Learning Service (port 5010)...%RESET%
start "Federated Learning" cmd /k "cd ..\ai_service_flask && python federated_learning_service.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM BACKEND AI MODULES
REM ========================================

echo %BLUE%🔧 Starting Backend AI Modules...%RESET%

REM 10. Proactive Opportunity Engine (Python) on port 5011
echo %YELLOW%10. Starting Proactive Opportunity Engine (port 5011)...%RESET%
start "Proactive Engine" cmd /k "python proactive_opportunity_engine.py"

REM 11. GNN Reasoning Engine (Python) on port 5012
echo %YELLOW%11. Starting GNN Reasoning Engine (port 5012)...%RESET%
start "GNN Reasoning" cmd /k "python gnn_reasoning_engine.py"

REM 12. AI Pricing Orchestrator (Python) on port 5013
echo %YELLOW%12. Starting AI Pricing Orchestrator (port 5013)...%RESET%
start "AI Pricing Orchestrator" cmd /k "python ai_pricing_orchestrator.py"

REM 13. Materials BERT Service (Python) on port 5014
echo %YELLOW%13. Starting Materials BERT Service (port 5014)...%RESET%
start "Materials BERT" cmd /k "python materials_bert_service_advanced.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM SPECIALIZED AI SERVICES
REM ========================================

echo %BLUE%🎯 Starting Specialized AI Services...%RESET%

REM 14. Dynamic Materials Integration (Python) on port 5015
echo %YELLOW%14. Starting Dynamic Materials Integration (port 5015)...%RESET%
start "Dynamic Materials" cmd /k "python dynamic_materials_integration_service.py"

REM 15. AI Production Orchestrator (Python) on port 5016
echo %YELLOW%15. Starting AI Production Orchestrator (port 5016)...%RESET%
start "AI Production Orchestrator" cmd /k "python ai_production_orchestrator.py"

REM 16. AI Monitoring Dashboard (Python) on port 5017
echo %YELLOW%16. Starting AI Monitoring Dashboard (port 5017)...%RESET%
start "AI Monitoring" cmd /k "python ai_monitoring_dashboard.py"

REM 17. System Health Monitor (Python) on port 5018
echo %YELLOW%17. Starting System Health Monitor (port 5018)...%RESET%
start "System Health Monitor" cmd /k "python system_health_monitor.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM ONBOARDING & ADAPTIVE SERVICES
REM ========================================

echo %BLUE%👤 Starting Onboarding & Adaptive Services...%RESET%

REM 18. Adaptive Onboarding Server (Python) on port 5019
echo %YELLOW%18. Starting Adaptive Onboarding Server (port 5019)...%RESET%
start "Adaptive Onboarding" cmd /k "python adaptive_onboarding_server.py"

REM 19. AI Feedback Orchestrator (Python) on port 5020
echo %YELLOW%19. Starting AI Feedback Orchestrator (port 5020)...%RESET%
start "AI Feedback Orchestrator" cmd /k "python ai_feedback_orchestrator.py"

timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo 🎉 ALL 19 MICRO-SERVICES STARTED!
echo ========================================
echo.
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:3000
echo 🤖 AI Gateway: http://localhost:5000
echo 🧠 GNN Service: http://localhost:5007
echo 🔄 Multi-Hop: http://localhost:5008
echo 📊 Analytics: http://localhost:5009
echo 🎓 Federated: http://localhost:5010
echo 🎯 Proactive: http://localhost:5011
echo 🧮 GNN Reasoning: http://localhost:5012
echo 💰 Pricing: http://localhost:5013
echo 🧪 Materials BERT: http://localhost:5014
echo 🔗 Dynamic Materials: http://localhost:5015
echo 🏭 Production: http://localhost:5016
echo 📈 Monitoring: http://localhost:5017
echo 🏥 Health Monitor: http://localhost:5018
echo 👤 Adaptive Onboarding: http://localhost:5019
echo 💬 Feedback: http://localhost:5020
echo.
echo 💡 Each service runs in its own window for easy debugging!
echo 🛑 Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped. 