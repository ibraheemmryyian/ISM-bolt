@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 SymbioFlows - ROBUST SYSTEM STARTUP
echo ======================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Starting system with robust error handling...%RESET%
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
REM CORE SERVICES (ALWAYS WORK)
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
REM ROBUST AI SERVICES (WITH FALLBACKS)
REM ========================================

echo %BLUE%🤖 Starting Robust AI Services...%RESET%

REM 3. Adaptive Onboarding Server (Python) on port 5019
echo %YELLOW%3. Starting Adaptive Onboarding Server (port 5019)...%RESET%
start "Adaptive Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"

REM 4. System Health Monitor (Python) on port 5018
echo %YELLOW%4. Starting System Health Monitor (port 5018)...%RESET%
start "System Health Monitor" cmd /k "cd backend && python system_health_monitor.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM CONDITIONAL AI SERVICES
REM ========================================

echo %BLUE%🧠 Starting Conditional AI Services...%RESET%

REM 5. AI Production Orchestrator (Python) on port 5016
echo %YELLOW%5. Starting AI Production Orchestrator (port 5016)...%RESET%
start "AI Production Orchestrator" cmd /k "cd backend && python start_production_ai_system.py"

REM 6. AI Monitoring Dashboard (Python) on port 5017
echo %YELLOW%6. Starting AI Monitoring Dashboard (port 5017)...%RESET%
start "AI Monitoring" cmd /k "cd backend && python ai_monitoring_dashboard.py"

timeout /t 5 /nobreak >nul

REM ========================================
REM OPTIONAL AI SERVICES (MAY FAIL)
REM ========================================

echo %BLUE%🎯 Starting Optional AI Services...%RESET%

REM 7. AI Gateway (Python) on port 5000
echo %YELLOW%7. Starting AI Gateway (port 5000)...%RESET%
start "AI Gateway" cmd /k "cd ai_service_flask && python ai_gateway.py"

REM 8. AI Pricing Service (Python) on port 5005
echo %YELLOW%8. Starting AI Pricing Service (port 5005)...%RESET%
start "AI Pricing Service" cmd /k "cd ai_service_flask && python ai_pricing_service_wrapper.py"

timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo 🎉 ROBUST SYSTEM STARTED!
echo ========================================
echo.
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:3000
echo 👤 Adaptive Onboarding: http://localhost:5019
echo 🏥 Health Monitor: http://localhost:5018
echo 🏭 Production Orchestrator: http://localhost:5016
echo 📈 Monitoring Dashboard: http://localhost:5017
echo 🤖 AI Gateway: http://localhost:5000
echo 💰 AI Pricing: http://localhost:5005
echo.
echo 💡 Services with import errors will show in their windows
echo 🛑 Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped. 