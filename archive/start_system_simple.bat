@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🚀 SYMBIOFLOWS - SIMPLIFIED SYSTEM STARTUP
echo ========================================
echo 🎯 Starting Core Services for Testing
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

echo %BLUE%🔍 Starting simplified system startup...%RESET%
echo.

REM ========================================
REM BASIC PREREQUISITE CHECK
REM ========================================

echo %YELLOW%Checking basic prerequisites...%RESET%

REM Check Node.js (warn but don't stop)
node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Node.js available%RESET%
) else (
    echo %YELLOW%   ⚠️ Node.js not found - some services may not work%RESET%
)

REM Check Python (warn but don't stop)
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Python available%RESET%
) else (
    echo %YELLOW%   ⚠️ Python not found - some services may not work%RESET%
)

REM Check npm (warn but don't stop)
npm --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ npm available%RESET%
) else (
    echo %YELLOW%   ⚠️ npm not found - some services may not work%RESET%
)

echo.

REM ========================================
REM CORE SERVICES STARTUP
REM ========================================

echo %BLUE%📦 Starting Core Services...%RESET%

REM 1. Try to start Backend (Node.js) on port 3001
echo %YELLOW%1. Attempting to start Node.js Backend (port 3001)...%RESET%
if exist "backend\package.json" (
    start "Backend API" cmd /k "cd backend && npm run dev"
    echo %GREEN%   ✅ Backend startup initiated%RESET%
) else (
    echo %RED%   ❌ backend\package.json not found%RESET%
)

REM 2. Try to start Frontend (React) on port 5173
echo %YELLOW%2. Attempting to start React Frontend (port 5173)...%RESET%
if exist "frontend\package.json" (
    start "Frontend" cmd /k "cd frontend && npm run dev"
    echo %GREEN%   ✅ Frontend startup initiated%RESET%
) else (
    echo %RED%   ❌ frontend\package.json not found%RESET%
)

timeout /t 5 /nobreak >nul

REM ========================================
REM PYTHON SERVICES STARTUP
REM ========================================

echo %BLUE%🤖 Starting Python Services...%RESET%

REM 3. Try to start AI Gateway
echo %YELLOW%3. Attempting to start AI Gateway...%RESET%
if exist "ai_service_flask\ai_gateway.py" (
    start "AI Gateway" cmd /k "cd ai_service_flask && python ai_gateway.py"
    echo %GREEN%   ✅ AI Gateway startup initiated%RESET%
) else (
    echo %RED%   ❌ ai_service_flask\ai_gateway.py not found%RESET%
)

REM 4. Try to start Advanced Analytics Service
echo %YELLOW%4. Attempting to start Advanced Analytics Service...%RESET%
if exist "ai_service_flask\advanced_analytics_service.py" (
    start "Advanced Analytics" cmd /k "cd ai_service_flask && python advanced_analytics_service.py"
    echo %GREEN%   ✅ Advanced Analytics startup initiated%RESET%
) else (
    echo %RED%   ❌ ai_service_flask\advanced_analytics_service.py not found%RESET%
)

timeout /t 3 /nobreak >nul

REM ========================================
REM BACKEND PYTHON SERVICES
REM ========================================

echo %BLUE%🔧 Starting Backend Python Services...%RESET%

REM 5. Try to start Adaptive Onboarding Server
echo %YELLOW%5. Attempting to start Adaptive Onboarding Server...%RESET%
if exist "backend\adaptive_onboarding_server.py" (
    start "Adaptive Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"
    echo %GREEN%   ✅ Adaptive Onboarding startup initiated%RESET%
) else (
    echo %RED%   ❌ backend\adaptive_onboarding_server.py not found%RESET%
)

REM 6. Try to start AI Listings Generator
echo %YELLOW%6. Attempting to start AI Listings Generator...%RESET%
if exist "backend\ai_listings_generator.py" (
    start "AI Listings Generator" cmd /k "cd backend && python ai_listings_generator.py"
    echo %GREEN%   ✅ AI Listings Generator startup initiated%RESET%
) else (
    echo %RED%   ❌ backend\ai_listings_generator.py not found%RESET%
)

REM 7. Try to start MaterialsBERT Service
echo %YELLOW%7. Attempting to start MaterialsBERT Service...%RESET%
if exist "backend\materials_bert_service.py" (
    start "MaterialsBERT Service" cmd /k "cd backend && python materials_bert_service.py"
    echo %GREEN%   ✅ MaterialsBERT Service startup initiated%RESET%
) else (
    echo %RED%   ❌ backend\materials_bert_service.py not found%RESET%
)

timeout /t 3 /nobreak >nul

REM ========================================
REM ROOT SERVICES
REM ========================================

echo %BLUE%🌐 Starting Root Services...%RESET%

REM 8. Try to start Enhanced AI Generator
echo %YELLOW%8. Attempting to start Enhanced AI Generator...%RESET%
if exist "enhanced_ai_generator.py" (
    start "Enhanced AI Generator" cmd /k "python enhanced_ai_generator.py"
    echo %GREEN%   ✅ Enhanced AI Generator startup initiated%RESET%
) else (
    echo %RED%   ❌ enhanced_ai_generator.py not found%RESET%
)

REM 9. Try to start Financial Analysis Engine
echo %YELLOW%9. Attempting to start Financial Analysis Engine...%RESET%
if exist "financial_analysis_engine.py" (
    start "Financial Analysis Engine" cmd /k "python financial_analysis_engine.py"
    echo %GREEN%   ✅ Financial Analysis Engine startup initiated%RESET%
) else (
    echo %RED%   ❌ financial_analysis_engine.py not found%RESET%
)

timeout /t 3 /nobreak >nul

REM ========================================
REM SUMMARY
REM ========================================

echo.
echo ========================================
echo 🎉 SIMPLIFIED SYSTEM STARTUP COMPLETE!
echo ========================================
echo.
echo 📊 SYSTEM SUMMARY:
echo ✅ Backend API: http://localhost:3001 (if Node.js available)
echo ✅ Frontend: http://localhost:5173 (if Node.js available)
echo ✅ AI Gateway: http://localhost:5000 (if Python available)
echo ✅ Advanced Analytics: http://localhost:5004 (if Python available)
echo ✅ Adaptive Onboarding: http://localhost:5003 (if Python available)
echo ✅ AI Listings Generator: http://localhost:5001 (if Python available)
echo ✅ MaterialsBERT Service: http://localhost:5002 (if Python available)
echo ✅ Enhanced AI Generator: http://localhost:5005 (if Python available)
echo ✅ Financial Analysis Engine: http://localhost:5006 (if Python available)
echo.
echo 🎯 SERVICES STARTED IN SEPARATE WINDOWS
echo.
echo 💡 Each service window shows real-time logs and errors
echo 💡 Check each window for any startup errors
echo 💡 Services may fail if prerequisites are missing
echo.
echo 📋 Next Steps:%RESET%
echo   1. Check each service window for errors
echo   2. Install missing prerequisites if needed
echo   3. Run .\diagnose_prerequisites.bat for detailed diagnostics
echo   4. Test individual services manually
echo.
echo 🛑 Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped. 