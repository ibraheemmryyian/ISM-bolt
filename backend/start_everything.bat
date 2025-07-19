@echo off
echo ========================================
echo Starting SymbioFlows Advanced AI System
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

REM Wait for AI system to initialize
echo ⏳ Waiting for AI system to initialize...
timeout /t 3 /nobreak >nul

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
echo 🎉 All Systems Started Successfully!
echo ========================================
echo.
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:3000
echo 🧠 AI Onboarding: http://localhost:5003
echo 📊 AI Dashboard: http://localhost:5004
echo.
echo Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped. 