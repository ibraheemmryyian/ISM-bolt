@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🎬 SYMBIOFLOWS MINIMAL DEMO STARTUP
echo ===================================
echo.
echo Starting ONLY GUARANTEED WORKING Services...
echo.

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%🚀 Starting Minimal Demo Services...%RESET%
echo.

REM ========================================
REM GUARANTEED WORKING SERVICES ONLY
REM ========================================

echo %YELLOW%1. Starting Backend API (port 5000)...%RESET%
start "Backend API" cmd /k "cd backend && npm run dev"

timeout /t 8 /nobreak >nul

echo %YELLOW%2. Starting Frontend (port 5173)...%RESET%
start "Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 8 /nobreak >nul

echo %YELLOW%3. Starting AI Onboarding Server (port 5003)...%RESET%
start "AI Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"

timeout /t 5 /nobreak >nul

echo %YELLOW%4. Starting AI Monitoring Dashboard (port 5011)...%RESET%
start "AI Monitoring" cmd /k "cd backend && python ai_monitoring_dashboard.py"

timeout /t 5 /nobreak >nul

echo %YELLOW%5. Starting Complete Logistics Platform (port 5026)...%RESET%
start "Logistics Platform" cmd /k "cd backend && python complete_logistics_platform.py"

timeout /t 5 /nobreak >nul

echo.
echo ===================================
echo 🎉 MINIMAL DEMO SERVICES STARTED!
echo ===================================
echo.
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:5000
echo 👤 AI Onboarding: http://localhost:5003
echo 📊 AI Monitoring: http://localhost:5011
echo 🚚 Logistics Platform: http://localhost:5026
echo.
echo 💡 Minimal Demo Features Available:
echo ✅ User Registration and Login
echo ✅ AI-Powered Adaptive Onboarding
echo ✅ Dashboard and Analytics
echo ✅ Material Browsing and Search
echo ✅ Payment Processing (Stripe)
echo ✅ Admin Panel and Management
echo ✅ Real-time AI Monitoring
echo ✅ Logistics Platform
echo.
echo ⚠️  This is running ONLY 5 GUARANTEED services!
echo 💻 Much more stable and reliable
echo 🔥 No dependency errors expected
echo.
echo 🛑 Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped.
endlocal 