@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🎬 SYMBIOFLOWS DEMO STARTUP
echo ===========================
echo.
echo Starting 5 Core Services for Demo...
echo.

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%🚀 Starting Core Demo Services...%RESET%
echo.

REM 1. Backend API (Node.js) - Essential for everything
echo %YELLOW%1. Starting Backend API (port 5000)...%RESET%
start "Backend API" cmd /k "cd backend && npm run dev"

timeout /t 3 /nobreak >nul

REM 2. Frontend (React) - User interface
echo %YELLOW%2. Starting Frontend (port 5173)...%RESET%
start "Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 3 /nobreak >nul

REM 3. AI Onboarding Server - Essential for user onboarding
echo %YELLOW%3. Starting AI Onboarding Server (port 5003)...%RESET%
start "AI Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"

timeout /t 3 /nobreak >nul

REM 4. AI Listings Generator - Core AI feature
echo %YELLOW%4. Starting AI Listings Generator...%RESET%
start "AI Listings" cmd /k "cd backend && python ai_listings_generator.py"

timeout /t 3 /nobreak >nul

REM 5. Revolutionary AI Matching - Core AI feature
echo %YELLOW%5. Starting Revolutionary AI Matching...%RESET%
start "AI Matching" cmd /k "cd backend && python revolutionary_ai_matching.py"

echo.
echo ===========================
echo 🎉 DEMO SERVICES STARTED!
echo ===========================
echo.
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:5000
echo 👤 AI Onboarding: http://localhost:5003
echo 🤖 AI Listings: Running
echo 🧠 AI Matching: Running
echo.
echo 💡 Demo Features Available:
echo ✅ User Registration and Login
echo ✅ AI-Powered Onboarding
echo ✅ Dashboard and Analytics
echo ✅ Material Browsing
echo ✅ AI-Generated Listings
echo ✅ AI Matching Algorithm
echo ✅ Payment Processing (Stripe)
echo ✅ Admin Panel
echo.
echo 🛑 Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped.
endlocal 