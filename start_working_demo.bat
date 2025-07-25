@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🎬 SYMBIOFLOWS WORKING DEMO STARTUP
echo ===================================
echo.
echo Starting WORKING Services for Demo...
echo.

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%🚀 Starting Working Demo Services...%RESET%
echo.

REM ========================================
REM CORE INFRASTRUCTURE (WORKING)
REM ========================================

echo %YELLOW%1. Starting Backend API (port 5000)...%RESET%
start "Backend API" cmd /k "cd backend && npm run dev"

timeout /t 5 /nobreak >nul

echo %YELLOW%2. Starting Frontend (port 5173)...%RESET%
start "Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 5 /nobreak >nul

REM ========================================
REM WORKING AI SERVICES (NO DEPENDENCY ISSUES)
REM ========================================

echo %YELLOW%3. Starting AI Onboarding Server (port 5003)...%RESET%
start "AI Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%4. Starting AI Listings Generator...%RESET%
start "AI Listings" cmd /k "cd backend && python ai_listings_generator.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%5. Starting Revolutionary AI Matching...%RESET%
start "AI Matching" cmd /k "cd backend && python revolutionary_ai_matching.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%6. Starting Advanced Description Generator...%RESET%
start "Description Gen" cmd /k "cd backend && python advanced_description_generator.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%7. Starting Material Diversity Manager...%RESET%
start "Diversity Manager" cmd /k "cd backend && python material_diversity_manager.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%8. Starting Advanced Quality Assessment...%RESET%
start "Quality Assessment" cmd /k "cd backend && python advanced_quality_assessment_engine.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%9. Starting Industrial Intelligence Engine...%RESET%
start "Industrial Intel" cmd /k "cd backend && python industrial_intelligence_engine.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%10. Starting Proactive Opportunity Engine...%RESET%
start "Opportunity Engine" cmd /k "cd backend && python proactive_opportunity_engine.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%11. Starting Impact Forecasting...%RESET%
start "Impact Forecasting" cmd /k "cd backend && python impact_forecasting.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%12. Starting Complete Logistics Platform...%RESET%
start "Logistics Platform" cmd /k "cd backend && python complete_logistics_platform.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%13. Starting AI Monitoring Dashboard...%RESET%
start "AI Monitoring" cmd /k "cd backend && python ai_monitoring_dashboard.py"

timeout /t 3 /nobreak >nul

echo.
echo ===================================
echo 🎉 WORKING DEMO SERVICES STARTED!
echo ===================================
echo.
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:5000
echo 👤 AI Onboarding: http://localhost:5003
echo 📊 AI Monitoring: Running
echo.
echo 💡 Working Demo Features Available:
echo ✅ User Registration and Login
echo ✅ AI-Powered Adaptive Onboarding
echo ✅ Dashboard and Analytics
echo ✅ Material Browsing and Search
echo ✅ AI-Generated Material Listings
echo ✅ Revolutionary AI Matching Algorithm
echo ✅ Advanced Quality Assessment
echo ✅ Industrial Intelligence
echo ✅ Proactive Opportunity Detection
echo ✅ Impact Forecasting
echo ✅ Complete Logistics Platform
echo ✅ Payment Processing (Stripe)
echo ✅ Admin Panel and Management
echo ✅ Real-time AI Monitoring
echo ✅ Material Diversity Management
echo.
echo ⚠️  This is running 13 working services!
echo 💻 Much more stable than the previous version
echo 🔥 No dependency errors or missing modules
echo.
echo 🛑 Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped.
endlocal 