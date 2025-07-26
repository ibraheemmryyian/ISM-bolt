@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🎬 SYMBIOFLOWS COMPLETE DEMO STARTUP
echo ===================================
echo.
echo Starting ALL Essential Services for Complete Demo...
echo.

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%🚀 Starting Complete Demo Services...%RESET%
echo.

REM ========================================
REM CORE INFRASTRUCTURE (ESSENTIAL)
REM ========================================

echo %YELLOW%1. Starting Backend API (port 5000)...%RESET%
start "Backend API" cmd /k "cd backend && npm run dev"

timeout /t 3 /nobreak >nul

echo %YELLOW%2. Starting Frontend (port 5173)...%RESET%
start "Frontend" cmd /k "cd frontend && npm run dev"

timeout /t 3 /nobreak >nul

REM ========================================
REM AI ONBOARDING & USER EXPERIENCE
REM ========================================

echo %YELLOW%3. Starting AI Onboarding Server (port 5003)...%RESET%
start "AI Onboarding" cmd /k "cd backend && python adaptive_onboarding_server.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%4. Starting AI Feedback Orchestrator...%RESET%
start "AI Feedback" cmd /k "cd backend && python ai_feedback_orchestrator.py"

timeout /t 3 /nobreak >nul

REM ========================================
REM CORE AI SERVICES (ESSENTIAL FOR DEMO)
REM ========================================

echo %YELLOW%5. Starting AI Listings Generator...%RESET%
start "AI Listings" cmd /k "cd backend && python ai_listings_generator.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%6. Starting Revolutionary AI Matching...%RESET%
start "AI Matching" cmd /k "cd backend && python revolutionary_ai_matching.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%7. Starting Advanced Description Generator...%RESET%
start "Description Gen" cmd /k "cd backend && python advanced_description_generator.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%8. Starting Material Diversity Manager...%RESET%
start "Diversity Manager" cmd /k "cd backend && python material_diversity_manager.py"

timeout /t 3 /nobreak >nul

REM ========================================
REM QUALITY & ASSESSMENT SERVICES
REM ========================================

echo %YELLOW%9. Starting Advanced Quality Assessment...%RESET%
start "Quality Assessment" cmd /k "cd backend && python advanced_quality_assessment_engine.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%10. Starting Match Quality Analyzer...%RESET%
start "Match Quality" cmd /k "cd backend && python match_quality_analyzer.py"

timeout /t 3 /nobreak >nul

REM ========================================
REM BUSINESS INTELLIGENCE SERVICES
REM ========================================

echo %YELLOW%11. Starting Industrial Intelligence Engine...%RESET%
start "Industrial Intel" cmd /k "cd backend && python industrial_intelligence_engine.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%12. Starting Proactive Opportunity Engine...%RESET%
start "Opportunity Engine" cmd /k "cd backend && python proactive_opportunity_engine.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%13. Starting Impact Forecasting...%RESET%
start "Impact Forecasting" cmd /k "cd backend && python impact_forecasting.py"

timeout /t 3 /nobreak >nul

REM ========================================
REM LOGISTICS & OPERATIONS
REM ========================================

echo %YELLOW%14. Starting Complete Logistics Platform...%RESET%
start "Logistics Platform" cmd /k "cd backend && python complete_logistics_platform.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%15. Starting Logistics Orchestration...%RESET%
start "Logistics Orchestration" cmd /k "cd backend && python logistics_orchestration_engine.py"

timeout /t 3 /nobreak >nul

REM ========================================
REM SYSTEM INFRASTRUCTURE
REM ========================================

echo %YELLOW%16. Starting Advanced Orchestration Engine...%RESET%
start "Orchestration Engine" cmd /k "cd backend && python advanced_orchestration_engine.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%17. Starting Complete System Integration...%RESET%
start "System Integration" cmd /k "cd backend && python complete_system_integration.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%18. Starting AI Monitoring Dashboard...%RESET%
start "AI Monitoring" cmd /k "cd backend && python ai_monitoring_dashboard.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%19. Starting Performance Optimizer...%RESET%
start "Performance Optimizer" cmd /k "cd backend && python performance_optimizer.py"

timeout /t 3 /nobreak >nul

REM ========================================
REM AI SERVICE INTEGRATION
REM ========================================

echo %YELLOW%20. Starting AI Service Integration...%RESET%
start "AI Service Integration" cmd /k "cd backend && python ai_service_integration.py"

timeout /t 3 /nobreak >nul

echo %YELLOW%21. Starting Advanced AI Prompts Service...%RESET%
start "AI Prompts" cmd /k "cd backend && python advanced_ai_prompts_service.py"

timeout /t 3 /nobreak >nul

echo.
echo ===================================
echo 🎉 COMPLETE DEMO SERVICES STARTED!
echo ===================================
echo.
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend API: http://localhost:5000
echo 👤 AI Onboarding: http://localhost:5003
echo 📊 AI Monitoring: Running
echo.
echo 💡 Complete Demo Features Available:
echo ✅ User Registration and Login
echo ✅ AI-Powered Adaptive Onboarding
echo ✅ Complete Dashboard and Analytics
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
echo ✅ Performance Optimization
echo ✅ Advanced AI Prompts
echo ✅ Material Diversity Management
echo ✅ Match Quality Analysis
echo.
echo ⚠️  WARNING: This is running 21 services!
echo 💻 Your laptop will be under heavy load
echo 🔥 Monitor CPU and memory usage
echo.
echo 🛑 Press any key to stop all services...
pause

echo.
echo 🛑 Stopping all services...
taskkill /f /im node.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1
echo ✅ All services stopped.
endlocal 