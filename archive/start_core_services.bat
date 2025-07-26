@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 ISM AI Platform - Core Services Startup
echo ==========================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Starting core services only...%RESET%
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
REM CORE SERVICES ONLY
REM ========================================

echo %BLUE%📦 Starting Core Services...%RESET%

REM 1. Backend (Node.js) on port 3000
echo %YELLOW%1. Starting Backend (port 3000)...%RESET%
start "Backend" cmd /k "cd backend && npm run dev"

REM Wait for backend to start
timeout /t 8 /nobreak >nul

REM 2. Frontend (React) on port 5173
echo %YELLOW%2. Starting Frontend (port 5173)...%RESET%
start "Frontend" cmd /k "cd frontend && npm run dev"

REM Wait for frontend to start
timeout /t 5 /nobreak >nul

REM 3. AI Gateway (Python) on port 5000
echo %YELLOW%3. Starting AI Gateway (port 5000)...%RESET%
start "AI Gateway" cmd /k "cd ai_service_flask && python ai_gateway.py"

REM Wait for AI gateway to start
timeout /t 5 /nobreak >nul

REM 4. AI Pricing Service (Python) on port 5005
echo %YELLOW%4. Starting AI Pricing Service (port 5005)...%RESET%
start "AI Pricing Service" cmd /k "cd ai_service_flask && python ai_pricing_service_wrapper.py"

REM Wait for AI pricing service to start
timeout /t 3 /nobreak >nul

REM 5. Logistics Service (Python) on port 5006
echo %YELLOW%5. Starting Logistics Service (port 5006)...%RESET%
start "Logistics Service" cmd /k "cd ai_service_flask && python logistics_service_wrapper.py"

REM Wait for all services to be ready
echo %YELLOW%6. Waiting for services to be ready...%RESET%
timeout /t 15 /nobreak >nul

REM ========================================
REM HEALTH CHECKS
REM ========================================

echo %YELLOW%🔍 Checking service health...%RESET%

REM Test backend
curl -s http://localhost:3000/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Backend (port 3000): Running%RESET%
) else (
    echo %RED%❌ Backend (port 3000): Not running%RESET%
)

REM Test frontend
curl -s http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Frontend (port 5173): Running%RESET%
) else (
    echo %RED%❌ Frontend (port 5173): Not running%RESET%
)

REM Test AI gateway
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ AI Gateway (port 5000): Running%RESET%
) else (
    echo %RED%❌ AI Gateway (port 5000): Not running%RESET%
)

REM Test AI pricing service
curl -s http://localhost:5005/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ AI Pricing Service (port 5005): Running%RESET%
) else (
    echo %RED%❌ AI Pricing Service (port 5005): Not running%RESET%
)

REM Test logistics service
curl -s http://localhost:5006/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Logistics Service (port 5006): Running%RESET%
) else (
    echo %RED%❌ Logistics Service (port 5006): Not running%RESET%
)

echo.
echo %GREEN%🎉 Core Services Started!%RESET%
echo.
echo %BLUE%📊 Service URLs:%RESET%
echo %GREEN%• Frontend: http://localhost:5173%RESET%
echo %GREEN%• Backend API: http://localhost:3000%RESET%
echo %GREEN%• AI Gateway: http://localhost:5000%RESET%
echo %GREEN%• AI Pricing Service: http://localhost:5005%RESET%
echo %GREEN%• Logistics Service: http://localhost:5006%RESET%
echo.
echo %BLUE%🧪 Ready for testing!%RESET%
echo %YELLOW%Run: python run_system_tests.py --test-type basic%RESET%
echo.
echo %BLUE%📋 Total Services Started: 5 (Core Only)%RESET%
echo %BLUE%📋 Ports Used: 3000, 5173, 5000, 5005, 5006%RESET%
echo.
pause 