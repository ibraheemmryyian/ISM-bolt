@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 ISM AI Platform - Complete Startup
echo =====================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Starting all services and importing real data...%RESET%
echo.

REM Start Backend (Node.js)
echo %YELLOW%1. Starting Node.js Backend...%RESET%
start "Backend" cmd /k "cd backend && npm run dev"

REM Wait a bit for backend to start
timeout /t 5 /nobreak >nul

REM Start AI Gateway (Python)
echo %YELLOW%2. Starting Python AI Gateway...%RESET%
start "AI Gateway" cmd /k "cd ai_service_flask && python ai_gateway.py"

REM Wait a bit for AI gateway to start
timeout /t 5 /nobreak >nul

REM Start Frontend (React)
echo %YELLOW%3. Starting React Frontend...%RESET%
start "Frontend" cmd /k "cd frontend && npm run dev"

REM Wait for all services to be ready
echo %YELLOW%4. Waiting for services to be ready...%RESET%
timeout /t 10 /nobreak >nul

REM Check if services are running
echo %YELLOW%5. Checking service status...%RESET%

REM Test backend
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Backend: Running%RESET%
) else (
    echo %RED%❌ Backend: Not running%RESET%
)

REM Test AI gateway
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ AI Gateway: Running%RESET%
) else (
    echo %RED%❌ AI Gateway: Not running%RESET%
)

REM Test frontend
curl -s http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Frontend: Running%RESET%
) else (
    echo %RED%❌ Frontend: Not running%RESET%
)

echo.
echo %BLUE%6. Running real data import...%RESET%

REM Run the import script
call scripts\import-real-data.bat

echo.
echo %GREEN%🎉 Everything is ready!%RESET%
echo.
echo %BLUE%📊 Access your platform:%RESET%
echo %GREEN%• Frontend: http://localhost:5173%RESET%
echo %GREEN%• Admin Dashboard: http://localhost:5173/admin%RESET%
echo %GREEN%• Backend API: http://localhost:5000%RESET%
echo %GREEN%• AI Gateway: http://localhost:5000%RESET%
echo.
echo %BLUE%📋 What you can do now:%RESET%
echo %YELLOW%• View 50 real Gulf companies%RESET%
echo %YELLOW%• See AI-generated listings and matches%RESET%
echo %YELLOW%• Explore the admin dashboard%RESET%
echo %YELLOW%• Test the AI features%RESET%
echo.
pause 