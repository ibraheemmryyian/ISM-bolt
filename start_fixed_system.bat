@echo off
echo ========================================
echo STARTING FIXED AI SYSTEM
echo ========================================

echo.
echo This script will start the complete AI system with the fixed MaterialsBERT service.
echo.

REM Check if we're in the right directory
if not exist "backend" (
    echo Error: Backend directory not found. Please run this script from the project root.
    pause
    exit /b 1
)

REM Check if MaterialsBERT dependencies are installed
echo Checking MaterialsBERT dependencies...
python -c "import flask, numpy, requests, sklearn" 2>nul
if errorlevel 1 (
    echo.
    echo MaterialsBERT dependencies not found. Installing now...
    call install_materials_bert.bat
    if errorlevel 1 (
        echo.
        echo Failed to install MaterialsBERT dependencies.
        echo Please run install_materials_bert.bat manually and try again.
        pause
        exit /b 1
    )
)

echo.
echo Starting backend services...

REM Start the main backend
echo Starting main backend...
start "Backend" cmd /k "cd backend && python app.js"

REM Wait a moment
timeout /t 3 /nobreak >nul

REM Start the simplified MaterialsBERT service
echo Starting MaterialsBERT service...
start "MaterialsBERT" cmd /k "python start_materials_bert_simple.py"

REM Wait for services to start
echo.
echo Waiting for services to start...
timeout /t 5 /nobreak >nul

REM Start the frontend
echo Starting frontend...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo SYSTEM STARTUP COMPLETE
echo ========================================
echo.
echo Services started:
echo - Main Backend: http://localhost:3000
echo - MaterialsBERT: http://localhost:5002
echo - Frontend: http://localhost:5173
echo.
echo To test the MaterialsBERT service, run:
echo   python test_materials_bert_simple.py
echo.
echo Press any key to continue...
pause 