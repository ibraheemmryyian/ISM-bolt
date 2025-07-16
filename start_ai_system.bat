@echo off
echo ========================================
echo ISM AI System Startup
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
python start_ai_system_simple.py --check-deps

echo.
echo Starting AI System...
echo Press Ctrl+C to stop the system
echo.

python start_ai_system_simple.py

echo.
echo AI System stopped.
pause 