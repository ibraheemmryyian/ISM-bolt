@echo off
echo Starting Advanced MaterialsBERT Service...
echo This will launch the sophisticated AI-powered materials analysis system

echo.
echo Checking Python environment...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Checking required packages...
python -c "import numpy, scipy, pandas, sklearn, flask, torch" 2>nul
if errorlevel 1 (
    echo ERROR: Required packages are not installed
    echo Please run install_advanced_materials_bert.bat first
    pause
    exit /b 1
)

echo.
echo Checking service file...
if not exist "backend\materials_bert_service_advanced.py" (
    echo ERROR: Advanced MaterialsBERT service file not found
    echo Please ensure the service file exists in the backend directory
    pause
    exit /b 1
)

echo.
echo Starting Advanced MaterialsBERT Service...
echo Service will be available at: http://localhost:5002
echo.
echo Press Ctrl+C to stop the service
echo.

cd backend
python materials_bert_service_advanced.py

echo.
echo Service stopped.
pause 