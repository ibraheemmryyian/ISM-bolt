@echo off
echo Testing Python AI Endpoints...
echo.

REM Check if backend is running
echo Checking if backend is running on port 5001...
curl -s http://localhost:5001/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Backend is not running on port 5001
    echo Please start the backend first with: start_backend.bat
    pause
    exit /b 1
)

echo Backend is running. Testing Python AI endpoints...
echo.

REM Run the Python test script
python test_python_ai_endpoints.py

echo.
echo Test complete!
pause 