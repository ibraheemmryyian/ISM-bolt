@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo   SYMBIOFLOWS PRODUCTION DEMO SYSTEM
echo ========================================
echo.

:: Set colors for output
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

echo %BLUE%🚀 Starting SymbioFlows Production Demo System%RESET%
echo %BLUE%⏰ Started at: %date% %time%%RESET%
echo.

:: Check if Python is installed
echo %BLUE%🔍 Checking Python installation...%RESET%
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Python is not installed or not in PATH%RESET%
    echo Please install Python 3.8+ and try again
    echo.
    echo %YELLOW%💡 Try running: python --version%RESET%
    pause
    exit /b 1
)

:: Show Python version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%✅ %PYTHON_VERSION%%RESET%

:: Check if the script exists
if not exist "start_production_demo.py" (
    echo %RED%❌ start_production_demo.py not found%RESET%
    echo Please ensure you're running this from the project root directory
    pause
    exit /b 1
)

:: Check if Node.js is installed
echo %BLUE%🔍 Checking Node.js installation...%RESET%
node --version >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%⚠️ Node.js is not installed or not in PATH%RESET%
    echo Some features may not work correctly
) else (
    for /f "tokens=*" %%i in ('node --version 2^>^&1') do set NODE_VERSION=%%i
    echo %GREEN%✅ Node.js %NODE_VERSION%%RESET%
)

:: Check if npm is installed
echo %BLUE%🔍 Checking npm installation...%RESET%
npm --version >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%⚠️ npm is not installed or not in PATH%RESET%
    echo Some features may not work correctly
) else (
    for /f "tokens=*" %%i in ('npm --version 2^>^&1') do set NPM_VERSION=%%i
    echo %GREEN%✅ npm %NPM_VERSION%%RESET%
)

echo.
echo %BLUE%🔧 Running test script first...%RESET%
echo %YELLOW%This will verify that Python execution is working correctly%RESET%
echo.

:: Run test script first (Windows compatible)
python test_script_windows.py
if errorlevel 1 (
    echo.
    echo %RED%❌ Test script failed - Python execution may have issues%RESET%
    echo %YELLOW%Please check the error messages above%RESET%
    pause
    exit /b 1
)

echo.
echo %GREEN%✅ Test script completed successfully%RESET%
echo.

:: Check for environment file
if not exist ".env" (
    echo %YELLOW%⚠️ .env file not found%RESET%
    echo %BLUE%💡 You can copy .env.example to .env and fill in your API keys%RESET%
    echo %YELLOW%Some features may not work without proper environment variables%RESET%
    echo.
    echo %BLUE%Press any key to continue anyway, or Ctrl+C to exit...%RESET%
    pause >nul
)

echo.
echo %BLUE%🚀 Starting production demo system...%RESET%
echo %YELLOW%📝 Detailed logs will be written to: production_demo.log%RESET%
echo %YELLOW%⏳ This may take several minutes to complete...%RESET%
echo.

:: Run the main production demo script
python start_production_demo.py

:: Check the exit code
if errorlevel 1 (
    echo.
    echo %RED%❌ Production demo failed with exit code %errorlevel%%RESET%
    echo %YELLOW%📝 Check production_demo.log for detailed error information%RESET%
    echo.
    echo %BLUE%💡 Common issues:%RESET%
    echo %YELLOW%  - Missing environment variables (.env file)%RESET%
    echo %YELLOW%  - Missing dependencies (npm install, pip install)%RESET%
    echo %YELLOW%  - Port conflicts (services already running)%RESET%
    echo %YELLOW%  - Network connectivity issues%RESET%
) else (
    echo.
    echo %GREEN%✅ Production demo completed successfully%RESET%
    echo.
    echo %BLUE%🌐 Access URLs:%RESET%
    echo %GREEN%  Frontend: http://localhost:5173%RESET%
    echo %GREEN%  Backend API: http://localhost:3000%RESET%
    echo %GREEN%  AI Services: http://localhost:5000%RESET%
    echo %GREEN%  API Documentation: http://localhost:3000/api-docs%RESET%
)

echo.
echo %BLUE%📝 Log files created:%RESET%
if exist "production_demo.log" (
    echo %GREEN%  ✅ production_demo.log%RESET%
) else (
    echo %RED%  ❌ production_demo.log (not found)%RESET%
)

if exist "test_execution.log" (
    echo %GREEN%  ✅ test_execution.log%RESET%
) else (
    echo %RED%  ❌ test_execution.log (not found)%RESET%
)

echo.
echo %BLUE%Thank you for using SymbioFlows!%RESET%
pause 