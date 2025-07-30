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

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Python is not installed or not in PATH%RESET%
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Node.js is not installed or not in PATH%RESET%
    echo Please install Node.js 18+ and try again
    pause
    exit /b 1
)

:: Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%⚠️ Docker is not installed or not in PATH%RESET%
    echo Docker is recommended for production deployment
    echo Continuing with local development mode...
    echo.
)

:: Check environment variables
echo %BLUE%🔧 Checking environment variables...%RESET%
set "MISSING_VARS="

if not defined SUPABASE_URL set "MISSING_VARS=!MISSING_VARS! SUPABASE_URL"
if not defined SUPABASE_ANON_KEY set "MISSING_VARS=!MISSING_VARS! SUPABASE_ANON_KEY"
if not defined DEEPSEEK_API_KEY set "MISSING_VARS=!MISSING_VARS! DEEPSEEK_API_KEY"
if not defined OPENAI_API_KEY set "MISSING_VARS=!MISSING_VARS! OPENAI_API_KEY"

if not "!MISSING_VARS!"=="" (
    echo %YELLOW%⚠️ Missing environment variables:!MISSING_VARS!%RESET%
    echo Please ensure all required environment variables are set
    echo.
)

:: Install Python dependencies
echo %BLUE%📦 Installing Python dependencies...%RESET%
cd /d "%~dp0"
pip install -r ai_service_flask/requirements.txt
if errorlevel 1 (
    echo %RED%❌ Failed to install Python dependencies%RESET%
    pause
    exit /b 1
)
echo %GREEN%✅ Python dependencies installed%RESET%

:: Install Node.js dependencies
echo %BLUE%📦 Installing Node.js dependencies...%RESET%
cd /d "%~dp0backend"
npm install
if errorlevel 1 (
    echo %RED%❌ Failed to install backend dependencies%RESET%
    pause
    exit /b 1
)
echo %GREEN%✅ Backend dependencies installed%RESET%

cd /d "%~dp0frontend"
npm install
if errorlevel 1 (
    echo %RED%❌ Failed to install frontend dependencies%RESET%
    pause
    exit /b 1
)
echo %GREEN%✅ Frontend dependencies installed%RESET%

:: Return to project root
cd /d "%~dp0"

:: Create necessary directories
echo %BLUE%🔧 Creating necessary directories...%RESET%
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "cache" mkdir cache
if not exist "backups" mkdir backups
echo %GREEN%✅ Directories created%RESET%

:: Start services
echo.
echo %BLUE%🚀 Starting SymbioFlows Production Demo System...%RESET%
echo.

:: Start backend server
echo %BLUE%🔧 Starting Backend API Server...%RESET%
start "SymbioFlows Backend" cmd /k "cd /d %~dp0backend && npm start"
timeout /t 5 /nobreak >nul

:: Start AI services
echo %BLUE%🤖 Starting AI Services Gateway...%RESET%
start "SymbioFlows AI Gateway" cmd /k "cd /d %~dp0ai_service_flask && python ai_gateway.py"
timeout /t 5 /nobreak >nul

:: Start frontend
echo %BLUE%🌐 Starting Frontend Application...%RESET%
start "SymbioFlows Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
timeout /t 5 /nobreak >nul

:: Wait for services to start
echo %BLUE%⏳ Waiting for services to start...%RESET%
timeout /t 15 /nobreak >nul

:: Check service health
echo %BLUE%🔍 Checking service health...%RESET%
echo.

:: Check backend
curl -f http://localhost:3000/api/health >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Backend API Server is not responding%RESET%
) else (
    echo %GREEN%✅ Backend API Server is running%RESET%
)

:: Check AI services
curl -f http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ AI Services Gateway is not responding%RESET%
) else (
    echo %GREEN%✅ AI Services Gateway is running%RESET%
)

:: Check frontend
curl -f http://localhost:5173 >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Frontend Application is not responding%RESET%
) else (
    echo %GREEN%✅ Frontend Application is running%RESET%
)

echo.
echo ========================================
echo   PRODUCTION DEMO SYSTEM STATUS
echo ========================================
echo.
echo %GREEN%🌐 Access URLs:%RESET%
echo   Frontend: http://localhost:5173
echo   Backend API: http://localhost:3000
echo   AI Services: http://localhost:5000
echo   API Documentation: http://localhost:3000/api-docs
echo.
echo %GREEN%📊 Monitoring:%RESET%
echo   Prometheus: http://localhost:9090
echo   Grafana: http://localhost:3001
echo.
echo %YELLOW%💡 Demo Features:%RESET%
echo   • User Signup & Authentication
echo   • AI-Powered Onboarding
echo   • Material Listings Generation
echo   • Intelligent Matchmaking
echo   • Real-time Analytics
echo.
echo %BLUE%🎭 To run the complete demo flow:%RESET%
echo   python start_production_demo.py
echo.
echo %YELLOW%⚠️ Press any key to stop all services...%RESET%
pause >nul

:: Stop all services
echo.
echo %BLUE%🛑 Stopping all services...%RESET%

:: Kill backend process
taskkill /f /im node.exe >nul 2>&1

:: Kill Python processes
taskkill /f /im python.exe >nul 2>&1

echo %GREEN%✅ All services stopped%RESET%
echo.
echo %BLUE%Thank you for using SymbioFlows!%RESET%
pause 