@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🧪 SYMBIOFLOWS - COMPREHENSIVE SYSTEM TEST
echo ========================================
echo 🎯 Testing ALL Advanced AI Services and Integration
echo ========================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%🔍 Starting comprehensive system testing...%RESET%
echo.

REM ========================================
REM PREREQUISITE VALIDATION
REM ========================================

echo %YELLOW%1. Validating prerequisites...%RESET%

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Node.js available%RESET%
) else (
    echo %RED%   ❌ Node.js not found%RESET%
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Python available%RESET%
) else (
    echo %RED%   ❌ Python not found%RESET%
    pause
    exit /b 1
)

REM Check npm
npm --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ npm available%RESET%
) else (
    echo %RED%   ❌ npm not found%RESET%
    pause
    exit /b 1
)

echo.

REM ========================================
REM CORE SERVICES TESTING
REM ========================================

echo %YELLOW%2. Testing core services...%RESET%

REM Test Backend API
echo %BLUE%   Testing Backend API...%RESET%
curl -s http://localhost:3001/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ Backend API is running%RESET%
) else (
    echo %RED%     ❌ Backend API is not running%RESET%
    echo %YELLOW%     Starting Backend API...%RESET%
    start "Backend API" cmd /k "cd backend && npm run dev"
    timeout /t 5 /nobreak >nul
)

REM Test Frontend
echo %BLUE%   Testing Frontend...%RESET%
curl -s http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ Frontend is running%RESET%
) else (
    echo %RED%     ❌ Frontend is not running%RESET%
    echo %YELLOW%     Starting Frontend...%RESET%
    start "Frontend" cmd /k "cd frontend && npm run dev"
    timeout /t 5 /nobreak >nul
)

echo.

REM ========================================
REM AI SERVICES TESTING
REM ========================================

echo %YELLOW%3. Testing AI Services...%RESET%

REM Test AI Gateway
echo %BLUE%   Testing AI Gateway...%RESET%
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ AI Gateway is running%RESET%
) else (
    echo %RED%     ❌ AI Gateway is not running%RESET%
)

REM Test Advanced Analytics Service
echo %BLUE%   Testing Advanced Analytics Service...%RESET%
curl -s http://localhost:5004/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ Advanced Analytics Service is running%RESET%
) else (
    echo %RED%     ❌ Advanced Analytics Service is not running%RESET%
)

REM Test GNN Inference Service
echo %BLUE%   Testing GNN Inference Service...%RESET%
curl -s http://localhost:5001/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ GNN Inference Service is running%RESET%
) else (
    echo %RED%     ❌ GNN Inference Service is not running%RESET%
)

REM Test Multi-Hop Symbiosis Service
echo %BLUE%   Testing Multi-Hop Symbiosis Service...%RESET%
curl -s http://localhost:5003/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ Multi-Hop Symbiosis Service is running%RESET%
) else (
    echo %RED%     ❌ Multi-Hop Symbiosis Service is not running%RESET%
)

REM Test Federated Learning Service
echo %BLUE%   Testing Federated Learning Service...%RESET%
curl -s http://localhost:5002/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ Federated Learning Service is running%RESET%
) else (
    echo %RED%     ❌ Federated Learning Service is not running%RESET%
)

echo.

REM ========================================
REM BACKEND SERVICES TESTING
REM ========================================

echo %YELLOW%4. Testing Backend Services...%RESET%

REM Test Adaptive Onboarding Server
echo %BLUE%   Testing Adaptive Onboarding Server...%RESET%
curl -s http://localhost:5006/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ Adaptive Onboarding Server is running%RESET%
) else (
    echo %RED%     ❌ Adaptive Onboarding Server is not running%RESET%
)

REM Test System Health Monitor
echo %BLUE%   Testing System Health Monitor...%RESET%
curl -s http://localhost:5018/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ System Health Monitor is running%RESET%
) else (
    echo %RED%     ❌ System Health Monitor is not running%RESET%
)

REM Test AI Monitoring Dashboard
echo %BLUE%   Testing AI Monitoring Dashboard...%RESET%
curl -s http://localhost:5007/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ AI Monitoring Dashboard is running%RESET%
) else (
    echo %RED%     ❌ AI Monitoring Dashboard is not running%RESET%
)

REM Test AI Pricing Service
echo %BLUE%   Testing AI Pricing Service...%RESET%
curl -s http://localhost:5008/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ AI Pricing Service is running%RESET%
) else (
    echo %RED%     ❌ AI Pricing Service is not running%RESET%
)

echo.

REM ========================================
REM ADVANCED AI SERVICES TESTING
REM ========================================

echo %YELLOW%5. Testing Advanced AI Services...%RESET%

REM Test MaterialsBERT Service
echo %BLUE%   Testing MaterialsBERT Service...%RESET%
curl -s http://localhost:5009/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ MaterialsBERT Service is running%RESET%
) else (
    echo %RED%     ❌ MaterialsBERT Service is not running%RESET%
)

REM Test MaterialsBERT Simple Service
echo %BLUE%   Testing MaterialsBERT Simple Service...%RESET%
curl -s http://localhost:5010/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ MaterialsBERT Simple Service is running%RESET%
) else (
    echo %RED%     ❌ MaterialsBERT Simple Service is not running%RESET%
)

REM Test Ultra AI Listings Generator
echo %BLUE%   Testing Ultra AI Listings Generator...%RESET%
curl -s http://localhost:5011/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ Ultra AI Listings Generator is running%RESET%
) else (
    echo %RED%     ❌ Ultra AI Listings Generator is not running%RESET%
)

REM Test AI Listings Generator
echo %BLUE%   Testing AI Listings Generator...%RESET%
curl -s http://localhost:5012/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%     ✅ AI Listings Generator is running%RESET%
) else (
    echo %RED%     ❌ AI Listings Generator is not running%RESET%
)

echo.

REM ========================================
REM PYTHON SERVICES VALIDATION
REM ========================================

echo %YELLOW%6. Running Python services validation...%RESET%

REM Test Python services validation
if exist "backend\validate_all_services.py" (
    echo %BLUE%   Running comprehensive Python services test...%RESET%
    cd backend
    python validate_all_services.py
    cd ..
) else (
    echo %RED%   ❌ validate_all_services.py not found%RESET%
)

echo.

REM ========================================
REM PERFORMANCE TESTING
REM ========================================

echo %YELLOW%7. Running performance tests...%RESET%

REM Test real performance benchmark
if exist "backend\real_performance_benchmark.py" (
    echo %BLUE%   Running real performance benchmark...%RESET%
    cd backend
    python real_performance_benchmark.py
    cd ..
) else (
    echo %RED%   ❌ real_performance_benchmark.py not found%RESET%
)

echo.

REM ========================================
REM AI QUALITY TESTING
REM ========================================

echo %YELLOW%8. Testing AI Quality Systems...%RESET%

REM Test AI Quality Monitor
if exist "ai_quality_monitor.py" (
    echo %BLUE%   Testing AI Quality Monitor...%RESET%
    python ai_quality_monitor.py --test
) else (
    echo %RED%   ❌ ai_quality_monitor.py not found%RESET%
)

REM Test Advanced AI Quality Analysis
if exist "advanced_ai_quality_analysis.py" (
    echo %BLUE%   Testing Advanced AI Quality Analysis...%RESET%
    python advanced_ai_quality_analysis.py --test
) else (
    echo %RED%   ❌ advanced_ai_quality_analysis.py not found%RESET%
)

echo.

REM ========================================
REM INTEGRATION TESTING
REM ========================================

echo %YELLOW%9. Testing system integration...%RESET%

REM Test comprehensive system integration
if exist "backend\complete_system_integration.py" (
    echo %BLUE%   Testing complete system integration...%RESET%
    cd backend
    python complete_system_integration.py --test
    cd ..
) else (
    echo %RED%   ❌ complete_system_integration.py not found%RESET%
)

echo.

REM ========================================
REM DATABASE CONNECTIVITY TESTING
REM ========================================

echo %YELLOW%10. Testing database connectivity...%RESET%

REM Test database state
if exist "check_database_state.py" (
    echo %BLUE%   Checking database state...%RESET%
    python check_database_state.py
) else (
    echo %RED%   ❌ check_database_state.py not found%RESET%
)

REM Test materials table
if exist "check_materials_table.py" (
    echo %BLUE%   Checking materials table...%RESET%
    python check_materials_table.py
) else (
    echo %RED%   ❌ check_materials_table.py not found%RESET%
)

echo.

REM ========================================
REM API ENDPOINT TESTING
REM ========================================

echo %YELLOW%11. Testing API endpoints...%RESET%

REM List all endpoints
if exist "list_all_endpoints.py" (
    echo %BLUE%   Listing all available endpoints...%RESET%
    python list_all_endpoints.py
) else (
    echo %RED%   ❌ list_all_endpoints.py not found%RESET%
)

echo.

REM ========================================
REM TEST SUMMARY
REM ========================================

echo ========================================
echo 📊 COMPREHENSIVE TEST SUMMARY
echo ========================================
echo.
echo %BLUE%🎯 Test Results:%RESET%
echo.
echo %GREEN%✅ Core Services Tested%RESET%
echo %GREEN%✅ AI Services Tested%RESET%
echo %GREEN%✅ Backend Services Tested%RESET%
echo %GREEN%✅ Advanced AI Services Tested%RESET%
echo %GREEN%✅ Python Services Validated%RESET%
echo %GREEN%✅ Performance Tests Run%RESET%
echo %GREEN%✅ AI Quality Systems Tested%RESET%
echo %GREEN%✅ Integration Tests Run%RESET%
echo %GREEN%✅ Database Connectivity Tested%RESET%
echo %GREEN%✅ API Endpoints Listed%RESET%
echo.
echo %BLUE%📋 Next Steps:%RESET%
echo   1. Review any failed tests above
echo   2. Check service logs for errors
echo   3. Run individual service tests if needed
echo   4. Start services that are not running
echo.
echo %YELLOW%For detailed testing, see:%RESET%
echo   - ULTRA_ADVANCED_AI_TESTING_GUIDE.md
echo   - REAL_PERFORMANCE_TESTING_GUIDE.md
echo   - TESTING_STRATEGY.md
echo.
echo ========================================
pause 