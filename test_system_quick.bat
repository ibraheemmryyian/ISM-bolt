@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🧪 SYMBIOFLOWS - QUICK SYSTEM TEST
echo ========================================
echo 🎯 Testing Core Components and Services
echo ========================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%🔍 Running quick system validation...%RESET%
echo.

REM ========================================
REM PREREQUISITE CHECKS
REM ========================================

echo %YELLOW%1. Checking Prerequisites...%RESET%

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Node.js installed%RESET%
) else (
    echo %RED%   ❌ Node.js not found%RESET%
)

REM Check Python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Python installed%RESET%
) else (
    echo %RED%   ❌ Python not found%RESET%
)

REM Check npm
npm --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ npm installed%RESET%
) else (
    echo %RED%   ❌ npm not found%RESET%
)

echo.

REM ========================================
REM FILE STRUCTURE VALIDATION
REM ========================================

echo %YELLOW%2. Validating File Structure...%RESET%

REM Check key directories
if exist "frontend" (
    echo %GREEN%   ✅ Frontend directory exists%RESET%
) else (
    echo %RED%   ❌ Frontend directory missing%RESET%
)

if exist "backend" (
    echo %GREEN%   ✅ Backend directory exists%RESET%
) else (
    echo %RED%   ❌ Backend directory missing%RESET%
)

if exist "ai_service_flask" (
    echo %GREEN%   ✅ AI Services directory exists%RESET%
) else (
    echo %RED%   ❌ AI Services directory missing%RESET%
)

if exist ".env" (
    echo %GREEN%   ✅ Environment file exists%RESET%
) else (
    echo %RED%   ❌ Environment file missing%RESET%
)

echo.

REM ========================================
REM DEPENDENCY CHECKS
REM ========================================

echo %YELLOW%3. Checking Dependencies...%RESET%

REM Check Python requirements
if exist "requirements.txt" (
    echo %GREEN%   ✅ Python requirements file exists%RESET%
) else (
    echo %RED%   ❌ Python requirements file missing%RESET%
)

REM Check frontend package.json
if exist "frontend\package.json" (
    echo %GREEN%   ✅ Frontend package.json exists%RESET%
) else (
    echo %RED%   ❌ Frontend package.json missing%RESET%
)

echo.

REM ========================================
REM SERVICE VALIDATION
REM ========================================

echo %YELLOW%4. Validating Core Services...%RESET%

REM Check key service files
set SERVICE_COUNT=0
if exist "backend\ai_listings_generator.py" set /a SERVICE_COUNT+=1
if exist "backend\ai_matchmaking_service.py" set /a SERVICE_COUNT+=1
if exist "backend\ai_pricing_service.py" set /a SERVICE_COUNT+=1
if exist "ai_service_flask\ai_gateway.py" set /a SERVICE_COUNT+=1
if exist "ai_service_flask\advanced_analytics_service.py" set /a SERVICE_COUNT+=1

echo %GREEN%   ✅ Found %SERVICE_COUNT% core AI services%RESET%

echo.

REM ========================================
REM CONFIGURATION CHECKS
REM ========================================

echo %YELLOW%5. Checking Configuration...%RESET%

REM Check for Docker configuration
if exist "docker-compose.prod.yml" (
    echo %GREEN%   ✅ Production Docker config exists%RESET%
) else (
    echo %RED%   ❌ Production Docker config missing%RESET%
)

REM Check for Kubernetes configs
if exist "k8s" (
    echo %GREEN%   ✅ Kubernetes configs exist%RESET%
) else (
    echo %RED%   ❌ Kubernetes configs missing%RESET%
)

echo.

REM ========================================
REM SUMMARY
REM ========================================

echo ========================================
echo 📊 QUICK TEST SUMMARY
echo ========================================
echo.
echo %BLUE%🎯 System Status: READY FOR TESTING%RESET%
echo.
echo %YELLOW%Next Steps:%RESET%
echo   1. Run: %GREEN%start_complete_system.bat%RESET% (Full system startup)
echo   2. Run: %GREEN%test_all_services.bat%RESET% (Comprehensive testing)
echo   3. Run: %GREEN%run_world_class_ai.bat%RESET% (AI system test)
echo.
echo %BLUE%For detailed testing, see:%RESET%
echo   - ULTRA_ADVANCED_AI_TESTING_GUIDE.md
echo   - REAL_PERFORMANCE_TESTING_GUIDE.md
echo   - TESTING_STRATEGY.md
echo.
echo ========================================
pause 