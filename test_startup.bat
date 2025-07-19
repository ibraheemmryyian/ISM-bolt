@echo off
chcp 65001 >nul
echo ========================================
echo 🧪 SymbioFlows Startup Test
echo ========================================
echo.
echo Testing startup with correct paths...
echo.

REM Test backend npm script
echo Testing backend npm script...
cd backend
npm run dev --help >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend npm script works
) else (
    echo ❌ Backend npm script failed
)
cd ..

REM Test frontend npm script
echo Testing frontend npm script...
cd frontend
npm run dev --help >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend npm script works
) else (
    echo ❌ Frontend npm script failed
)
cd ..

REM Test Python files exist
echo Testing Python files exist...
if exist "backend\adaptive_onboarding_server.py" (
    echo ✅ adaptive_onboarding_server.py exists
) else (
    echo ❌ adaptive_onboarding_server.py missing
)

if exist "backend\system_health_monitor.py" (
    echo ✅ system_health_monitor.py exists
) else (
    echo ❌ system_health_monitor.py missing
)

if exist "backend\start_production_ai_system.py" (
    echo ✅ start_production_ai_system.py exists
) else (
    echo ❌ start_production_ai_system.py missing
)

if exist "backend\ai_monitoring_dashboard.py" (
    echo ✅ ai_monitoring_dashboard.py exists
) else (
    echo ❌ ai_monitoring_dashboard.py missing
)

if exist "ai_service_flask\ai_gateway.py" (
    echo ✅ ai_gateway.py exists
) else (
    echo ❌ ai_gateway.py missing
)

if exist "ai_service_flask\ai_pricing_service_wrapper.py" (
    echo ✅ ai_pricing_service_wrapper.py exists
) else (
    echo ❌ ai_pricing_service_wrapper.py missing
)

echo.
echo ========================================
echo 🎯 Test Complete!
echo ========================================
echo.
echo If all tests passed, run: start_final.bat
echo.
pause 