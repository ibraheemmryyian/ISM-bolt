@echo off
echo ========================================
echo Verifying All Required Files Exist
echo ========================================
echo.

set MISSING_FILES=0

REM Check backend files
echo Checking Backend Files...
if not exist "proactive_opportunity_engine.py" (
    echo ❌ Missing: proactive_opportunity_engine.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: proactive_opportunity_engine.py
)

if not exist "gnn_reasoning_engine.py" (
    echo ❌ Missing: gnn_reasoning_engine.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: gnn_reasoning_engine.py
)

if not exist "ai_pricing_orchestrator.py" (
    echo ❌ Missing: ai_pricing_orchestrator.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ai_pricing_orchestrator.py
)

if not exist "materials_bert_service_advanced.py" (
    echo ❌ Missing: materials_bert_service_advanced.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: materials_bert_service_advanced.py
)

if not exist "dynamic_materials_integration_service.py" (
    echo ❌ Missing: dynamic_materials_integration_service.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: dynamic_materials_integration_service.py
)

if not exist "ai_production_orchestrator.py" (
    echo ❌ Missing: ai_production_orchestrator.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ai_production_orchestrator.py
)

if not exist "ai_monitoring_dashboard.py" (
    echo ❌ Missing: ai_monitoring_dashboard.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ai_monitoring_dashboard.py
)

if not exist "system_health_monitor.py" (
    echo ❌ Missing: system_health_monitor.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: system_health_monitor.py
)

if not exist "adaptive_onboarding_server.py" (
    echo ❌ Missing: adaptive_onboarding_server.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: adaptive_onboarding_server.py
)

if not exist "ai_feedback_orchestrator.py" (
    echo ❌ Missing: ai_feedback_orchestrator.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ai_feedback_orchestrator.py
)

echo.
echo Checking AI Service Flask Files...
if not exist "..\ai_service_flask\ai_gateway.py" (
    echo ❌ Missing: ..\ai_service_flask\ai_gateway.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ..\ai_service_flask\ai_gateway.py
)

if not exist "..\ai_service_flask\ai_pricing_service_wrapper.py" (
    echo ❌ Missing: ..\ai_service_flask\ai_pricing_service_wrapper.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ..\ai_service_flask\ai_pricing_service_wrapper.py
)

if not exist "..\ai_service_flask\logistics_service_wrapper.py" (
    echo ❌ Missing: ..\ai_service_flask\logistics_service_wrapper.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ..\ai_service_flask\logistics_service_wrapper.py
)

if not exist "..\ai_service_flask\gnn_inference_service.py" (
    echo ❌ Missing: ..\ai_service_flask\gnn_inference_service.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ..\ai_service_flask\gnn_inference_service.py
)

if not exist "..\ai_service_flask\multi_hop_symbiosis_service.py" (
    echo ❌ Missing: ..\ai_service_flask\multi_hop_symbiosis_service.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ..\ai_service_flask\multi_hop_symbiosis_service.py
)

if not exist "..\ai_service_flask\advanced_analytics_service.py" (
    echo ❌ Missing: ..\ai_service_flask\advanced_analytics_service.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ..\ai_service_flask\advanced_analytics_service.py
)

if not exist "..\ai_service_flask\federated_learning_service.py" (
    echo ❌ Missing: ..\ai_service_flask\federated_learning_service.py
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ..\ai_service_flask\federated_learning_service.py
)

echo.
echo Checking Frontend...
if not exist "..\frontend\package.json" (
    echo ❌ Missing: ..\frontend\package.json
    set /a MISSING_FILES+=1
) else (
    echo ✅ Found: ..\frontend\package.json
)

echo.
echo ========================================
if %MISSING_FILES% EQU 0 (
    echo 🎉 ALL FILES FOUND! Ready to start services.
    echo.
    echo Press any key to continue with startup...
    pause >nul
    exit /b 0
) else (
    echo ❌ MISSING %MISSING_FILES% FILES! Cannot start services.
    echo.
    echo Please ensure all files are present before starting.
    pause
    exit /b 1
) 