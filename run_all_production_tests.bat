@echo off
echo ========================================
echo ISM AI System - Complete Production Test Suite
echo ========================================
echo.

echo Starting comprehensive production testing...
echo This will test all components of the AI system
echo.

echo [1/5] Running Production Test Suite...
python production_test_suite.py
if %errorlevel% neq 0 (
    echo ⚠️ Production tests had issues
) else (
    echo ✅ Production tests completed
)

echo.
echo [2/5] Running AI Services Tests...
python test_ai_services.py
if %errorlevel% neq 0 (
    echo ⚠️ AI Services tests had issues
) else (
    echo ✅ AI Services tests completed
)

echo.
echo [3/5] Running Database Integration Tests...
python test_database_integration.py
if %errorlevel% neq 0 (
    echo ⚠️ Database tests had issues
) else (
    echo ✅ Database tests completed
)

echo.
echo [4/5] Running Frontend Integration Tests...
python test_frontend_integration.py
if %errorlevel% neq 0 (
    echo ⚠️ Frontend tests had issues
) else (
    echo ✅ Frontend tests completed
)

echo.
echo [5/5] Generating Combined Test Report...
python generate_combined_report.py
if %errorlevel% neq 0 (
    echo ⚠️ Report generation had issues
) else (
    echo ✅ Combined report generated
)

echo.
echo ========================================
echo PRODUCTION TESTING COMPLETED
echo ========================================
echo.
echo Test Reports Generated:
echo - production_test_report.json
echo - ai_services_test_report.json
echo - database_integration_test_report.json
echo - frontend_integration_test_report.json
echo - combined_test_report.json
echo.
echo Check the reports for detailed results and recommendations.
echo.
pause 