@echo off
chcp 65001 >nul
echo ========================================
echo 🔍 SymbioFlows - SYSTEM VALIDATION
echo ========================================
echo.
echo Validating all microservices for full usability...
echo.

REM Wait for services to start
echo ⏳ Waiting for services to initialize...
timeout /t 30 /nobreak >nul

REM Run comprehensive validation
echo 🚀 Running comprehensive service validation...
python validate_all_services.py

echo.
echo ========================================
echo ✅ Validation Complete!
echo ========================================
echo.
echo Check the validation report for detailed results.
echo.
pause 