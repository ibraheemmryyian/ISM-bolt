@echo off
chcp 65001 >nul
echo ========================================
echo 🚀 SymbioFlows FINAL STARTUP
echo ========================================
echo.
echo 🎯 Using ROBUST SYSTEM with error handling
echo 💡 Only starts services that work reliably
echo 🔧 Handles import errors gracefully
echo.
echo Press any key to start the robust system...
pause >nul

REM Run the robust startup directly
call backend\start_robust_system.bat

echo.
echo ========================================
echo 🎉 SymbioFlows FINAL startup complete!
echo ========================================
echo.
echo 💡 You should now have 8 separate CMD windows:
echo    1. Backend API (Node.js)
echo    2. Frontend (React)
echo    3. Adaptive Onboarding
echo    4. System Health Monitor
echo    5. AI Production Orchestrator
echo    6. AI Monitoring Dashboard
echo    7. AI Gateway (may show errors)
echo    8. AI Pricing Service (may show errors)
echo.
pause 