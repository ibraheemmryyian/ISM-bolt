@echo off
echo ========================================
echo 🚀 SymbioFlows One-Click Startup
echo ========================================
echo.
echo Starting the complete SymbioFlows AI system...
echo.

REM Navigate to backend directory and run the robust startup
cd backend
call start_robust_system.bat

REM Return to root directory
cd ..

echo.
echo ========================================
echo 🎉 SymbioFlows startup complete!
echo ========================================
pause 