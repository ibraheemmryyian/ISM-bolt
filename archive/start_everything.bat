@echo off
chcp 65001 >nul
echo ========================================
echo 🚀 SymbioFlows - Original Working Startup
echo ========================================
echo.
echo Starting the original working system...
echo.

REM Navigate to backend directory and run the original script
cd backend
call start_everything.bat

REM Return to root directory
cd ..
echo.
echo ✅ System startup complete!
echo.
pause 