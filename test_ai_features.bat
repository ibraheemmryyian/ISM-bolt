@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🧠 ISM AI - Testing AI Features with Real Data
echo ===============================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Testing AI features with your 50 real companies...%RESET%
echo.

REM Check if backend is running
echo %YELLOW%Checking backend status...%RESET%
curl -s http://localhost:5001/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Backend is running%RESET%
) else (
    echo %RED%❌ Backend is not running%RESET%
    echo %YELLOW%Please start the backend first%RESET%
    pause
    exit /b 1
)

echo.
echo %BLUE%Running AI features test...%RESET%
python test_ai_with_real_data.py

echo.
echo %GREEN%Test completed! Check results above.%RESET%
pause 