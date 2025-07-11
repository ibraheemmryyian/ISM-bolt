@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🧪 ISM AI - Comprehensive Python Services Test
echo ===============================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Testing all Python services and backend integration...%RESET%
echo.

REM Check if backend is running
echo %YELLOW%Checking backend status...%RESET%
curl -s http://localhost:5001/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Backend is running%RESET%
) else (
    echo %RED%❌ Backend is not running%RESET%
    echo %YELLOW%Please start the backend first: cd backend ^& npm start%RESET%
    echo.
    echo %BLUE%Starting backend...%RESET%
    start "Backend Server" cmd /k "cd backend && npm start"
    echo %YELLOW%Waiting for backend to start...%RESET%
    timeout /t 5 /nobreak >nul
)

echo.
echo %BLUE%Running comprehensive Python services test...%RESET%
python test_python_services.py

echo.
echo %GREEN%Test completed! Check results above.%RESET%
pause 