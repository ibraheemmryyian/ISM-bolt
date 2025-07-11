@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 ISM AI - Complete Real Data Testing
echo ======================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Starting comprehensive real data testing...%RESET%
echo.

REM Step 1: Check if backend is running
echo %YELLOW%Step 1: Checking backend status...%RESET%
curl -s http://localhost:5001/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Backend is running on port 5001%RESET%
) else (
    echo %RED%❌ Backend is not running on port 5001%RESET%
    echo %YELLOW%Please start the backend: cd backend ^& npm start%RESET%
    pause
    exit /b 1
)

REM Step 2: Check data files
echo.
echo %YELLOW%Step 2: Checking data files...%RESET%
if exist "data\50_real_gulf_companies_cleaned.json" (
    echo %GREEN%✅ Found: data\50_real_gulf_companies_cleaned.json%RESET%
) else (
    echo %RED%❌ Missing: data\50_real_gulf_companies_cleaned.json%RESET%
)

if exist "data\50_gulf_companies_fixed.json" (
    echo %GREEN%✅ Found: data\50_gulf_companies_fixed.json%RESET%
) else (
    echo %RED%❌ Missing: data\50_gulf_companies_fixed.json%RESET%
)

REM Step 3: Test API endpoints
echo.
echo %YELLOW%Step 3: Testing API endpoints...%RESET%
python test_api_endpoints.py

REM Step 4: Run simple import test
echo.
echo %YELLOW%Step 4: Running simple import test...%RESET%
python test_real_data_simple.py

REM Step 5: Check database connection
echo.
echo %YELLOW%Step 5: Checking database connection...%RESET%
curl -s http://localhost:5001/api/companies >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Database connection working%RESET%
) else (
    echo %RED%❌ Database connection failed%RESET%
)

echo.
echo %BLUE%=====================================%RESET%
echo %BLUE%Testing completed! Check results above.%RESET%
echo %BLUE%=====================================%RESET%
pause 