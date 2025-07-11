@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 ISM AI Real Data Import Process
echo ===================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Starting real data import process...%RESET%
echo.

REM Check if backend is running
echo %YELLOW%Checking if backend is running...%RESET%
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Backend is not running! Please start the backend first.%RESET%
    echo %YELLOW%Run: cd backend ^& npm run dev%RESET%
    pause
    exit /b 1
)
echo %GREEN%✅ Backend is running%RESET%

REM Check if AI gateway is running
echo %YELLOW%Checking if AI gateway is running...%RESET%
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ AI Gateway is not running! Please start the AI gateway first.%RESET%
    echo %YELLOW%Run: cd ai_service_flask ^& python ai_gateway.py%RESET%
    pause
    exit /b 1
)
echo %GREEN%✅ AI Gateway is running%RESET%

REM Check if company data exists
echo %YELLOW%Checking for company data...%RESET%
if not exist "data\50_real_gulf_companies_cleaned.json" (
    echo %RED%❌ Company data file not found!%RESET%
    echo %YELLOW%Expected: data\50_real_gulf_companies_cleaned.json%RESET%
    pause
    exit /b 1
)
echo %GREEN%✅ Company data file found%RESET%

REM Run the bulk importer
echo.
echo %BLUE%🔧 Running bulk importer...%RESET%
cd backend
python real_data_bulk_importer.py

if %errorlevel% equ 0 (
    echo.
    echo %GREEN%🎉 SUCCESS! Real data import completed!%RESET%
    echo.
    echo %BLUE%📊 What was imported:%RESET%
    echo %GREEN%✅ 50 Gulf companies with detailed profiles%RESET%
    echo %GREEN%✅ AI-generated waste and requirement listings%RESET%
    echo %GREEN%✅ Symbiosis matches between companies%RESET%
    echo.
    echo %BLUE%🌐 Access your admin dashboard:%RESET%
    echo %GREEN%http://localhost:5173/admin%RESET%
    echo.
    echo %BLUE%📋 You can now:%RESET%
    echo %YELLOW%• View all companies in the Companies tab%RESET%
    echo %YELLOW%• See all AI-generated listings in the Materials tab%RESET%
    echo %YELLOW%• Explore matches in the Matches tab%RESET%
    echo %YELLOW%• Check AI insights in the AI Insights tab%RESET%
) else (
    echo.
    echo %RED%❌ Import failed! Check the error messages above.%RESET%
)

cd ..
echo.
echo %GREEN%✅ Import script completed!%RESET%
pause 