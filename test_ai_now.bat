@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🧠 ISM AI - Quick AI Testing
echo ============================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set CYAN=[96m
set RESET=[0m

echo %BLUE%Testing AI systems before introducing 50 companies...%RESET%
echo.

REM Check if backend is running
echo %YELLOW%Checking backend status...%RESET%
curl -s http://localhost:3001/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Backend is running%RESET%
) else (
    echo %RED%❌ Backend is not running%RESET%
    echo %YELLOW%Please start the backend first:%RESET%
    echo %CYAN%  cd backend && npm start%RESET%
    pause
    exit /b 1
)

echo.
echo %BLUE%Running comprehensive AI tests...%RESET%
echo.

REM Run the comprehensive AI testing script
powershell -ExecutionPolicy Bypass -File "scripts\test-ai-comprehensive.ps1"

echo.
echo %BLUE%Quick Manual Tests:%RESET%
echo %CYAN%1. Test AI Portfolio Generation:%RESET%
curl -X POST http://localhost:3001/api/ai/generate-all-listings -H "Content-Type: application/json" -d "{\"test\": true}" 2>nul
echo.
echo.

echo %CYAN%2. Test Material Pricing:%RESET%
curl -X POST http://localhost:3001/api/materials/pricing -H "Content-Type: application/json" -d "{\"material_name\": \"steel scrap\", \"quantity\": 1000, \"unit\": \"kg\"}" 2>nul
echo.
echo.

echo %CYAN%3. Test Logistics Preview:%RESET%
curl -X POST http://localhost:3001/api/logistics-preview -H "Content-Type: application/json" -d "{\"origin\": \"Dubai\", \"destination\": \"Abu Dhabi\", \"material\": \"steel scrap\", \"weight_kg\": 1000}" 2>nul
echo.
echo.

echo %GREEN%✅ AI testing completed!%RESET%
echo.
echo %BLUE%Next Steps:%RESET%
echo %YELLOW%1. Check the test results above%RESET%
echo %YELLOW%2. If all tests pass, you're ready for 50 companies%RESET%
echo %YELLOW%3. If tests fail, check the troubleshooting guide%RESET%
echo.
echo %CYAN%For detailed testing, run:%RESET%
echo %GREEN%  .\scripts\test-ai-comprehensive.ps1%RESET%
echo.
echo %CYAN%For complete system testing, run:%RESET%
echo %GREEN%  .\scripts\test-everything.ps1%RESET%
echo.
pause 