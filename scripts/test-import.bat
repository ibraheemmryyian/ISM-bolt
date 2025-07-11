@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🧪 Testing Real Data Import Process
echo ===================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Running import test...%RESET%
echo.

REM Check if services are running
echo %YELLOW%Checking services...%RESET%

REM Test backend
curl -s http://localhost:3000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Backend: Running%RESET%
) else (
    echo %RED%❌ Backend: Not running%RESET%
)

REM Test AI gateway
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ AI Gateway: Running%RESET%
) else (
    echo %RED%❌ AI Gateway: Not running%RESET%
)

REM Test frontend
curl -s http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Frontend: Running%RESET%
) else (
    echo %RED%❌ Frontend: Not running%RESET%
)

echo.
echo %BLUE%Testing data import with 5 companies...%RESET%

REM Create a test file with just 5 companies
echo %YELLOW%Creating test data...%RESET%
cd backend
python -c "
import json
with open('../data/50_real_gulf_companies_cleaned.json', 'r') as f:
    data = json.load(f)
with open('test_companies.json', 'w') as f:
    json.dump(data[:5], f, indent=2)
print('Test data created with 5 companies')
"

REM Run test import
echo %YELLOW%Running test import...%RESET%
python -c "
import asyncio
import sys
sys.path.append('.')
from real_data_bulk_importer import RealDataBulkImporter

async def test_import():
    importer = RealDataBulkImporter()
    importer.companies_data = importer.companies_data[:5]  # Only 5 companies
    success = await importer.run_complete_import()
    return success

result = asyncio.run(test_import())
print('Test import completed:', 'SUCCESS' if result else 'FAILED')
"

cd ..
echo.
echo %GREEN%✅ Test completed!%RESET%
echo %BLUE%Check the admin dashboard to see the imported data:%RESET%
echo %GREEN%http://localhost:5173/admin%RESET%
echo.
pause 