@echo off
echo ========================================
echo Clean Material Data Generator
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found. Checking dependencies...

REM Check if required packages are installed
python -c "import torch, numpy, aiohttp, asyncio" >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install torch numpy aiohttp asyncio
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Dependencies OK. Starting Clean Material Data Generator...
echo.

REM Run the clean material data generator
python generate_supervised_materials_and_matches.py

if errorlevel 1 (
    echo.
    echo ERROR: Material data generation failed
    echo Check the error messages above for details
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS: Material Data Generation Complete
echo ========================================
echo.
echo Generated files:
echo - material_listings.csv
echo - material_matches.csv
echo.
echo Check the current directory for the output files.
pause 