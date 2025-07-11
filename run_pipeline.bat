@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 ISM AI - COMPLETE PIPELINE RUNNER
echo ====================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%Starting complete AI pipeline for 50 companies...%RESET%
echo.

REM Check if backend is running
echo %YELLOW%Checking backend...%RESET%
curl -s http://localhost:5001/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Backend is running%RESET%
) else (
    echo %RED%❌ Backend is not running. Please start the backend first.%RESET%
    pause
    exit /b 1
)

echo.
echo %BLUE%Running complete pipeline...%RESET%
python run_complete_pipeline.py

if %errorlevel% equ 0 (
    echo.
    echo %GREEN%✅ Pipeline completed successfully!%RESET%
    echo.
    echo %BLUE%Check your results:%RESET%
    echo %GREEN%• Admin Dashboard: http://localhost:5001/api/admin/stats%RESET%
    echo %GREEN%• Frontend: http://localhost:5173%RESET%
    echo %GREEN%• Report: pipeline_report.json%RESET%
) else (
    echo.
    echo %RED%❌ Pipeline failed. Check the errors above.%RESET%
)

echo.
pause 