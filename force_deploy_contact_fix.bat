@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🚀 FORCE DEPLOYMENT - CONTACT INFO FIX
echo ========================================
echo 🎯 Updating hosted website with correct contact information
echo ========================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%🔍 Starting forced deployment...%RESET%
echo.

REM ========================================
REM VERIFY CHANGES ARE COMMITTED
REM ========================================

echo %YELLOW%1. Verifying changes are committed...%RESET%

REM Check if we're on main branch
git branch --show-current | findstr "main" >nul
if %errorlevel% neq 0 (
    echo %RED%❌ Not on main branch%RESET%
    echo %YELLOW%Please switch to main branch first%RESET%
    pause
    exit /b 1
)

REM Check if working tree is clean
git status --porcelain | findstr . >nul
if %errorlevel% equ 0 (
    echo %RED%❌ Working tree has uncommitted changes%RESET%
    echo %YELLOW%Please commit all changes first%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ Changes are committed%RESET%
echo.

REM ========================================
REM FORCE PUSH TO TRIGGER DEPLOYMENT
REM ========================================

echo %YELLOW%2. Force pushing to trigger deployment...%RESET%

REM Create a small change to force deployment
echo %BLUE%Creating deployment trigger...%RESET%
echo # Contact information updated - $(Get-Date) >> DEPLOYMENT_TRIGGER.md

REM Add and commit the trigger
git add DEPLOYMENT_TRIGGER.md
git commit -m "Trigger deployment: Update contact information to +962 792 313 484"

REM Push to trigger GitHub Actions
echo %BLUE%Pushing to trigger deployment...%RESET%
git push origin main

if %errorlevel% neq 0 (
    echo %RED%❌ Push failed%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ Push successful - deployment triggered%RESET%
echo.

REM ========================================
REM CLEANUP
REM ========================================

echo %YELLOW%3. Cleaning up...%RESET%

REM Remove the trigger file
del DEPLOYMENT_TRIGGER.md
git add DEPLOYMENT_TRIGGER.md
git commit -m "Remove deployment trigger file"

echo %GREEN%✅ Cleanup completed%RESET%
echo.

REM ========================================
REM DEPLOYMENT SUMMARY
REM ========================================

echo ========================================
echo 📊 DEPLOYMENT TRIGGERED
echo ========================================
echo.
echo %GREEN%🎉 Deployment has been triggered!%RESET%
echo.
echo %BLUE%What happens next:%RESET%
echo   1. GitHub Actions will build new Docker images
echo   2. Images will be deployed to production
echo   3. Contact information will be updated on hosted site
echo   4. Deployment typically takes 5-10 minutes
echo.
echo %BLUE%Monitor deployment:%RESET%
echo   • GitHub Actions: https://github.com/your-repo/actions
echo   • Production URL: https://your-domain.com
echo.
echo %BLUE%Expected changes:%RESET%
echo   • Phone: +962 792 313 484
echo   • Location: Amman, Jordan
echo   • Email: info@symbioflows.ai (unchanged)
echo.
echo ========================================
pause 