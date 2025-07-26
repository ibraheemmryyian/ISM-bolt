@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🚀 UPDATE HOSTED CONTACT INFORMATION
echo ========================================
echo 🎯 Force updating hosted website with correct contact info
echo ========================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%🔍 Starting hosted contact info update...%RESET%
echo.

REM ========================================
REM VERIFY CURRENT CONTACT INFO
REM ========================================

echo %YELLOW%1. Verifying current contact information...%RESET%

REM Check SymbioFlows_Pitch_Deck.html
findstr /C:"+962 792 313 484" "SymbioFlows_Pitch_Deck.html" >nul
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Phone number is correct in pitch deck%RESET%
) else (
    echo %RED%   ❌ Phone number is incorrect in pitch deck%RESET%
)

REM Check for old Pittsburgh references
findstr /C:"Pittsburgh" "SymbioFlows_Pitch_Deck.html" >nul
if %errorlevel% equ 0 (
    echo %RED%   ❌ Still contains Pittsburgh reference%RESET%
) else (
    echo %GREEN%   ✅ No Pittsburgh references found%RESET%
)

echo.

REM ========================================
REM FORCE REBUILD AND DEPLOYMENT
REM ========================================

echo %YELLOW%2. Forcing rebuild and deployment...%RESET%

REM Create a timestamp file to force rebuild
echo %BLUE%Creating rebuild trigger...%RESET%
echo # Contact Information Update - $(Get-Date -Format "yyyy-MM-dd HH:mm:ss") > REBUILD_TRIGGER.txt
echo # Phone: +962 792 313 484 >> REBUILD_TRIGGER.txt
echo # Location: Amman, Jordan >> REBUILD_TRIGGER.txt
echo # Email: info@symbioflows.ai >> REBUILD_TRIGGER.txt

REM Add to git
git add REBUILD_TRIGGER.txt
git commit -m "Force rebuild: Update contact information to +962 792 313 484"

REM Push to trigger deployment
echo %BLUE%Pushing to trigger deployment...%RESET%
git push origin main

if %errorlevel% neq 0 (
    echo %RED%❌ Push failed%RESET%
    echo %YELLOW%Please check your git configuration and try again%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ Push successful - deployment triggered%RESET%
echo.

REM ========================================
REM CLEAR CACHE INSTRUCTIONS
REM ========================================

echo %YELLOW%3. Cache clearing instructions...%RESET%
echo.
echo %BLUE%If you still see old contact information after deployment:%RESET%
echo   1. Clear your browser cache (Ctrl+Shift+Delete)
echo   2. Hard refresh the page (Ctrl+F5)
echo   3. Try incognito/private browsing mode
echo   4. Wait 5-10 minutes for CDN cache to clear
echo.

REM ========================================
REM DEPLOYMENT STATUS
REM ========================================

echo ========================================
echo 📊 DEPLOYMENT STATUS
echo ========================================
echo.
echo %GREEN%🎉 Deployment has been triggered!%RESET%
echo.
echo %BLUE%Expected timeline:%RESET%
echo   • GitHub Actions: 2-5 minutes to build
echo   • Kubernetes deployment: 3-5 minutes
echo   • DNS/CDN propagation: 5-10 minutes
echo   • Total time: 10-20 minutes
echo.
echo %BLUE%Monitor deployment:%RESET%
echo   • Check GitHub Actions tab in your repository
echo   • Look for "deploy-production" job
echo   • Verify all steps complete successfully
echo.
echo %BLUE%Contact information should show:%RESET%
echo   • Phone: +962 792 313 484
echo   • Location: Amman, Jordan  
echo   • Email: info@symbioflows.ai
echo.
echo ========================================
pause 