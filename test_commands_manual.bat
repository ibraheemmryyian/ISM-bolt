@echo off
chcp 65001 >nul

echo ========================================
echo 🧪 MANUAL COMMAND TESTING
echo ========================================
echo Testing basic commands manually
echo ========================================

echo.
echo Testing Node.js...
node --version
if %errorlevel% equ 0 (
    echo ✅ Node.js is working
) else (
    echo ❌ Node.js failed
)

echo.
echo Testing Python...
python --version
if %errorlevel% equ 0 (
    echo ✅ Python is working
) else (
    echo ❌ Python failed
)

echo.
echo Testing npm...
npm --version
if %errorlevel% equ 0 (
    echo ✅ npm is working
) else (
    echo ❌ npm failed
)

echo.
echo ========================================
echo 📊 RESULTS SUMMARY
echo ========================================

set WORKING=0
node --version >nul 2>&1 && set /a WORKING+=1
python --version >nul 2>&1 && set /a WORKING+=1
npm --version >nul 2>&1 && set /a WORKING+=1

echo Working commands: %WORKING% out of 3
echo.

if %WORKING% equ 3 (
    echo 🎉 All commands working! You can run the system.
) else (
    echo ⚠️ Some commands failed. Check the output above.
)

echo.
pause 