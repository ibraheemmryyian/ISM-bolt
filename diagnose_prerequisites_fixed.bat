@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🔍 SYMBIOFLOWS - PREREQUISITE DIAGNOSTIC (FIXED)
echo ========================================
echo 🎯 Checking System Prerequisites
echo ========================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%🔍 Running prerequisite diagnostics...%RESET%
echo.

REM ========================================
REM NODE.JS CHECK
REM ========================================

echo %YELLOW%1. Checking Node.js...%RESET%

node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Node.js found and working%RESET%
    node --version
) else (
    echo %RED%   ❌ Node.js not found in PATH%RESET%
    echo %YELLOW%   💡 Solution: Install Node.js from https://nodejs.org/%RESET%
    echo %YELLOW%   💡 Make sure to check "Add to PATH" during installation%RESET%
)

echo.

REM ========================================
REM PYTHON CHECK
REM ========================================

echo %YELLOW%2. Checking Python...%RESET%

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Python found and working%RESET%
    python --version
) else (
    echo %RED%   ❌ Python not found in PATH%RESET%
    echo %YELLOW%   💡 Solution: Install Python from https://python.org/%RESET%
    echo %YELLOW%   💡 Make sure to check "Add to PATH" during installation%RESET%
)

echo.

REM ========================================
REM NPM CHECK (WITH TIMEOUT)
REM ========================================

echo %YELLOW%3. Checking npm...%RESET%

REM Try npm with a timeout to prevent hanging
timeout /t 1 /nobreak >nul
npm --version >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ npm found and working%RESET%
    npm --version
) else (
    echo %RED%   ❌ npm not found or not responding%RESET%
    echo %YELLOW%   💡 Solution: npm comes with Node.js%RESET%
    echo %YELLOW%   💡 Reinstall Node.js and ensure PATH is set%RESET%
)

echo.

REM ========================================
REM MANUAL COMMAND TESTING
REM ========================================

echo %YELLOW%4. Manual command testing...%RESET%

echo %BLUE%   Testing commands manually:%RESET%
echo %YELLOW%   node --version%RESET%
node --version
echo.

echo %YELLOW%   python --version%RESET%
python --version
echo.

echo %YELLOW%   npm --version%RESET%
npm --version
echo.

REM ========================================
REM COMMON INSTALLATION PATHS
REM ========================================

echo %YELLOW%5. Checking common installation paths...%RESET%

REM Check for Node.js in common paths
if exist "C:\Program Files\nodejs\node.exe" (
    echo %GREEN%   ✅ Node.js found in C:\Program Files\nodejs\%RESET%
) else (
    echo %RED%   ❌ Node.js not found in C:\Program Files\nodejs\%RESET%
)

if exist "C:\Program Files (x86)\nodejs\node.exe" (
    echo %GREEN%   ✅ Node.js found in C:\Program Files (x86)\nodejs\%RESET%
) else (
    echo %RED%   ❌ Node.js not found in C:\Program Files (x86)\nodejs\%RESET%
)

REM Check for Python in common paths
if exist "C:\Python*\python.exe" (
    echo %GREEN%   ✅ Python found in C:\Python*%RESET%
) else (
    echo %RED%   ❌ Python not found in C:\Python*%RESET%
)

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python*\python.exe" (
    echo %GREEN%   ✅ Python found in AppData\Local\Programs\Python\%RESET%
) else (
    echo %RED%   ❌ Python not found in AppData\Local\Programs\Python\%RESET%
)

echo.

REM ========================================
REM SUMMARY
REM ========================================

echo ========================================
echo 📊 DIAGNOSTIC SUMMARY
echo ========================================
echo.
echo %BLUE%🎯 Status:%RESET%

REM Count missing prerequisites
set MISSING_COUNT=0
node --version >nul 2>&1 || set /a MISSING_COUNT+=1
python --version >nul 2>&1 || set /a MISSING_COUNT+=1
npm --version >nul 2>&1 || set /a MISSING_COUNT+=1

if %MISSING_COUNT% equ 0 (
    echo %GREEN%✅ All prerequisites are available!%RESET%
    echo %GREEN%✅ You can run start_complete_system.bat%RESET%
) else (
    echo %RED%❌ %MISSING_COUNT% prerequisite(s) missing%RESET%
    echo %YELLOW%💡 Please install missing prerequisites first%RESET%
)

echo.
echo %BLUE%📋 Next Steps:%RESET%
echo   1. Install any missing prerequisites
echo   2. Ensure they are added to PATH
echo   3. Restart command prompt
echo   4. Run this diagnostic again
echo   5. Then run start_complete_system.bat
echo.
echo ========================================
pause 