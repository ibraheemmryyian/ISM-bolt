@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🔍 SYMBIOFLOWS - PREREQUISITE DIAGNOSTIC
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
    for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
    echo %GREEN%   ✅ Node.js found: %NODE_VERSION%%RESET%
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
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo %GREEN%   ✅ Python found: %PYTHON_VERSION%%RESET%
) else (
    echo %RED%   ❌ Python not found in PATH%RESET%
    echo %YELLOW%   💡 Solution: Install Python from https://python.org/%RESET%
    echo %YELLOW%   💡 Make sure to check "Add to PATH" during installation%RESET%
)

echo.

REM ========================================
REM NPM CHECK
REM ========================================

echo %YELLOW%3. Checking npm...%RESET%

npm --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('npm --version') do set NPM_VERSION=%%i
    echo %GREEN%   ✅ npm found: %NPM_VERSION%%RESET%
) else (
    echo %RED%   ❌ npm not found in PATH%RESET%
    echo %YELLOW%   💡 Solution: npm comes with Node.js%RESET%
    echo %YELLOW%   💡 Reinstall Node.js and ensure PATH is set%RESET%
)

echo.

REM ========================================
REM PATH CHECK
REM ========================================

echo %YELLOW%4. Checking PATH environment...%RESET%

echo %BLUE%   Current PATH entries:%RESET%
echo %BLUE%   --------------------%RESET%
for %%p in (%PATH%) do (
    echo    %%p
)

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
REM QUICK FIXES
REM ========================================

echo %YELLOW%6. Quick Fixes...%RESET%

echo %BLUE%   If prerequisites are installed but not in PATH:%RESET%
echo %YELLOW%   1. Open System Properties (Win + Pause/Break)%RESET%
echo %YELLOW%   2. Click "Environment Variables"%RESET%
echo %YELLOW%   3. Edit "Path" variable%RESET%
echo %YELLOW%   4. Add missing paths:%RESET%
echo %YELLOW%      - C:\Program Files\nodejs\%RESET%
echo %YELLOW%      - C:\Python*\ (or your Python installation path)%RESET%
echo %YELLOW%   5. Restart command prompt%RESET%

echo.

REM ========================================
REM ALTERNATIVE COMMANDS
REM ========================================

echo %YELLOW%7. Alternative commands to try...%RESET%

echo %BLUE%   Try these commands if the above fail:%RESET%
echo %YELLOW%   node --version%RESET%
echo %YELLOW%   python --version%RESET%
echo %YELLOW%   python3 --version%RESET%
echo %YELLOW%   npm --version%RESET%

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