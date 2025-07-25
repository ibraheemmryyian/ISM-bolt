@echo off
echo ========================================
echo SYMBIOFLOWS SYSTEM TEST
echo ========================================
echo.

echo [1/5] Checking Node.js installation...
node --version
if %errorlevel% neq 0 (
    echo ERROR: Node.js not found! Please install Node.js 18+
    pause
    exit /b 1
)

echo [2/5] Checking backend dependencies...
cd backend
if not exist node_modules (
    echo Installing backend dependencies...
    npm install
) else (
    echo Backend dependencies found.
)

echo [3/5] Checking frontend dependencies...
cd ../frontend
if not exist node_modules (
    echo Installing frontend dependencies...
    npm install
) else (
    echo Frontend dependencies found.
)

echo [4/5] Testing backend build...
cd ../backend
echo Starting backend server...
start "Backend Server" cmd /k "npm start"

echo [5/5] Testing frontend build...
cd ../frontend
echo Starting frontend server...
start "Frontend Server" cmd /k "npm run dev"

echo.
echo ========================================
echo SYSTEM STARTED!
echo ========================================
echo.
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:5000
echo.
echo Press any key to open the frontend...
pause >nul
start http://localhost:5173

echo.
echo System is running! Check the browser and terminal windows.
echo.
pause 