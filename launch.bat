@echo off
echo Starting Industrial Symbiosis AI Platform...

echo.
echo Step 1: Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Step 2: Starting Backend Server...
cd backend
start "Backend Server" cmd /k "npm start"

echo.
echo Step 3: Starting Frontend...
cd ../frontend
start "Frontend" cmd /k "npm run dev"

echo.
echo Step 4: Opening browser...
timeout /t 5 /nobreak >nul
start http://localhost:5173

echo.
echo System is starting up! 
echo Backend: http://localhost:3001
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit this launcher...
pause >nul 