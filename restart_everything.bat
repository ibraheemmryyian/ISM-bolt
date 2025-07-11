@echo off
echo Killing all Node.js processes...
taskkill /F /IM node.exe 2>nul
echo.

echo Waiting 3 seconds...
timeout /t 3 /nobreak >nul
echo.

echo Starting backend on port 5001...
cd backend
start "Backend Server" cmd /k "npm start"
echo.

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul
echo.

echo Testing backend health...
curl -s http://localhost:5001/api/health
echo.

echo Backend should now be running on port 5001
echo You can now test the Python AI endpoints!
echo.
pause 