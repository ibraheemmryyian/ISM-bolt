@echo off
echo 🚀 Starting ISM AI System...
echo.

REM Change to the correct directory
cd /d "C:\Users\amrey\Desktop\ISM [AI]"

echo 📁 Current directory: %CD%
echo.

REM Start backend
echo 🔧 Starting backend...
cd backend
start "Backend Server" cmd /k "npm start"

REM Wait a moment for backend to start
timeout /t 5 /nobreak >nul

REM Go back to root and run test
cd ..
echo.
echo 🧪 Running 50 companies test...
python test_50_companies.py

echo.
echo ✅ System started! Check:
echo    - Backend: http://localhost:5001/api/health
echo    - Frontend: http://localhost:5173
echo    - Admin: http://localhost:5173/admin
echo.
pause 