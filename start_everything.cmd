@echo off
echo ========================================
echo   ISM AI - START EVERYTHING
echo ========================================
echo.

echo [1/5] Killing any existing processes...
taskkill /f /im node.exe 2>nul
taskkill /f /im npm.exe 2>nul
echo ✅ Cleaned up existing processes
echo.

echo [2/5] Starting backend server...
cd backend
start "Backend Server" cmd /k "npm start"
echo ✅ Backend started in new window
echo.

echo [3/5] Waiting for backend to be ready...
timeout /t 15 /nobreak >nul
echo ✅ Backend should be ready now
echo.

echo [4/5] Testing backend with curl...
curl -s http://localhost:5000/api/health
if %errorlevel% neq 0 (
    echo ❌ Backend not responding
    echo Please check the backend window for errors
    pause
    exit /b 1
)
echo ✅ Backend is responding
echo.

echo [5/5] Running complete setup...
cd ..
python scripts/complete_system_setup.py
if %errorlevel% neq 0 (
    echo ❌ Setup failed
    pause
    exit /b 1
)
echo ✅ Setup completed
echo.

echo ========================================
echo 🎉 SYSTEM IS READY!
echo ========================================
echo.
echo 📊 What you now have:
echo    - Backend running on port 5000
echo    - 50 Gulf companies imported
echo    - AI-generated material listings
echo    - Advanced AI matching
echo    - Admin user management
echo.
echo 🔗 Access your system:
echo    - Frontend: http://localhost:5173
echo    - Backend: http://localhost:5000
echo.
echo 🧪 Test with curl:
echo    curl http://localhost:5000/api/health
echo    curl http://localhost:5000/api/companies
echo.
pause 