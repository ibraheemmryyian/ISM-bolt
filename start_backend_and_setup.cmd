@echo off
echo ========================================
echo   ISM AI - START BACKEND & SETUP
echo ========================================
echo.

echo [1/4] Starting backend server...
cd backend
start "Backend Server" cmd /k "npm start"
echo ✅ Backend started in new window
echo.

echo [2/4] Waiting for backend to be ready...
timeout /t 10 /nobreak >nul
echo ✅ Backend should be ready now
echo.

echo [3/4] Running complete system setup...
cd ..
python scripts/complete_system_setup.py
if %errorlevel% neq 0 (
    echo ❌ Setup failed
    pause
    exit /b 1
)
echo ✅ Setup completed
echo.

echo [4/4] Testing all systems...
python scripts/test_complete_system.py
if %errorlevel% neq 0 (
    echo ❌ Tests failed
    pause
    exit /b 1
)
echo ✅ All tests passed
echo.

echo ========================================
echo 🎉 SYSTEM IS FULLY OPERATIONAL!
echo ========================================
echo.
echo 📊 What you now have:
echo    - 50 Gulf companies with complete data
echo    - AI-generated material listings
echo    - Advanced AI matching (4-factor + GNN)
echo    - Multi-hop symbiosis networks
echo    - Admin user management
echo    - Real Freightos logistics integration
echo.
echo 🔗 Access your system:
echo    - Frontend: http://localhost:5173
echo    - Backend: http://localhost:5000
echo.
echo 🧪 Test everything:
echo    - Browse companies and listings
echo    - View AI matches and insights
echo    - Use admin features
echo    - Test logistics calculations
echo.
pause 