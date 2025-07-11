@echo off
echo 🧪 Testing ISM AI Backend...
echo.

REM Test backend health
echo Testing backend health...
curl -s http://localhost:5001/api/health
echo.
echo.

REM Test admin stats
echo Testing admin stats...
curl -s http://localhost:5001/api/admin/stats
echo.
echo.

echo ✅ Backend is running!
echo 🌐 Check your system:
echo    - Backend: http://localhost:5001/api/health
echo    - Admin: http://localhost:5001/api/admin/stats
echo.
pause 