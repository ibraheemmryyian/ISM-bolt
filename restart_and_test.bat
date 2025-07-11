@echo off
echo 🔄 Restarting Backend and Testing AI Services
echo ================================================

echo.
echo 1. Stopping any existing Node.js processes...
taskkill /f /im node.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo 2. Starting backend...
cd backend
start /b npm start
timeout /t 5 /nobreak >nul

echo.
echo 3. Testing AI Services endpoint...
curl -s http://localhost:3000/api/ai/services/status

echo.
echo 4. Running full test suite...
cd ..
python test_final_repairs.py

echo.
echo ✅ Test complete!
pause 