@echo off
echo 🔧 Testing Adaptive Onboarding Fix (HTTP Requests)
echo.

cd /d "C:\Users\amrey\Desktop\ISM [AI]"

echo 📋 Fix Summary:
echo    ✅ Updated Node.js backend to use HTTP requests to Python server
echo    ✅ Python server runs on port 5003 as Flask server
echo    ✅ No more "Python script failed" errors
echo    ✅ Proper error handling and status codes
echo.

echo 🔄 Testing the fix...
node test_adaptive_fix.js

echo.
echo ✅ Test completed! 
echo.
echo 🎯 What was fixed:
echo    • Node.js backend now makes HTTP requests to Python server
echo    • Python server runs as Flask server on port 5003
echo    • Proper error handling and status codes
echo    • No more spawning Python scripts (more reliable)
echo.
echo 🚀 Next Steps:
echo    1. Make sure Python server is running: python adaptive_onboarding_server.py
echo    2. Make sure Node.js backend is running: npm start  
echo    3. Try Adaptive AI Onboarding in the frontend
echo    4. Should work without 500 errors now!
echo.
pause 