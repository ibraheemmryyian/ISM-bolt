@echo off
REM run_demo.bat - Script to run the complete demo system for video recording

echo ========================================================
echo   SymbioFlows Demo System Launcher
echo ========================================================
echo This script will prepare and start all necessary services
echo for recording a demo video of the complete user journey.
echo.

REM Check for required commands
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is required but not installed. Aborting.
    exit /b 1
)

where npm >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo npm is required but not installed. Aborting.
    exit /b 1
)

REM Set up environment
echo Setting up environment...
set NODE_ENV=development
set DEMO_MODE=true

REM Create data directory if it doesn't exist
if not exist data mkdir data

REM Prepare demo data
echo Preparing demo data...
python scripts\demo_video_preparation.py

REM Check if preparation was successful
if %ERRORLEVEL% NEQ 0 (
    echo Error: Demo preparation failed. Please check logs.
    exit /b 1
)

REM Start backend services in a new window
echo Starting backend services...
start "SymbioFlows Backend" cmd /c "cd backend && npm start"

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 10 /nobreak

REM Start frontend in a new window
echo Starting frontend...
start "SymbioFlows Frontend" cmd /c "cd frontend && npm run dev"

REM Display success message
echo.
echo ========================================================
echo   Demo System Started Successfully!
echo ========================================================
echo.
echo Access the application at: http://localhost:3000
echo.
echo Follow the demo guide at: docs/DEMO_VIDEO_GUIDE.md
echo.
echo Close the terminal windows to stop the services when done.
echo ========================================================

REM Keep the main window open
echo.
echo Press any key to exit this window (services will continue running)...
pause >nul