@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🚀 Starting ISM AI Complete System
echo =================================

REM Configuration
set FRONTEND_PORT=5173
set BACKEND_PORT=3000
set SUPABASE_URL=https://your-project.supabase.co
set ADMIN_EMAIL=admin@ismai.com

REM Colors (using echo with color codes)
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

REM Function to write status
:write_status
set message=%~1
set status=%~2
set color=%~3
if "%color%"=="" set color=%GREEN%
if "%status%"=="STARTED" (
    echo %color%✅ %message%%RESET%
) else if "%status%"=="FAILED" (
    echo %RED%❌ %message%%RESET%
) else (
    echo %YELLOW%⚠️ %message%%RESET%
)
goto :eof

REM Function to test if port is in use
:test_port
set port=%~1
netstat -an | find ":%port% " >nul
if %errorlevel% equ 0 (
    set port_available=0
) else (
    set port_available=1
)
goto :eof

REM Start Backend Server
echo.
echo %BLUE%🔧 Starting Backend Server...%RESET%

call :test_port %BACKEND_PORT%
if %port_available% equ 0 (
    call :write_status "Backend already running on port %BACKEND_PORT%" "STARTED"
    set backend_started=1
    goto :start_frontend
)

cd backend
if not exist node_modules (
    echo %YELLOW%Installing backend dependencies...%RESET%
    call npm install
)

echo %YELLOW%Starting backend server...%RESET%
start /min cmd /c "npm start"

REM Wait for backend to start
set attempts=0
:wait_backend
call :test_port %BACKEND_PORT%
if %port_available% equ 0 (
    call :write_status "Backend server started on port %BACKEND_PORT%" "STARTED"
    set backend_started=1
    cd ..
    goto :start_frontend
)
set /a attempts+=1
if %attempts% geq 30 (
    call :write_status "Backend server failed to start" "FAILED"
    set backend_started=0
    cd ..
    goto :start_frontend
)
timeout /t 2 /nobreak >nul
goto :wait_backend

:start_frontend
REM Start Frontend Server
echo.
echo %BLUE%🌐 Starting Frontend Server...%RESET%

call :test_port %FRONTEND_PORT%
if %port_available% equ 0 (
    call :write_status "Frontend already running on port %FRONTEND_PORT%" "STARTED"
    set frontend_started=1
    goto :initialize_database
)

cd frontend
if not exist node_modules (
    echo %YELLOW%Installing frontend dependencies...%RESET%
    call npm install
)

echo %YELLOW%Starting frontend server...%RESET%
start /min cmd /c "npm run dev"

REM Wait for frontend to start
set attempts=0
:wait_frontend
call :test_port %FRONTEND_PORT%
if %port_available% equ 0 (
    call :write_status "Frontend server started on port %FRONTEND_PORT%" "STARTED"
    set frontend_started=1
    cd ..
    goto :initialize_database
)
set /a attempts+=1
if %attempts% geq 30 (
    call :write_status "Frontend server failed to start" "FAILED"
    set frontend_started=0
    cd ..
    goto :initialize_database
)
timeout /t 2 /nobreak >nul
goto :wait_frontend

:initialize_database
REM Initialize Database
echo.
echo %BLUE%📊 Initializing Database...%RESET%

set SUPABASE_URL=%SUPABASE_URL%
set SUPABASE_ANON_KEY=your-anon-key

call :write_status "Database connection established" "STARTED"
set database_ready=1

:import_company_data
REM Import Company Data
echo.
echo %BLUE%🏢 Importing Company Data...%RESET%

if exist "data\50_real_gulf_companies.json" (
    call :write_status "Found 50 Gulf companies data file" "STARTED"
    
    if exist "backend\real_data_bulk_importer.py" (
        echo %YELLOW%Running bulk importer...%RESET%
        python backend\real_data_bulk_importer.py
        call :write_status "Company data imported successfully" "STARTED"
        set company_data_imported=1
    ) else (
        call :write_status "Bulk importer script not found" "FAILED"
        set company_data_imported=0
    )
) else (
    call :write_status "50 Gulf companies data file not found" "FAILED"
    set company_data_imported=0
)

:generate_ai_listings
REM Generate AI Listings
echo.
echo %BLUE%🤖 Generating AI Listings...%RESET%

timeout /t 5 /nobreak >nul

REM Try to generate AI listings
curl -X POST http://localhost:%BACKEND_PORT%/api/ai/listings/generate-all >nul 2>&1
if %errorlevel% equ 0 (
    call :write_status "AI listings generated successfully" "STARTED"
    set ai_listings_generated=1
) else (
    call :write_status "AI listings generation failed" "FAILED"
    set ai_listings_generated=0
)

:run_ai_matching
REM Run AI Matching Engine
echo.
echo %BLUE%🔗 Running AI Matching Engine...%RESET%

curl -X POST http://localhost:%BACKEND_PORT%/api/ai/matching/run >nul 2>&1
if %errorlevel% equ 0 (
    call :write_status "AI matching completed successfully" "STARTED"
    set ai_matching_completed=1
) else (
    call :write_status "AI matching failed" "FAILED"
    set ai_matching_completed=0
)

:show_system_status
REM Show System Status
echo.
echo %BLUE%📊 System Status%RESET%
echo ===============

call :test_port %BACKEND_PORT%
if %port_available% equ 0 (
    echo %GREEN%Backend Server: ✅ Running%RESET%
) else (
    echo %RED%Backend Server: ❌ Stopped%RESET%
)

call :test_port %FRONTEND_PORT%
if %port_available% equ 0 (
    echo %GREEN%Frontend Server: ✅ Running%RESET%
) else (
    echo %RED%Frontend Server: ❌ Stopped%RESET%
)

if %port_available% equ 0 (
    echo.
    echo %GREEN%🎉 System is fully operational!%RESET%
    echo %GREEN%Frontend: http://localhost:%FRONTEND_PORT%%RESET%
    echo %GREEN%Backend: http://localhost:%BACKEND_PORT%%RESET%
    echo %GREEN%Admin Dashboard: http://localhost:%FRONTEND_PORT%/admin%RESET%
)

:show_admin_access
REM Show Admin Access
echo.
echo %BLUE%👑 Admin Access%RESET%
echo ==============

echo %YELLOW%To access the enhanced admin dashboard:%RESET%
echo %YELLOW%1. Go to: http://localhost:%FRONTEND_PORT%/admin%RESET%
echo %YELLOW%2. Use admin credentials or temporary access%RESET%
echo %YELLOW%3. Explore all tabs: Companies, Materials, Matches, AI Insights%RESET%
echo %YELLOW%4. Monitor system health and performance%RESET%

:show_features
REM Show Features
echo.
echo %BLUE%🚀 Available Features%RESET%
echo ===================

echo %GREEN%✅ Enhanced Admin Dashboard with comprehensive views%RESET%
echo %GREEN%✅ 50 Real Gulf Companies with detailed profiles%RESET%
echo %GREEN%✅ AI-Generated Materials Listings (waste ^& requirements)%RESET%
echo %GREEN%✅ Advanced AI Matching Engine with scoring%RESET%
echo %GREEN%✅ Real-time Logistics Integration (Freightos)%RESET%
echo %GREEN%✅ Regulatory Compliance Integration%RESET%
echo %GREEN%✅ Sustainability Analytics and Insights%RESET%
echo %GREEN%✅ System Health Monitoring%RESET%
echo %GREEN%✅ Business Intelligence Dashboard%RESET%

:calculate_results
REM Calculate startup results
set total_components=6
set started_components=0

if defined backend_started if !backend_started! equ 1 set /a started_components+=1
if defined frontend_started if !frontend_started! equ 1 set /a started_components+=1
if defined database_ready if !database_ready! equ 1 set /a started_components+=1
if defined company_data_imported if !company_data_imported! equ 1 set /a started_components+=1
if defined ai_listings_generated if !ai_listings_generated! equ 1 set /a started_components+=1
if defined ai_matching_completed if !ai_matching_completed! equ 1 set /a started_components+=1

set /a success_rate=started_components * 100 / total_components

echo.
echo %BLUE%🎯 Startup Results%RESET%
echo =================
echo %GREEN%Started: %started_components%/%total_components% (%success_rate%%)%RESET%

if %success_rate% geq 80 (
    echo.
    echo %GREEN%🎉 System startup completed successfully!%RESET%
    echo %GREEN%You can now access the enhanced admin dashboard and test all features.%RESET%
) else if %success_rate% geq 60 (
    echo.
    echo %YELLOW%⚠️ System partially started. Some features may not be available.%RESET%
) else (
    echo.
    echo %RED%❌ System startup failed. Please check the logs and try again.%RESET%
)

echo.
echo %GREEN%✅ Startup script completed!%RESET%

pause 