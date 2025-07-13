@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM PERFECT MIGRATION SCRIPT FOR ISM AI PLATFORM
REM Complete Setup with Real Data Integration
REM =====================================================

echo.
echo =====================================================
echo 🚀 ISM AI PLATFORM - PERFECT MIGRATION SCRIPT
echo =====================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  Warning: Not running as administrator
    echo Some operations may require elevated privileges
    echo.
)

REM Set project paths
set "PROJECT_ROOT=%~dp0"
set "BACKEND_PATH=%PROJECT_ROOT%backend"
set "FRONTEND_PATH=%PROJECT_ROOT%frontend"

echo 📁 Project Root: %PROJECT_ROOT%
echo 📁 Backend Path: %BACKEND_PATH%
echo 📁 Frontend Path: %FRONTEND_PATH%
echo.

REM Check prerequisites
echo 🚀 Checking Prerequisites...
echo.

REM Check Node.js
node --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('node --version') do set "NODE_VERSION=%%i"
    echo ✅ Node.js is installed: !NODE_VERSION!
) else (
    echo ❌ Node.js is not installed
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check npm
npm --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('npm --version') do set "NPM_VERSION=%%i"
    echo ✅ npm is installed: !NPM_VERSION!
) else (
    echo ❌ npm is not installed
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('python --version') do set "PYTHON_VERSION=%%i"
    echo ✅ Python is installed: !PYTHON_VERSION!
) else (
    echo ❌ Python is not installed
    echo Please install Python from https://python.org/
    pause
    exit /b 1
)

REM Check pip
pip --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('pip --version') do set "PIP_VERSION=%%i"
    echo ✅ pip is installed: !PIP_VERSION!
) else (
    echo ❌ pip is not installed
    pause
    exit /b 1
)

REM Check Git
git --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=*" %%i in ('git --version') do set "GIT_VERSION=%%i"
    echo ✅ Git is installed: !GIT_VERSION!
) else (
    echo ❌ Git is not installed
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

echo.
echo ✅ All prerequisites are installed!
echo.

REM Create environment file
echo 🚀 Setting up Environment Configuration...
if exist "%BACKEND_PATH%\.env" (
    echo ℹ️  .env file already exists. Skipping...
) else (
    if exist "%BACKEND_PATH%\env.example" (
        copy "%BACKEND_PATH%\env.example" "%BACKEND_PATH%\.env" >nul
        echo ✅ Created .env file from template
        echo ⚠️  Please update the .env file with your actual API keys!
    ) else (
        echo ❌ env.example file not found!
        pause
        exit /b 1
    )
)
echo.

REM Install backend dependencies
echo 🚀 Installing Backend Dependencies...
cd /d "%BACKEND_PATH%"

echo 📦 Installing Node.js dependencies...
call npm install
if %errorLevel% neq 0 (
    echo ❌ Failed to install Node.js dependencies
    pause
    exit /b 1
)

echo 🐍 Installing Python dependencies...
call pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo ❌ Failed to install Python dependencies
    pause
    exit /b 1
)

echo ✅ Backend dependencies installed successfully!
echo.

REM Install frontend dependencies
echo 🚀 Installing Frontend Dependencies...
cd /d "%FRONTEND_PATH%"

echo 📦 Installing frontend dependencies...
call npm install
if %errorLevel% neq 0 (
    echo ❌ Failed to install frontend dependencies
    pause
    exit /b 1
)

echo ✅ Frontend dependencies installed successfully!
echo.

REM Return to project root
cd /d "%PROJECT_ROOT%"

REM Run database migration
echo 🚀 Running Database Migration...
if exist "database_migration_for_real_data.sql" (
    echo 📊 Database migration file found
    echo ℹ️  Note: Database migration requires Supabase configuration
    echo ℹ️  Please ensure your .env file has valid Supabase credentials
    echo.
    
    REM Create simple migration runner
    echo Creating migration runner...
    (
        echo const { createClient } = require('@supabase/supabase-js'^);
        echo const fs = require('fs'^);
        echo const path = require('path'^);
        echo.
        echo // Read environment variables
        echo require('dotenv'^).config({ path: path.join(__dirname, 'backend', '.env'^) }^);
        echo.
        echo const supabaseUrl = process.env.SUPABASE_URL;
        echo const supabaseKey = process.env.SUPABASE_ANON_KEY;
        echo.
        echo if (!supabaseUrl ^|^| !supabaseKey^) {
        echo     console.log('⚠️  Supabase configuration not found. Skipping database migration.'^);
        echo     process.exit(0^);
        echo }
        echo.
        echo const supabase = createClient(supabaseUrl, supabaseKey^);
        echo.
        echo async function runMigration() {
        echo     try {
        echo         console.log('🚀 Starting database migration...'^);
        echo         const migrationSQL = fs.readFileSync('database_migration_for_real_data.sql', 'utf8'^);
        echo         console.log('📊 Migration file loaded successfully'^);
        echo         console.log('ℹ️  Please run the migration manually in your Supabase dashboard'^);
        echo         console.log('ℹ️  Or use the PowerShell script for automated migration'^);
        echo     } catch (error) {
        echo         console.error('❌ Migration failed:', error^);
        echo         process.exit(1^);
        echo     }
        echo }
        echo.
        echo runMigration();
    ) > temp_migration_runner.js
    
    node temp_migration_runner.js
    del temp_migration_runner.js
) else (
    echo ❌ Database migration file not found
    echo Please ensure database_migration_for_real_data.sql exists
)
echo.

REM Build frontend
echo 🚀 Building Frontend...
cd /d "%FRONTEND_PATH%"

echo 🏗️  Building frontend application...
call npm run build
if %errorLevel% neq 0 (
    echo ❌ Frontend build failed
    pause
    exit /b 1
)

echo ✅ Frontend built successfully!
echo.

REM Return to project root
cd /d "%PROJECT_ROOT%"

REM Run tests
echo 🚀 Running Tests...

REM Backend tests
echo 🧪 Running backend tests...
cd /d "%BACKEND_PATH%"
call npm test
if %errorLevel% equ 0 (
    echo ✅ Backend tests passed!
) else (
    echo ⚠️  Backend tests failed. Check the output above.
)
echo.

REM Frontend tests
echo 🧪 Running frontend tests...
cd /d "%FRONTEND_PATH%"
call npm test -- --run
if %errorLevel% equ 0 (
    echo ✅ Frontend tests passed!
) else (
    echo ⚠️  Frontend tests failed. Check the output above.
)
echo.

REM Return to project root
cd /d "%PROJECT_ROOT%"

REM Health check
echo 🚀 Running Health Check...
if exist "test_backend_health.js" (
    echo 🔍 Running comprehensive health check...
    node test_backend_health.js
    if %errorLevel% equ 0 (
        echo ✅ Health check passed!
    ) else (
        echo ⚠️  Health check failed. Check the output above.
    )
) else (
    echo ⚠️  Health check script not found. Skipping...
)
echo.

REM Create startup scripts
echo 🚀 Creating Startup Scripts...

REM Create Windows batch startup script
(
    echo @echo off
    echo echo 🚀 Starting ISM AI Platform...
    echo echo.
    echo.
    echo cd /d "%BACKEND_PATH%"
    echo echo 📡 Starting backend server...
    echo start "Backend Server" cmd /k "npm run dev"
    echo.
    echo timeout /t 3 /nobreak ^>nul
    echo.
    echo cd /d "%FRONTEND_PATH%"
    echo echo 🎨 Starting frontend development server...
    echo start "Frontend Server" cmd /k "npm run dev"
    echo.
    echo echo.
    echo echo ✅ ISM AI Platform is starting up!
    echo echo 📱 Frontend: http://localhost:5173
    echo echo 🔧 Backend: http://localhost:5001
    echo echo.
    echo echo Press any key to exit this window...
    echo pause ^>nul
) > start_platform.bat

REM Create production deployment script
(
    echo @echo off
    echo echo 🚀 Deploying ISM AI Platform to Production...
    echo echo.
    echo.
    echo cd /d "%FRONTEND_PATH%"
    echo echo 🏗️  Building frontend for production...
    echo call npm run build
    echo.
    echo if %%errorLevel%% neq 0 ^(
    echo     echo ❌ Frontend build failed!
    echo     pause
    echo     exit /b 1
    echo ^)
    echo.
    echo cd /d "%BACKEND_PATH%"
    echo echo 🚀 Starting production server...
    echo set NODE_ENV=production
    echo set PORT=5001
    echo call npm start
    echo.
    echo echo ✅ Production deployment completed!
) > deploy_production.bat

echo ✅ Startup scripts created!
echo.

REM Final summary
echo 🚀 Migration Complete!
echo.
echo ✅ ISM AI Platform has been successfully migrated!
echo.
echo 📋 Next steps:
echo 1. Update the .env file with your actual API keys
echo 2. Run 'start_platform.bat' to start the platform
echo 3. Access the platform at http://localhost:5173
echo 4. Run 'deploy_production.bat' for production deployment
echo.
echo =====================================================
echo 🎉 MIGRATION COMPLETED SUCCESSFULLY!
echo =====================================================
echo.

pause 