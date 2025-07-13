@echo off
echo.
echo =====================================================
echo 🚀 ISM AI Platform - Supabase Migration Runner
echo =====================================================
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Node.js is not installed
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if migration file exists
if not exist "database_migration_for_real_data.sql" (
    echo ❌ Migration file not found: database_migration_for_real_data.sql
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist "backend\.env" (
    echo ❌ .env file not found in backend directory
    echo Please run the migration script first to create the .env file
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed
echo.
echo 🚀 Running Supabase migration...
echo.

REM Run the migration script
node run_supabase_migration.js

echo.
echo =====================================================
echo Migration completed!
echo =====================================================
echo.
pause 