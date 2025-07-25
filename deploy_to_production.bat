@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 🚀 SYMBIOFLOWS - PRODUCTION DEPLOYMENT
echo ========================================
echo 🎯 Deploying to Production with Full Validation
echo ========================================

REM Colors
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%🔍 Starting production deployment...%RESET%
echo.

REM ========================================
REM PRE-DEPLOYMENT CHECKS
REM ========================================

echo %YELLOW%1. Pre-deployment validation...%RESET%

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Docker is not installed or not in PATH%RESET%
    echo %YELLOW%Please install Docker Desktop and try again%RESET%
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Docker Compose is not available%RESET%
    echo %YELLOW%Please install Docker Compose and try again%RESET%
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo %RED%❌ Environment file (.env) not found%RESET%
    echo %YELLOW%Please create .env file with required variables%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ Pre-deployment checks passed%RESET%
echo.

REM ========================================
REM STOP EXISTING SERVICES
REM ========================================

echo %YELLOW%2. Stopping existing services...%RESET%

REM Stop any running containers
docker-compose -f docker-compose.prod.yml down 2>nul
if %errorlevel% equ 0 (
    echo %GREEN%✅ Stopped existing services%RESET%
) else (
    echo %YELLOW%ℹ️ No existing services to stop%RESET%
)

echo.

REM ========================================
REM BUILD DOCKER IMAGES
REM ========================================

echo %YELLOW%3. Building Docker images...%RESET%

REM Build frontend
echo %BLUE%Building frontend image...%RESET%
docker-compose -f docker-compose.prod.yml build frontend
if %errorlevel% neq 0 (
    echo %RED%❌ Frontend build failed%RESET%
    pause
    exit /b 1
)

REM Build backend
echo %BLUE%Building backend image...%RESET%
docker-compose -f docker-compose.prod.yml build backend
if %errorlevel% neq 0 (
    echo %RED%❌ Backend build failed%RESET%
    pause
    exit /b 1
)

REM Build AI services
echo %BLUE%Building AI services image...%RESET%
docker-compose -f docker-compose.prod.yml build ai-services
if %errorlevel% neq 0 (
    echo %RED%❌ AI services build failed%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ All images built successfully%RESET%
echo.

REM ========================================
REM START SERVICES
REM ========================================

echo %YELLOW%4. Starting production services...%RESET%

REM Start all services
docker-compose -f docker-compose.prod.yml up -d
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to start services%RESET%
    pause
    exit /b 1
)

echo %GREEN%✅ Services started successfully%RESET%
echo.

REM ========================================
REM HEALTH CHECKS
REM ========================================

echo %YELLOW%5. Running health checks...%RESET%

REM Wait for services to be ready
echo %BLUE%Waiting for services to initialize...%RESET%
timeout /t 30 /nobreak >nul

REM Check Redis
echo %BLUE%Checking Redis...%RESET%
docker exec ism-redis redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Redis is healthy%RESET%
) else (
    echo %RED%   ❌ Redis health check failed%RESET%
)

REM Check Backend
echo %BLUE%Checking Backend...%RESET%
curl -s http://localhost:3001/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Backend is healthy%RESET%
) else (
    echo %RED%   ❌ Backend health check failed%RESET%
)

REM Check Frontend
echo %BLUE%Checking Frontend...%RESET%
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ Frontend is healthy%RESET%
) else (
    echo %RED%   ❌ Frontend health check failed%RESET%
)

REM Check AI Gateway
echo %BLUE%Checking AI Gateway...%RESET%
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%   ✅ AI Gateway is healthy%RESET%
) else (
    echo %RED%   ❌ AI Gateway health check failed%RESET%
)

echo.

REM ========================================
REM DEPLOYMENT SUMMARY
REM ========================================

echo ========================================
echo 📊 DEPLOYMENT SUMMARY
echo ========================================
echo.
echo %GREEN%🎉 Production deployment completed!%RESET%
echo.
echo %BLUE%Services running:%RESET%
echo   • Frontend: %GREEN%http://localhost:3000%RESET%
echo   • Backend: %GREEN%http://localhost:3001%RESET%
echo   • AI Gateway: %GREEN%http://localhost:5000%RESET%
echo   • Redis: %GREEN%localhost:6379%RESET%
echo.
echo %BLUE%Monitoring:%RESET%
echo   • Prometheus: %GREEN%http://localhost:9090%RESET%
echo   • Grafana: %GREEN%http://localhost:3000%RESET%
echo.
echo %BLUE%Useful commands:%RESET%
echo   • View logs: %YELLOW%docker-compose -f docker-compose.prod.yml logs -f%RESET%
echo   • Stop services: %YELLOW%docker-compose -f docker-compose.prod.yml down%RESET%
echo   • Restart services: %YELLOW%docker-compose -f docker-compose.prod.yml restart%RESET%
echo.
echo %BLUE%Next steps:%RESET%
echo   1. Test the application at http://localhost:3000
echo   2. Monitor logs for any issues
echo   3. Configure your domain and SSL certificates
echo   4. Set up monitoring and alerting
echo.
echo ========================================
pause 