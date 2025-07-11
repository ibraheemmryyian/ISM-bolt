# SymbioFlows Demo Startup Script
# This script starts all services needed for the millionaire demo

Write-Host "🚀 Starting SymbioFlows Demo Environment..." -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Function to check if port is in use
function Test-Port {
    param($Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    }
    catch {
        return $false
    }
}

# Function to kill process on port
function Stop-ProcessOnPort {
    param($Port)
    try {
        $process = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
        if ($process) {
            $processId = $process.OwningProcess
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
            Write-Host "Killed process on port $Port" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "No process found on port $Port" -ForegroundColor Gray
    }
}

# Kill any existing processes on our ports
Write-Host "Cleaning up existing processes..." -ForegroundColor Yellow
Stop-ProcessOnPort 5001
Stop-ProcessOnPort 5003
Stop-ProcessOnPort 5173

Start-Sleep -Seconds 2

# Start Python AI Onboarding Server
Write-Host "Starting Python AI Onboarding Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\amrey\Desktop\ISM [AI]\backend'; python adaptive_onboarding_server.py" -WindowStyle Minimized

# Wait for Python server to start
Write-Host "Waiting for Python server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if Python server is running
if (Test-Port 5003) {
    Write-Host "✅ Python AI Onboarding Server running on port 5003" -ForegroundColor Green
} else {
    Write-Host "❌ Python server failed to start" -ForegroundColor Red
}

# Start Backend Server
Write-Host "Starting Backend Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\amrey\Desktop\ISM [AI]\backend'; npm start" -WindowStyle Minimized

# Wait for backend to start
Write-Host "Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check if backend is running
if (Test-Port 5001) {
    Write-Host "✅ Backend Server running on port 5001" -ForegroundColor Green
} else {
    Write-Host "❌ Backend server failed to start" -ForegroundColor Red
}

# Start Frontend
Write-Host "Starting Frontend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\amrey\Desktop\ISM [AI]\frontend'; npm run dev" -WindowStyle Minimized

# Wait for frontend to start
Write-Host "Waiting for frontend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Check if frontend is running
if (Test-Port 5173) {
    Write-Host "✅ Frontend running on port 5173" -ForegroundColor Green
} else {
    Write-Host "❌ Frontend failed to start" -ForegroundColor Red
}

# Final status check
Write-Host ""
Write-Host "🎯 Demo Environment Status:" -ForegroundColor Green
Write-Host "==========================" -ForegroundColor Green

if (Test-Port 5003) {
    Write-Host "✅ Python AI Server: http://localhost:5003" -ForegroundColor Green
} else {
    Write-Host "❌ Python AI Server: Not running" -ForegroundColor Red
}

if (Test-Port 5001) {
    Write-Host "✅ Backend API: http://localhost:5001" -ForegroundColor Green
} else {
    Write-Host "❌ Backend API: Not running" -ForegroundColor Red
}

if (Test-Port 5173) {
    Write-Host "✅ Frontend App: http://localhost:5173" -ForegroundColor Green
} else {
    Write-Host "❌ Frontend App: Not running" -ForegroundColor Red
}

Write-Host ""
Write-Host "🌐 Demo URLs:" -ForegroundColor Cyan
Write-Host "=============" -ForegroundColor Cyan
Write-Host "Landing Page: http://localhost:5173/" -ForegroundColor White
Write-Host "Demo Dashboard: http://localhost:5173/demo/dashboard" -ForegroundColor White
Write-Host "AI Matching: http://localhost:5173/demo/ai-matching" -ForegroundColor White
Write-Host "Backend API: http://localhost:5001/api/health" -ForegroundColor White

Write-Host ""
Write-Host "📋 Demo Checklist:" -ForegroundColor Yellow
Write-Host "==================" -ForegroundColor Yellow
Write-Host "1. Open landing page and verify metrics" -ForegroundColor White
Write-Host "2. Test AI inference flow" -ForegroundColor White
Write-Host "3. Verify matching results" -ForegroundColor White
Write-Host "4. Check logistics preview" -ForegroundColor White
Write-Host "5. Test carbon tracking" -ForegroundColor White

Write-Host ""
Write-Host "🎉 Demo environment is ready!" -ForegroundColor Green
Write-Host "Press any key to open the demo in your browser..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Open demo in browser
Start-Process "http://localhost:5173/"

Write-Host ""
Write-Host "Demo is now running! Good luck with your meeting! 🚀" -ForegroundColor Green 