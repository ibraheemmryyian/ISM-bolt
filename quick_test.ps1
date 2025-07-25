Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 QUICK PREREQUISITE TEST (PowerShell)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Testing Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>$null
    if ($nodeVersion) {
        Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Node.js not found" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Node.js not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "Testing Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>$null
    if ($pythonVersion) {
        Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Python not found" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Python not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "Testing npm..." -ForegroundColor Yellow
try {
    $npmVersion = npm --version 2>$null
    if ($npmVersion) {
        Write-Host "✅ npm: $npmVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ npm not found" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ npm not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ Test Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Read-Host "Press Enter to continue" 