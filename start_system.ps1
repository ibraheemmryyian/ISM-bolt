# ISM AI System Starter
Write-Host "🚀 Starting ISM AI System..." -ForegroundColor Green
Write-Host ""

# Change to the correct directory
$projectPath = "C:\Users\amrey\Desktop\ISM [AI]"
Set-Location $projectPath

Write-Host "📁 Current directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""

# Start backend
Write-Host "🔧 Starting backend..." -ForegroundColor Yellow
Set-Location "backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm start" -WindowStyle Normal

# Wait for backend to start
Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

# Go back to root and run test
Set-Location $projectPath
Write-Host ""
Write-Host "🧪 Running 50 companies test..." -ForegroundColor Yellow
python test_50_companies.py

Write-Host ""
Write-Host "✅ System started! Check:" -ForegroundColor Green
Write-Host "   - Backend: http://localhost:5001/api/health" -ForegroundColor Cyan
Write-Host "   - Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "   - Admin: http://localhost:5173/admin" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to continue..." 