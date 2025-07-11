Write-Host "🔄 Restarting Backend and Testing AI Services" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green

Write-Host ""
Write-Host "1. Stopping any existing Node.js processes..." -ForegroundColor Yellow
Get-Process -Name "node" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "2. Starting backend..." -ForegroundColor Yellow
Set-Location "backend"
Start-Process -FilePath "npm" -ArgumentList "start" -WindowStyle Hidden
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "3. Testing AI Services endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:3000/api/ai/services/status" -Method Get -TimeoutSec 10
    Write-Host "✅ AI Services endpoint working!" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ AI Services endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "4. Running full test suite..." -ForegroundColor Yellow
Set-Location ".."
python test_final_repairs.py

Write-Host ""
Write-Host "✅ Test complete!" -ForegroundColor Green
Read-Host "Press Enter to continue" 