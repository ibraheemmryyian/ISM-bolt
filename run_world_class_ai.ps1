# World-Class AI Material Generation and Matching System
# PowerShell Script

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  WORLD-CLASS AI MATERIAL SYSTEM" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to backend directory
Set-Location backend

Write-Host "🚀 Starting world-class AI material generation and matching..." -ForegroundColor Green
Write-Host ""

Write-Host "📊 Loading 115 companies from real-world data..." -ForegroundColor Yellow
Write-Host "🤖 Initializing world-class AI algorithms..." -ForegroundColor Yellow
Write-Host "🔗 Building advanced material matching networks..." -ForegroundColor Yellow
Write-Host ""

try {
    # Run the world-class AI system
    python generate_supervised_materials_and_matches.py
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  GENERATION COMPLETE!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "📁 Generated files:" -ForegroundColor White
    Write-Host "   - material_listings.csv" -ForegroundColor White
    Write-Host "   - material_matches.csv" -ForegroundColor White
    Write-Host ""
    Write-Host "🎉 World-class AI system execution finished!" -ForegroundColor Green
    Write-Host ""
    
} catch {
    Write-Host ""
    Write-Host "❌ ERROR: Failed to run world-class AI system" -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

# Return to original directory
Set-Location ..

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 