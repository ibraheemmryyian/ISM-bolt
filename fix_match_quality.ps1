# PowerShell script for comprehensive match quality fix and improvement

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🔧 COMPREHENSIVE MATCH QUALITY FIX AND IMPROVEMENT" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will:" -ForegroundColor Yellow
Write-Host "1. Fix existing problematic matches (duplicates, generic names, etc.)" -ForegroundColor White
Write-Host "2. Generate new high-quality matches using improved AI engine" -ForegroundColor White
Write-Host "3. Validate results and provide comprehensive reporting" -ForegroundColor White
Write-Host ""
Write-Host "Starting process..." -ForegroundColor Green
Write-Host ""

# Change to backend directory
Set-Location -Path "$PSScriptRoot\backend"

# Run the Python script
try {
    python fix_and_improve_matches.py
    
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Green
    Write-Host "✅ Process completed successfully!" -ForegroundColor Green
    Write-Host "📊 Check material_matches.csv for the improved data" -ForegroundColor White
    Write-Host "📋 Check match_quality_improvement_report.json for detailed results" -ForegroundColor White
    Write-Host "================================================================================" -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Red
    Write-Host "❌ Process failed with error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "================================================================================" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to continue" 