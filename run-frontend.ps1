# PowerShell script to run the frontend development server
Write-Host "Starting ISM AI Frontend..." -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "frontend")) {
    Write-Host "Error: frontend directory not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

# Navigate to frontend directory
Set-Location frontend

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Start the development server
Write-Host "Starting development server..." -ForegroundColor Green
npm run dev 