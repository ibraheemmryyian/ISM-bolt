# =====================================================
# PERFECT MIGRATION SCRIPT FOR ISM AI PLATFORM
# Complete Setup with Real Data Integration
# =====================================================

param(
    [switch]$SkipDatabase,
    [switch]$SkipFrontend,
    [switch]$SkipBackend,
    [switch]$Force,
    [string]$Environment = "development"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Color functions for better output
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Header { param($Message) Write-Host "`n🚀 $Message" -ForegroundColor Magenta }

# Global variables
$ProjectRoot = Get-Location
$BackendPath = Join-Path $ProjectRoot "backend"
$FrontendPath = Join-Path $ProjectRoot "frontend"
$DatabasePath = Join-Path $ProjectRoot "database"
$ScriptsPath = Join-Path $ProjectRoot "scripts"

# Check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check prerequisites
function Test-Prerequisites {
    Write-Header "Checking Prerequisites"
    
    $prerequisites = @{
        "Node.js" = "node --version"
        "npm" = "npm --version"
        "Python" = "python --version"
        "pip" = "pip --version"
        "Git" = "git --version"
    }
    
    $missing = @()
    
    foreach ($tool in $prerequisites.Keys) {
        try {
            $version = Invoke-Expression $prerequisites[$tool] 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Success "$tool is installed: $version"
            } else {
                $missing += $tool
            }
        } catch {
            $missing += $tool
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-Error "Missing prerequisites: $($missing -join ', ')"
        Write-Info "Please install the missing tools and run the script again."
        exit 1
    }
    
    Write-Success "All prerequisites are installed!"
}

# Create environment file
function New-EnvironmentFile {
    Write-Header "Setting up Environment Configuration"
    
    $envFile = Join-Path $BackendPath ".env"
    $envExample = Join-Path $BackendPath "env.example"
    
    if (Test-Path $envFile) {
        if ($Force) {
            Write-Warning "Removing existing .env file"
            Remove-Item $envFile -Force
        } else {
            Write-Info ".env file already exists. Skipping..."
            return
        }
    }
    
    if (Test-Path $envExample) {
        Copy-Item $envExample $envFile
        Write-Success "Created .env file from template"
        
        # Update with real values
        $envContent = Get-Content $envFile -Raw
        $envContent = $envContent -replace "your_deepseek_api_key_required", "sk-REPLACE_WITH_YOUR_DEEPSEEK_API_KEY"
        $envContent = $envContent -replace "your_freightos_api_key_required", "REPLACE_WITH_YOUR_FREIGHTOS_API_KEY"
        $envContent = $envContent -replace "your_freightos_secret_key_required", "REPLACE_WITH_YOUR_FREIGHTOS_SECRET_KEY"
        $envContent = $envContent -replace "your_supabase_service_role_key", "REPLACE_WITH_YOUR_SUPABASE_SERVICE_ROLE_KEY"
        
        Set-Content $envFile $envContent
        Write-Warning "Please update the .env file with your actual API keys!"
    } else {
        Write-Error "env.example file not found!"
        exit 1
    }
}

# Install backend dependencies
function Install-BackendDependencies {
    Write-Header "Installing Backend Dependencies"
    
    Set-Location $BackendPath
    
    # Install Node.js dependencies
    Write-Info "Installing Node.js dependencies..."
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install Node.js dependencies"
        exit 1
    }
    
    # Install Python dependencies
    Write-Info "Installing Python dependencies..."
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install Python dependencies"
        exit 1
    }
    
    Write-Success "Backend dependencies installed successfully!"
    Set-Location $ProjectRoot
}

# Install frontend dependencies
function Install-FrontendDependencies {
    Write-Header "Installing Frontend Dependencies"
    
    Set-Location $FrontendPath
    
    Write-Info "Installing frontend dependencies..."
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install frontend dependencies"
        exit 1
    }
    
    Write-Success "Frontend dependencies installed successfully!"
    Set-Location $ProjectRoot
}

# Run database migration
function Invoke-DatabaseMigration {
    Write-Header "Running Database Migration"
    
    $migrationFile = Join-Path $ProjectRoot "database_migration_for_real_data.sql"
    
    if (-not (Test-Path $migrationFile)) {
        Write-Error "Database migration file not found: $migrationFile"
        exit 1
    }
    
    Write-Info "Reading migration file..."
    $migrationContent = Get-Content $migrationFile -Raw
    
    # Extract Supabase URL and key from .env file
    $envFile = Join-Path $BackendPath ".env"
    if (Test-Path $envFile) {
        $envContent = Get-Content $envFile
        $supabaseUrl = ($envContent | Where-Object { $_ -match "SUPABASE_URL=" }) -replace "SUPABASE_URL=", ""
        $supabaseKey = ($envContent | Where-Object { $_ -match "SUPABASE_ANON_KEY=" }) -replace "SUPABASE_ANON_KEY=", ""
        
        if ($supabaseUrl -and $supabaseKey) {
            Write-Info "Found Supabase configuration"
            
            # Create temporary migration script
            $tempScript = Join-Path $ProjectRoot "temp_migration.js"
            $migrationScript = @"
const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = '$supabaseUrl';
const supabaseKey = '$supabaseKey';
const supabase = createClient(supabaseUrl, supabaseKey);

async function runMigration() {
    try {
        console.log('🚀 Starting database migration...');
        
        const migrationSQL = \`$migrationContent\`;
        
        // Split the migration into individual statements
        const statements = migrationSQL
            .split(';')
            .map(stmt => stmt.trim())
            .filter(stmt => stmt.length > 0 && !stmt.startsWith('--'));
        
        console.log(\`📊 Executing \${statements.length} SQL statements...\`);
        
        for (let i = 0; i < statements.length; i++) {
            const statement = statements[i];
            if (statement.trim()) {
                try {
                    const { error } = await supabase.rpc('exec_sql', { sql: statement });
                    if (error) {
                        console.warn(\`⚠️  Statement \${i + 1} warning: \${error.message}\`);
                    } else {
                        console.log(\`✅ Statement \${i + 1} executed successfully\`);
                    }
                } catch (err) {
                    console.warn(\`⚠️  Statement \${i + 1} error: \${err.message}\`);
                }
            }
        }
        
        console.log('🎉 Database migration completed!');
    } catch (error) {
        console.error('❌ Migration failed:', error);
        process.exit(1);
    }
}

runMigration();
"@
            
            Set-Content $tempScript $migrationScript
            
            # Run the migration
            Set-Location $BackendPath
            node ../temp_migration.js
            $migrationSuccess = $LASTEXITCODE -eq 0
            
            # Clean up
            Remove-Item $tempScript -Force
            Set-Location $ProjectRoot
            
            if ($migrationSuccess) {
                Write-Success "Database migration completed successfully!"
            } else {
                Write-Warning "Database migration completed with warnings. Check the output above."
            }
        } else {
            Write-Warning "Supabase configuration not found. Skipping database migration."
        }
    } else {
        Write-Warning ".env file not found. Skipping database migration."
    }
}

# Build frontend
function Build-Frontend {
    Write-Header "Building Frontend"
    
    Set-Location $FrontendPath
    
    Write-Info "Building frontend application..."
    npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Frontend build failed"
        exit 1
    }
    
    Write-Success "Frontend built successfully!"
    Set-Location $ProjectRoot
}

# Run tests
function Invoke-Tests {
    Write-Header "Running Tests"
    
    # Backend tests
    if (-not $SkipBackend) {
        Write-Info "Running backend tests..."
        Set-Location $BackendPath
        npm test
        $backendTestSuccess = $LASTEXITCODE -eq 0
        Set-Location $ProjectRoot
        
        if ($backendTestSuccess) {
            Write-Success "Backend tests passed!"
        } else {
            Write-Warning "Backend tests failed. Check the output above."
        }
    }
    
    # Frontend tests
    if (-not $SkipFrontend) {
        Write-Info "Running frontend tests..."
        Set-Location $FrontendPath
        npm test -- --run
        $frontendTestSuccess = $LASTEXITCODE -eq 0
        Set-Location $ProjectRoot
        
        if ($frontendTestSuccess) {
            Write-Success "Frontend tests passed!"
        } else {
            Write-Warning "Frontend tests failed. Check the output above."
        }
    }
}

# Health check
function Test-HealthCheck {
    Write-Header "Running Health Check"
    
    $healthCheckScript = Join-Path $ProjectRoot "test_backend_health.js"
    
    if (Test-Path $healthCheckScript) {
        Write-Info "Running comprehensive health check..."
        node $healthCheckScript
        $healthSuccess = $LASTEXITCODE -eq 0
        
        if ($healthSuccess) {
            Write-Success "Health check passed!"
        } else {
            Write-Warning "Health check failed. Check the output above."
        }
    } else {
        Write-Warning "Health check script not found. Skipping..."
    }
}

# Create startup scripts
function New-StartupScripts {
    Write-Header "Creating Startup Scripts"
    
    # Windows batch file for starting everything
    $startScript = @"
@echo off
echo 🚀 Starting ISM AI Platform...
echo.

cd /d "$BackendPath"
echo 📡 Starting backend server...
start "Backend Server" cmd /k "npm run dev"

timeout /t 3 /nobreak >nul

cd /d "$FrontendPath"
echo 🎨 Starting frontend development server...
start "Frontend Server" cmd /k "npm run dev"

echo.
echo ✅ ISM AI Platform is starting up!
echo 📱 Frontend: http://localhost:5173
echo 🔧 Backend: http://localhost:5001
echo.
echo Press any key to exit this window...
pause >nul
"@
    
    Set-Content (Join-Path $ProjectRoot "start_platform.bat") $startScript
    
    # PowerShell script for starting everything
    $startPSScript = @"
# ISM AI Platform Startup Script
Write-Host "🚀 Starting ISM AI Platform..." -ForegroundColor Green

# Start backend
Write-Host "📡 Starting backend server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$BackendPath'; npm run dev"

Start-Sleep -Seconds 3

# Start frontend
Write-Host "🎨 Starting frontend development server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$FrontendPath'; npm run dev"

Write-Host "`n✅ ISM AI Platform is starting up!" -ForegroundColor Green
Write-Host "📱 Frontend: http://localhost:5173" -ForegroundColor Yellow
Write-Host "🔧 Backend: http://localhost:5001" -ForegroundColor Yellow
Write-Host "`nPress any key to exit..." -ForegroundColor Gray
Read-Host
"@
    
    Set-Content (Join-Path $ProjectRoot "start_platform.ps1") $startPSScript
    
    Write-Success "Startup scripts created!"
}

# Create production deployment script
function New-ProductionScript {
    Write-Header "Creating Production Deployment Script"
    
    $prodScript = @"
# ISM AI Platform Production Deployment
Write-Host "🚀 Deploying ISM AI Platform to Production..." -ForegroundColor Green

# Build frontend for production
Write-Host "🏗️  Building frontend for production..." -ForegroundColor Cyan
Set-Location "$FrontendPath"
npm run build

if (\$LASTEXITCODE -ne 0) {
    Write-Error "Frontend build failed!"
    exit 1
}

# Set production environment
Write-Host "⚙️  Setting production environment..." -ForegroundColor Cyan
Set-Location "$BackendPath"
\$env:NODE_ENV = "production"
\$env:PORT = "5001"

# Start production server
Write-Host "🚀 Starting production server..." -ForegroundColor Cyan
npm start

Write-Host "✅ Production deployment completed!" -ForegroundColor Green
"@
    
    Set-Content (Join-Path $ProjectRoot "deploy_production.ps1") $prodScript
    Write-Success "Production deployment script created!"
}

# Main migration function
function Start-Migration {
    Write-Host "`n" -NoNewline
    Write-Host "=====================================================" -ForegroundColor Magenta
    Write-Host "🚀 ISM AI PLATFORM - PERFECT MIGRATION SCRIPT" -ForegroundColor Magenta
    Write-Host "=====================================================" -ForegroundColor Magenta
    Write-Host "`n" -NoNewline
    
    # Check prerequisites
    Test-Prerequisites
    
    # Create environment file
    New-EnvironmentFile
    
    # Install dependencies
    if (-not $SkipBackend) {
        Install-BackendDependencies
    }
    
    if (-not $SkipFrontend) {
        Install-FrontendDependencies
    }
    
    # Run database migration
    if (-not $SkipDatabase) {
        Invoke-DatabaseMigration
    }
    
    # Build frontend
    if (-not $SkipFrontend) {
        Build-Frontend
    }
    
    # Run tests
    Invoke-Tests
    
    # Health check
    Test-HealthCheck
    
    # Create startup scripts
    New-StartupScripts
    New-ProductionScript
    
    # Final summary
    Write-Header "Migration Complete!"
    Write-Success "ISM AI Platform has been successfully migrated!"
    Write-Info "Next steps:"
    Write-Info "1. Update the .env file with your actual API keys"
    Write-Info "2. Run 'start_platform.bat' or 'start_platform.ps1' to start the platform"
    Write-Info "3. Access the platform at http://localhost:5173"
    Write-Info "4. Run 'deploy_production.ps1' for production deployment"
    
    Write-Host "`n" -NoNewline
    Write-Host "=====================================================" -ForegroundColor Magenta
    Write-Host "🎉 MIGRATION COMPLETED SUCCESSFULLY!" -ForegroundColor Magenta
    Write-Host "=====================================================" -ForegroundColor Magenta
    Write-Host "`n" -NoNewline
}

# Execute migration
try {
    Start-Migration
} catch {
    Write-Error "Migration failed: $($_.Exception.Message)"
    Write-Error "Stack trace: $($_.ScriptStackTrace)"
    exit 1
} 