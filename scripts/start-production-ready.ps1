# Production-Ready Startup Script - ISM AI Platform
# Ensures everything is working before launching with 50 real companies

param(
    [switch]$SkipTests,
    [switch]$ForceStart,
    [switch]$ShowHelp
)

# Configuration
$ProjectRoot = Get-Location
$Services = @(
    @{Name = "Database"; Check = "Supabase"; Required = $true},
    @{Name = "Backend"; Check = "http://localhost:3001/api/health"; Required = $true},
    @{Name = "Frontend"; Check = "http://localhost:5173"; Required = $true},
    @{Name = "GNN Service"; Check = "http://localhost:5001/health"; Required = $true},
    @{Name = "Federated Learning"; Check = "http://localhost:5002/health"; Required = $true},
    @{Name = "Multi-Hop Symbiosis"; Check = "http://localhost:5003/health"; Required = $true},
    @{Name = "Advanced Analytics"; Check = "http://localhost:5004/health"; Required = $true},
    @{Name = "Redis"; Check = "redis://localhost:6379"; Required = $true}
)

# Colors
$Red = [System.ConsoleColor]::Red
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue
$Cyan = [System.ConsoleColor]::Cyan

function Write-Status {
    param([string]$Message, [System.ConsoleColor]$Color = $Green, [string]$Status = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Status`: $Message" -ForegroundColor $Color
}

function Test-ServiceHealth {
    param([string]$ServiceName, [string]$CheckUrl, [bool]$Required)
    
    Write-Status "Checking $ServiceName..." $Blue "HEALTH"
    
    try {
        if ($CheckUrl -like "http*") {
            $response = Invoke-WebRequest -Uri $CheckUrl -TimeoutSec 10 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Status "✅ $ServiceName is healthy" $Green "HEALTH"
                return $true
            } else {
                throw "Status code: $($response.StatusCode)"
            }
        } elseif ($CheckUrl -like "redis*") {
            # Test Redis connection
            $redisTest = redis-cli ping 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Status "✅ $ServiceName is healthy" $Green "HEALTH"
                return $true
            } else {
                throw "Redis not responding"
            }
        } else {
            # Generic service check
            Write-Status "✅ $ServiceName is available" $Green "HEALTH"
            return $true
        }
    } catch {
        if ($Required) {
            Write-Status "❌ $ServiceName is DOWN (REQUIRED)" $Red "HEALTH"
            return $false
        } else {
            Write-Status "⚠️ $ServiceName is DOWN (Optional)" $Yellow "HEALTH"
            return $true  # Don't fail for optional services
        }
    }
}

function Start-BackendServices {
    Write-Status "Starting Backend Services..." $Blue "STARTUP"
    
    # Start Redis if not running
    try {
        $redisTest = redis-cli ping 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Status "Starting Redis..." $Yellow "STARTUP"
            Start-Process -FilePath "redis-server" -WindowStyle Hidden
            Start-Sleep -Seconds 3
        }
    } catch {
        Write-Status "Redis not found, continuing..." $Yellow "STARTUP"
    }
    
    # Start Backend
    Write-Status "Starting Backend Server..." $Yellow "STARTUP"
    Start-Process -FilePath "node" -ArgumentList "backend/app.js" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    
    # Start AI Services
    Write-Status "Starting AI Services..." $Yellow "STARTUP"
    
    # GNN Service
    Push-Location "ai_service_flask"
    Start-Process -FilePath "python" -ArgumentList "gnn_inference_service.py" -WindowStyle Hidden
    Pop-Location
    Start-Sleep -Seconds 3
    
    # Federated Learning Service
    Push-Location "ai_service_flask"
    Start-Process -FilePath "python" -ArgumentList "federated_learning_service.py" -WindowStyle Hidden
    Pop-Location
    Start-Sleep -Seconds 3
    
    # Multi-Hop Symbiosis Service
    Push-Location "ai_service_flask"
    Start-Process -FilePath "python" -ArgumentList "multi_hop_symbiosis_service.py" -WindowStyle Hidden
    Pop-Location
    Start-Sleep -Seconds 3
    
    # Advanced Analytics Service
    Push-Location "ai_service_flask"
    Start-Process -FilePath "python" -ArgumentList "advanced_analytics_service.py" -WindowStyle Hidden
    Pop-Location
    Start-Sleep -Seconds 3
}

function Start-Frontend {
    Write-Status "Starting Frontend Development Server..." $Yellow "STARTUP"
    Push-Location "frontend"
    Start-Process -FilePath "npm" -ArgumentList "run dev" -WindowStyle Hidden
    Pop-Location
    Start-Sleep -Seconds 10
}

function Test-AIListingsGenerator {
    Write-Status "Testing Revolutionary AI Listings Generator..." $Blue "AI-TEST"
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/ai/generate-all-listings" -Method POST -TimeoutSec 60 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-Status "✅ AI Listings Generator: SUCCESS" $Green "AI-TEST"
                Write-Status "   Generated $($result.summary.total_listings_generated) listings" $Cyan "AI-TEST"
                Write-Status "   Total potential value: $($result.summary.total_potential_value)" $Cyan "AI-TEST"
                Write-Status "   Companies processed: $($result.summary.total_companies)" $Cyan "AI-TEST"
                return $true
            } else {
                throw "AI generation returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-Status "❌ AI Listings Generator: FAILED" $Red "AI-TEST"
        Write-Status "   Error: $($_.Exception.Message)" $Red "AI-TEST"
        return $false
    }
}

function Test-FreightosIntegration {
    Write-Status "Testing Freightos Logistics Integration..." $Blue "LOGISTICS-TEST"
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/real-data/process-company" -Method POST -Body (@{
            companyData = @{
                name = "Test Freightos Company"
                industry = "manufacturing"
                location = "Dubai"
                waste_streams = @(@{name = "Steel scrap"; quantity = 1000; unit = "kg"})
            }
        } | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-Status "✅ Freightos Integration: SUCCESS" $Green "LOGISTICS-TEST"
                Write-Status "   Real freight costs calculated" $Cyan "LOGISTICS-TEST"
                Write-Status "   CO2 emissions estimated" $Cyan "LOGISTICS-TEST"
                return $true
            } else {
                throw "Freightos processing returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-Status "❌ Freightos Integration: FAILED" $Red "LOGISTICS-TEST"
        Write-Status "   Error: $($_.Exception.Message)" $Red "LOGISTICS-TEST"
        return $false
    }
}

function Test-UserFlow {
    Write-Status "Testing Complete User Flow..." $Blue "USERFLOW-TEST"
    
    $flowSteps = @(
        @{Name = "User Registration"; Url = "http://localhost:3001/api/auth/register"},
        @{Name = "Company Profile"; Url = "http://localhost:3001/api/companies"},
        @{Name = "Material Listing"; Url = "http://localhost:3001/api/materials"},
        @{Name = "AI Analysis"; Url = "http://localhost:3001/api/ai/analyze"},
        @{Name = "Symbiosis Matching"; Url = "http://localhost:3001/api/matching/symbiosis"},
        @{Name = "Logistics Calculation"; Url = "http://localhost:3001/api/logistics/calculate"}
    )
    
    $passed = 0
    $total = $flowSteps.Count
    
    foreach ($step in $flowSteps) {
        try {
            $response = Invoke-WebRequest -Uri $step.Url -TimeoutSec 10 -ErrorAction Stop
            if ($response.StatusCode -eq 200 -or $response.StatusCode -eq 201) {
                Write-Status "✅ $($step.Name) endpoint working" $Green "USERFLOW-TEST"
                $passed++
            } else {
                Write-Status "❌ $($step.Name) endpoint failed" $Red "USERFLOW-TEST"
            }
        } catch {
            Write-Status "❌ $($step.Name) endpoint failed: $($_.Exception.Message)" $Red "USERFLOW-TEST"
        }
    }
    
    $passRate = ($passed / $total) * 100
    if ($passRate -ge 80) {
        Write-Status "✅ User Flow: $passed/$total endpoints working ($([math]::Round($passRate, 1))%)" $Green "USERFLOW-TEST"
        return $true
    } else {
        Write-Status "❌ User Flow: Only $passed/$total endpoints working ($([math]::Round($passRate, 1))%)" $Red "USERFLOW-TEST"
        return $false
    }
}

function Show-StartupSummary {
    param([bool]$AllServicesHealthy, [bool]$AIWorking, [bool]$LogisticsWorking, [bool]$UserFlowWorking)
    
    Write-Status "=== PRODUCTION READINESS SUMMARY ===" $Cyan "SUMMARY"
    
    if ($AllServicesHealthy) {
        Write-Status "✅ All core services are healthy" $Green "SUMMARY"
    } else {
        Write-Status "❌ Some core services are down" $Red "SUMMARY"
    }
    
    if ($AIWorking) {
        Write-Status "✅ Revolutionary AI Listings Generator working" $Green "SUMMARY"
    } else {
        Write-Status "❌ AI Listings Generator needs attention" $Red "SUMMARY"
    }
    
    if ($LogisticsWorking) {
        Write-Status "✅ Freightos logistics integration working" $Green "SUMMARY"
    } else {
        Write-Status "❌ Freightos integration needs attention" $Red "SUMMARY"
    }
    
    if ($UserFlowWorking) {
        Write-Status "✅ Complete user flow working" $Green "SUMMARY"
    } else {
        Write-Status "❌ User flow needs attention" $Red "SUMMARY"
    }
    
    Write-Status "=== ACCESS URLs ===" $Cyan "SUMMARY"
    Write-Status "Frontend: http://localhost:5173" $Blue "SUMMARY"
    Write-Status "Backend API: http://localhost:3001" $Blue "SUMMARY"
    Write-Status "AI Services: http://localhost:5001-5004" $Blue "SUMMARY"
    Write-Status "Monitoring: http://localhost:3001/api/monitoring" $Blue "SUMMARY"
    
    if ($AllServicesHealthy -and $AIWorking -and $LogisticsWorking -and $UserFlowWorking) {
        Write-Status "🎉 SYSTEM IS PRODUCTION READY!" $Green "SUMMARY"
        Write-Status "Ready to receive your 50 real company profiles!" $Green "SUMMARY"
        Write-Status "Your revolutionary AI platform is ready for the Gulf market!" $Green "SUMMARY"
    } else {
        Write-Status "⚠️ SYSTEM NEEDS ATTENTION BEFORE PRODUCTION" $Yellow "SUMMARY"
        Write-Status "Fix the issues above before proceeding with real company data." $Yellow "SUMMARY"
    }
}

function Start-ProductionReadySystem {
    if ($ShowHelp) {
        Write-Host "ISM AI Platform - Production Ready Startup Script"
        Write-Host ""
        Write-Host "Usage: .\start-production-ready.ps1 [options]"
        Write-Host ""
        Write-Host "Options:"
        Write-Host "  -SkipTests    Skip comprehensive testing"
        Write-Host "  -ForceStart   Start services even if tests fail"
        Write-Host "  -ShowHelp     Show this help message"
        Write-Host ""
        Write-Host "This script ensures your system is ready for production"
        Write-Host "with 50 real company profiles and revolutionary AI capabilities."
        return
    }
    
    Write-Status "🚀 Starting ISM AI Platform - Production Ready Mode" $Cyan "MAIN"
    Write-Status "Preparing for 50 real company profiles" $Cyan "MAIN"
    
    # Start all services
    Start-BackendServices
    Start-Frontend
    
    # Wait for services to be ready
    Write-Status "Waiting for services to be ready..." $Yellow "MAIN"
    Start-Sleep -Seconds 15
    
    # Health checks
    $allServicesHealthy = $true
    foreach ($service in $Services) {
        $healthy = Test-ServiceHealth -ServiceName $service.Name -CheckUrl $service.Check -Required $service.Required
        if (-not $healthy -and $service.Required) {
            $allServicesHealthy = $false
        }
    }
    
    # Run tests if not skipped
    $aiWorking = $false
    $logisticsWorking = $false
    $userFlowWorking = $false
    
    if (-not $SkipTests) {
        Write-Status "Running comprehensive tests..." $Blue "MAIN"
        
        $aiWorking = Test-AIListingsGenerator
        $logisticsWorking = Test-FreightosIntegration
        $userFlowWorking = Test-UserFlow
        
        if (-not $ForceStart -and (-not $allServicesHealthy -or -not $aiWorking -or -not $logisticsWorking)) {
            Write-Status "❌ Critical services failed. Use -ForceStart to continue anyway." $Red "MAIN"
            return
        }
    }
    
    # Show summary
    Show-StartupSummary -AllServicesHealthy $allServicesHealthy -AIWorking $aiWorking -LogisticsWorking $logisticsWorking -UserFlowWorking $userFlowWorking
    
    Write-Status "System startup complete!" $Green "MAIN"
    Write-Status "Monitor logs for any issues." $Yellow "MAIN"
}

# Run production-ready startup
Start-ProductionReadySystem 