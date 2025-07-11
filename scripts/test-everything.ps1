# Comprehensive System Testing Script - ISM AI Platform
# Tests every component from user flow to AI algorithms

param(
    [Parameter(Position=0)]
    [ValidateSet("all", "database", "backend", "frontend", "ai", "logistics", "user-flow", "help")]
    [string]$Command = "all"
)

# Configuration
$ProjectRoot = Get-Location
$LogsDir = "logs"
$TestResultsDir = "test-results"

# Colors for output
$Red = [System.ConsoleColor]::Red
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue
$Cyan = [System.ConsoleColor]::Cyan

# Test results tracking
$TestResults = @{
    "Database" = @{ passed = 0; failed = 0; total = 0 }
    "Backend" = @{ passed = 0; failed = 0; total = 0 }
    "Frontend" = @{ passed = 0; failed = 0; total = 0 }
    "AI" = @{ passed = 0; failed = 0; total = 0 }
    "Logistics" = @{ passed = 0; failed = 0; total = 0 }
    "UserFlow" = @{ passed = 0; failed = 0; total = 0 }
}

function Write-TestLog {
    param([string]$Message, [System.ConsoleColor]$Color = $Green, [string]$Category = "General")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] TEST-$Category`: $Message"
    Write-Host $logMessage -ForegroundColor $Color
    
    # Also write to log file
    $logFile = "$LogsDir/comprehensive-testing.log"
    if (!(Test-Path $LogsDir)) {
        New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
    }
    Add-Content -Path $logFile -Value $logMessage
}

function Test-Database {
    Write-TestLog "Testing Database Connectivity and Schema..." $Blue "DATABASE"
    $TestResults.Database.total++
    
    try {
        # Test Supabase connection
        $response = Invoke-WebRequest -Uri "$env:SUPABASE_URL/rest/v1/" -Headers @{"apikey" = $env:SUPABASE_ANON_KEY} -TimeoutSec 10
        
        if ($response.StatusCode -eq 200) {
            Write-TestLog "✅ Supabase connection successful" $Green "DATABASE"
            $TestResults.Database.passed++
        } else {
            throw "Unexpected status code: $($response.StatusCode)"
        }
        
        # Test tables exist
        $tables = @('companies', 'materials', 'bulk_imports', 'users')
        foreach ($table in $tables) {
            $TestResults.Database.total++
            try {
                $tableResponse = Invoke-WebRequest -Uri "$env:SUPABASE_URL/rest/v1/$table" -Headers @{"apikey" = $env:SUPABASE_ANON_KEY} -TimeoutSec 10
                if ($tableResponse.StatusCode -eq 200) {
                    Write-TestLog "✅ Table '$table' exists and accessible" $Green "DATABASE"
                    $TestResults.Database.passed++
                } else {
                    throw "Table '$table' not accessible"
                }
            } catch {
                Write-TestLog "❌ Table '$table' test failed: $($_.Exception.Message)" $Red "DATABASE"
                $TestResults.Database.failed++
            }
        }
        
        # Test views exist
        $views = @('high_value_targets', 'symbiosis_opportunities')
        foreach ($view in $views) {
            $TestResults.Database.total++
            try {
                $viewResponse = Invoke-WebRequest -Uri "$env:SUPABASE_URL/rest/v1/$view" -Headers @{"apikey" = $env:SUPABASE_ANON_KEY} -TimeoutSec 10
                if ($viewResponse.StatusCode -eq 200) {
                    Write-TestLog "✅ View '$view' exists and accessible" $Green "DATABASE"
                    $TestResults.Database.passed++
                } else {
                    throw "View '$view' not accessible"
                }
            } catch {
                Write-TestLog "❌ View '$view' test failed: $($_.Exception.Message)" $Red "DATABASE"
                $TestResults.Database.failed++
            }
        }
        
    } catch {
        Write-TestLog "❌ Database connectivity failed: $($_.Exception.Message)" $Red "DATABASE"
        $TestResults.Database.failed++
    }
}

function Test-Backend {
    Write-TestLog "Testing Backend Services..." $Blue "BACKEND"
    
    $endpoints = @(
        @{Name = "Health Check"; Url = "http://localhost:3001/api/health"},
        @{Name = "Monitoring Metrics"; Url = "http://localhost:3001/api/monitoring/metrics"},
        @{Name = "System Health"; Url = "http://localhost:3001/api/monitoring/health"},
        @{Name = "AI Listings Stats"; Url = "http://localhost:3001/api/ai/listings-stats"},
        @{Name = "High Value Targets"; Url = "http://localhost:3001/api/real-data/high-value-targets"},
        @{Name = "Market Analysis"; Url = "http://localhost:3001/api/real-data/market-analysis"}
    )
    
    foreach ($endpoint in $endpoints) {
        $TestResults.Backend.total++
        try {
            $response = Invoke-WebRequest -Uri $endpoint.Url -TimeoutSec 10 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-TestLog "✅ $($endpoint.Name) endpoint working" $Green "BACKEND"
                $TestResults.Backend.passed++
            } else {
                throw "Status code: $($response.StatusCode)"
            }
        } catch {
            Write-TestLog "❌ $($endpoint.Name) endpoint failed: $($_.Exception.Message)" $Red "BACKEND"
            $TestResults.Backend.failed++
        }
    }
}

function Test-AI {
    Write-TestLog "Testing AI Services..." $Blue "AI"
    
    # Test AI services health
    $aiServices = @(
        @{Name = "GNN Reasoning"; Url = "http://localhost:5001/health"},
        @{Name = "Federated Learning"; Url = "http://localhost:5002/health"},
        @{Name = "Multi-Hop Symbiosis"; Url = "http://localhost:5003/health"},
        @{Name = "Advanced Analytics"; Url = "http://localhost:5004/health"}
    )
    
    foreach ($service in $aiServices) {
        $TestResults.AI.total++
        try {
            $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 10 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-TestLog "✅ $($service.Name) service healthy" $Green "AI"
                $TestResults.AI.passed++
            } else {
                throw "Status code: $($response.StatusCode)"
            }
        } catch {
            Write-TestLog "❌ $($service.Name) service failed: $($_.Exception.Message)" $Red "AI"
            $TestResults.AI.failed++
        }
    }
    
    # Test AI listings generation
    $TestResults.AI.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/ai/generate-all-listings" -Method POST -TimeoutSec 30 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-TestLog "✅ AI listings generation working" $Green "AI"
                Write-TestLog "   Generated $($result.summary.total_listings_generated) listings" $Cyan "AI"
                Write-TestLog "   Total potential value: $($result.summary.total_potential_value)" $Cyan "AI"
                $TestResults.AI.passed++
            } else {
                throw "AI generation returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ AI listings generation failed: $($_.Exception.Message)" $Red "AI"
        $TestResults.AI.failed++
    }
}

function Test-Logistics {
    Write-TestLog "Testing Logistics Integration..." $Blue "LOGISTICS"
    
    # Test Freightos API integration
    $TestResults.Logistics.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/real-data/process-company" -Method POST -Body (@{
            companyData = @{
                name = "Test Logistics Company"
                industry = "manufacturing"
                location = "Dubai"
                waste_streams = @(@{name = "Steel scrap"; quantity = 1000; unit = "kg"})
            }
        } | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-TestLog "✅ Freightos logistics integration working" $Green "LOGISTICS"
                $TestResults.Logistics.passed++
            } else {
                throw "Logistics processing returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ Freightos logistics integration failed: $($_.Exception.Message)" $Red "LOGISTICS"
        $TestResults.Logistics.failed++
    }
    
    # Test logistics cost calculations
    $TestResults.Logistics.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/real-data/symbiosis-network" -TimeoutSec 10 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-TestLog "✅ Symbiosis network analysis working" $Green "LOGISTICS"
            $TestResults.Logistics.passed++
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ Symbiosis network analysis failed: $($_.Exception.Message)" $Red "LOGISTICS"
        $TestResults.Logistics.failed++
    }
}

function Test-Frontend {
    Write-TestLog "Testing Frontend Components..." $Blue "FRONTEND"
    
    # Test frontend build
    $TestResults.Frontend.total++
    try {
        Push-Location frontend
        npm run build 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-TestLog "✅ Frontend build successful" $Green "FRONTEND"
            $TestResults.Frontend.passed++
        } else {
            throw "Build failed with exit code: $LASTEXITCODE"
        }
        Pop-Location
    } catch {
        Write-TestLog "❌ Frontend build failed: $($_.Exception.Message)" $Red "FRONTEND"
        $TestResults.Frontend.failed++
    }
    
    # Test frontend dev server
    $TestResults.Frontend.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 10 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-TestLog "✅ Frontend dev server running" $Green "FRONTEND"
            $TestResults.Frontend.passed++
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ Frontend dev server not accessible: $($_.Exception.Message)" $Red "FRONTEND"
        $TestResults.Frontend.failed++
    }
}

function Test-UserFlow {
    Write-TestLog "Testing User Flow..." $Blue "USERFLOW"
    
    # Test complete user journey
    $userFlowSteps = @(
        @{Name = "User Registration"; Url = "http://localhost:3001/api/auth/register"},
        @{Name = "Company Profile Creation"; Url = "http://localhost:3001/api/companies"},
        @{Name = "Material Listing Creation"; Url = "http://localhost:3001/api/materials"},
        @{Name = "AI Analysis"; Url = "http://localhost:3001/api/ai/analyze"},
        @{Name = "Symbiosis Matching"; Url = "http://localhost:3001/api/matching/symbiosis"},
        @{Name = "Logistics Calculation"; Url = "http://localhost:3001/api/logistics/calculate"}
    )
    
    foreach ($step in $userFlowSteps) {
        $TestResults.UserFlow.total++
        try {
            $response = Invoke-WebRequest -Uri $step.Url -TimeoutSec 10 -ErrorAction Stop
            if ($response.StatusCode -eq 200 -or $response.StatusCode -eq 201) {
                Write-TestLog "✅ $($step.Name) endpoint accessible" $Green "USERFLOW"
                $TestResults.UserFlow.passed++
            } else {
                throw "Status code: $($response.StatusCode)"
            }
        } catch {
            Write-TestLog "❌ $($step.Name) endpoint failed: $($_.Exception.Message)" $Red "USERFLOW"
            $TestResults.UserFlow.failed++
        }
    }
}

function Show-TestSummary {
    Write-TestLog "=== COMPREHENSIVE TESTING SUMMARY ===" $Cyan "SUMMARY"
    
    $totalPassed = 0
    $totalFailed = 0
    $totalTests = 0
    
    foreach ($category in $TestResults.Keys) {
        $categoryResults = $TestResults[$category]
        $totalTests += $categoryResults.total
        $totalPassed += $categoryResults.passed
        $totalFailed += $categoryResults.failed
        
        $passRate = if ($categoryResults.total -gt 0) { ($categoryResults.passed / $categoryResults.total * 100) } else { 0 }
        $color = if ($passRate -ge 80) { $Green } elseif ($passRate -ge 60) { $Yellow } else { $Red }
        
        Write-TestLog "$category`: $($categoryResults.passed)/$($categoryResults.total) passed ($([math]::Round($passRate, 1))%)" $color "SUMMARY"
    }
    
    $overallPassRate = if ($totalTests -gt 0) { ($totalPassed / $totalTests * 100) } else { 0 }
    $overallColor = if ($overallPassRate -ge 80) { $Green } elseif ($overallPassRate -ge 60) { $Yellow } else { $Red }
    
    Write-TestLog "=== OVERALL RESULTS ===" $Cyan "SUMMARY"
    Write-TestLog "Total Tests: $totalTests" $Cyan "SUMMARY"
    Write-TestLog "Passed: $totalPassed" $Green "SUMMARY"
    Write-TestLog "Failed: $totalFailed" $Red "SUMMARY"
    Write-TestLog "Pass Rate: $([math]::Round($overallPassRate, 1))%" $overallColor "SUMMARY"
    
    # Save detailed results
    if (!(Test-Path $TestResultsDir)) {
        New-Item -ItemType Directory -Path $TestResultsDir -Force | Out-Null
    }
    
    $resultsFile = "$TestResultsDir/test-results-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
    $TestResults | ConvertTo-Json -Depth 3 | Set-Content -Path $resultsFile
    
    Write-TestLog "Detailed results saved to: $resultsFile" $Cyan "SUMMARY"
    
    # Final recommendation
    if ($overallPassRate -ge 90) {
        Write-TestLog "🎉 SYSTEM IS READY FOR PRODUCTION!" $Green "SUMMARY"
        Write-TestLog "All critical components are working. Ready to receive your 50 company profiles!" $Green "SUMMARY"
    } elseif ($overallPassRate -ge 70) {
        Write-TestLog "⚠️ SYSTEM MOSTLY READY - Some issues need attention" $Yellow "SUMMARY"
        Write-TestLog "Review failed tests above and fix critical issues before production." $Yellow "SUMMARY"
    } else {
        Write-TestLog "❌ SYSTEM NOT READY - Significant issues detected" $Red "SUMMARY"
        Write-TestLog "Fix all failed tests before proceeding with production deployment." $Red "SUMMARY"
    }
}

function Start-ComprehensiveTesting {
    Write-TestLog "Starting Comprehensive System Testing..." $Blue "MAIN"
    Write-TestLog "Testing every component from user flow to AI algorithms" $Blue "MAIN"
    
    switch ($Command) {
        "database" {
            Test-Database
        }
        "backend" {
            Test-Backend
        }
        "frontend" {
            Test-Frontend
        }
        "ai" {
            Test-AI
        }
        "logistics" {
            Test-Logistics
        }
        "user-flow" {
            Test-UserFlow
        }
        "all" {
            Test-Database
            Test-Backend
            Test-Frontend
            Test-AI
            Test-Logistics
            Test-UserFlow
        }
        "help" {
            Write-Host "ISM AI Platform - Comprehensive Testing Script"
            Write-Host ""
            Write-Host "Usage: .\test-everything.ps1 [command]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  all        Test all components (default)"
            Write-Host "  database   Test database connectivity and schema"
            Write-Host "  backend    Test backend services and endpoints"
            Write-Host "  frontend   Test frontend build and dev server"
            Write-Host "  ai         Test AI services and algorithms"
            Write-Host "  logistics  Test Freightos integration and logistics"
            Write-Host "  user-flow  Test complete user journey"
            Write-Host "  help       Show this help message"
            Write-Host ""
            Write-Host "This script tests every component of your ISM AI platform"
            Write-Host "to ensure it's ready for your 50 real company profiles."
        }
        default {
            Test-Database
            Test-Backend
            Test-Frontend
            Test-AI
            Test-Logistics
            Test-UserFlow
        }
    }
    
    Show-TestSummary
}

# Run comprehensive testing
Start-ComprehensiveTesting 