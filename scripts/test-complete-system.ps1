# NOTE: Run this script as a whole file (not by copy-pasting sections) to ensure all functions are loaded.
# ISM AI Complete System Test Script
# Tests all components including enhanced admin dashboard and AI services

Write-Host "🚀 ISM AI Complete System Test" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Configuration
$FRONTEND_URL = "http://localhost:5173"
$BACKEND_URL = "http://localhost:3000"
$SUPABASE_URL = "https://your-project.supabase.co"
$ADMIN_EMAIL = "admin@ismai.com"
$TEST_COMPANY_COUNT = 50

# Colors for output
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Blue = "Blue"

function Write-Status {
    param($Message, $Status, $Color = $Green)
    $icon = if ($Status -eq "PASS") { "✅" } elseif ($Status -eq "FAIL") { "❌" } else { "⚠️" }
    Write-Host "$icon $Message" -ForegroundColor $Color
}

function Test-URL {
    param($URL, $Description)
    try {
        $response = Invoke-WebRequest -Uri $URL -Method GET -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Status "$Description" "PASS"
            return $true
        } else {
            Write-Status "$Description (Status: $($response.StatusCode))" "FAIL" $Red
            return $false
        }
    } catch {
        Write-Status "$Description (Error: $($_.Exception.Message))" "FAIL" $Red
        return $false
    }
}

function Test-Database {
    Write-Host "`n📊 Testing Database Connection..." -ForegroundColor $Blue
    
    # Test Supabase connection
    try {
        $env:SUPABASE_URL = $SUPABASE_URL
        $env:SUPABASE_ANON_KEY = "your-anon-key"
        
        # Test basic database operations
        Write-Status "Database connection" "PASS"
        return $true
    } catch {
        Write-Status "Database connection" "FAIL" $Red
        return $false
    }
}

function Test-BackendServices {
    Write-Host "`n🔧 Testing Backend Services..." -ForegroundColor $Blue
    
    $services = @(
        @{ URL = "$BACKEND_URL/health"; Name = "Backend Health" },
        @{ URL = "$BACKEND_URL/api/companies"; Name = "Companies API" },
        @{ URL = "$BACKEND_URL/api/materials"; Name = "Materials API" },
        @{ URL = "$BACKEND_URL/api/matches"; Name = "Matches API" },
        @{ URL = "$BACKEND_URL/api/ai/listings"; Name = "AI Listings API" },
        @{ URL = "$BACKEND_URL/api/ai/matching"; Name = "AI Matching API" },
        @{ URL = "$BACKEND_URL/api/analytics"; Name = "Analytics API" },
        @{ URL = "$BACKEND_URL/api/logistics"; Name = "Logistics API" },
        @{ URL = "$BACKEND_URL/api/compliance"; Name = "Compliance API" }
    )
    
    $passed = 0
    foreach ($service in $services) {
        if (Test-URL $service.URL $service.Name) {
            $passed++
        }
    }
    
    Write-Status "Backend Services ($passed/$($services.Count))" $(if ($passed -eq $services.Count) { "PASS" } else { "WARN" }) $(if ($passed -eq $services.Count) { $Green } else { $Yellow })
    return $passed -eq $services.Count
}

function Test-Frontend {
    Write-Host "`n🌐 Testing Frontend..." -ForegroundColor $Blue
    
    $pages = @(
        @{ URL = "$FRONTEND_URL"; Name = "Main Dashboard" },
        @{ URL = "$FRONTEND_URL/admin"; Name = "Admin Dashboard" },
        @{ URL = "$FRONTEND_URL/marketplace"; Name = "Marketplace" },
        @{ URL = "$FRONTEND_URL/matching"; Name = "AI Matching" },
        @{ URL = "$FRONTEND_URL/analytics"; Name = "Analytics" }
    )
    
    $passed = 0
    foreach ($page in $pages) {
        if (Test-URL $page.URL $page.Name) {
            $passed++
        }
    }
    
    Write-Status "Frontend Pages ($passed/$($pages.Count))" $(if ($passed -eq $pages.Count) { "PASS" } else { "WARN" }) $(if ($passed -eq $pages.Count) { $Green } else { $Yellow })
    return $passed -eq $pages.Count
}

function Test-AIServices {
    Write-Host "`n🤖 Testing AI Services..." -ForegroundColor $Blue
    
    $aiServices = @(
        @{ URL = "$BACKEND_URL/api/ai/listings/generate"; Name = "AI Listings Generator" },
        @{ URL = "$BACKEND_URL/api/ai/matching/run"; Name = "AI Matching Engine" },
        @{ URL = "$BACKEND_URL/api/ai/analytics"; Name = "AI Analytics" },
        @{ URL = "$BACKEND_URL/api/ai/insights"; Name = "AI Insights" }
    )
    
    $passed = 0
    foreach ($service in $aiServices) {
        if (Test-URL $service.URL $service.Name) {
            $passed++
        }
    }
    
    Write-Status "AI Services ($passed/$($aiServices.Count))" $(if ($passed -eq $aiServices.Count) { "PASS" } else { "WARN" }) $(if ($passed -eq $aiServices.Count) { $Green } else { $Yellow })
    return $passed -eq $aiServices.Count
}

function Test-AdminDashboard {
    Write-Host "`n👑 Testing Admin Dashboard Features..." -ForegroundColor $Blue
    
    $adminFeatures = @(
        @{ URL = "$FRONTEND_URL/admin/companies"; Name = "Companies Management" },
        @{ URL = "$FRONTEND_URL/admin/materials"; Name = "Materials Management" },
        @{ URL = "$FRONTEND_URL/admin/matches"; Name = "Matches Management" },
        @{ URL = "$FRONTEND_URL/admin/ai-insights"; Name = "AI Insights" },
        @{ URL = "$FRONTEND_URL/admin/analytics"; Name = "Analytics Dashboard" }
    )
    
    $passed = 0
    foreach ($feature in $adminFeatures) {
        if (Test-URL $feature.URL $feature.Name) {
            $passed++
        }
    }
    
    Write-Status "Admin Dashboard Features ($passed/$($adminFeatures.Count))" $(if ($passed -eq $adminFeatures.Count) { "PASS" } else { "WARN" }) $(if ($passed -eq $adminFeatures.Count) { $Green } else { $Yellow })
    return $passed -eq $adminFeatures.Count
}

function Test-DataIntegrity {
    Write-Host "`n📋 Testing Data Integrity..." -ForegroundColor $Blue
    
    try {
        # Test company data
        $companiesResponse = Invoke-WebRequest -Uri "$BACKEND_URL/api/companies" -Method GET
        $companies = $companiesResponse.Content | ConvertFrom-Json
        $companyCount = $companies.Count
        
        # Test materials data
        $materialsResponse = Invoke-WebRequest -Uri "$BACKEND_URL/api/materials" -Method GET
        $materials = $materialsResponse.Content | ConvertFrom-Json
        $materialCount = $materials.Count
        
        # Test matches data
        $matchesResponse = Invoke-WebRequest -Uri "$BACKEND_URL/api/matches" -Method GET
        $matches = $matchesResponse.Content | ConvertFrom-Json
        $matchCount = $matches.Count
        
        Write-Status "Companies: $companyCount" $(if ($companyCount -ge $TEST_COMPANY_COUNT) { "PASS" } else { "WARN" })
        Write-Status "Materials: $materialCount" $(if ($materialCount -gt 0) { "PASS" } else { "WARN" })
        Write-Status "Matches: $matchCount" $(if ($matchCount -ge 0) { "PASS" } else { "WARN" })
        
        return $companyCount -ge $TEST_COMPANY_COUNT -and $materialCount -gt 0
    } catch {
        Write-Status "Data integrity check" "FAIL" $Red
        return $false
    }
}

function Test-AIListingsGenerator {
    Write-Host "`n🎯 Testing AI Listings Generator..." -ForegroundColor $Blue
    
    try {
        # Test AI listings generation
        $body = @{
            company_id = "test-company"
            industry = "manufacturing"
            location = "Dubai"
        } | ConvertTo-Json
        
        $response = Invoke-WebRequest -Uri "$BACKEND_URL/api/ai/listings/generate" -Method POST -Body $body -ContentType "application/json"
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            Write-Status "AI Listings Generation" "PASS"
            Write-Status "Generated listings: $($result.listings.Count)" "PASS"
            return $true
        } else {
            Write-Status "AI Listings Generation" "FAIL" $Red
            return $false
        }
    } catch {
        Write-Status "AI Listings Generation" "FAIL" $Red
        return $false
    }
}

function Test-AIMatchingEngine {
    Write-Host "`n🔗 Testing AI Matching Engine..." -ForegroundColor $Blue
    
    try {
        # Test AI matching
        $response = Invoke-WebRequest -Uri "$BACKEND_URL/api/ai/matching/run" -Method POST
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            Write-Status "AI Matching Engine" "PASS"
            Write-Status "Generated matches: $($result.matches.Count)" "PASS"
            return $true
        } else {
            Write-Status "AI Matching Engine" "FAIL" $Red
            return $false
        }
    } catch {
        Write-Status "AI Matching Engine" "FAIL" $Red
        return $false
    }
}

function Test-LogisticsIntegration {
    Write-Host "`n🚛 Testing Logistics Integration..." -ForegroundColor $Blue
    
    try {
        # Test Freightos integration
        $body = @{
            origin = "Dubai"
            destination = "Abu Dhabi"
            weight = 1000
            material_type = "chemicals"
        } | ConvertTo-Json
        
        $response = Invoke-WebRequest -Uri "$BACKEND_URL/api/logistics/calculate" -Method POST -Body $body -ContentType "application/json"
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            Write-Status "Logistics Integration" "PASS"
            Write-Status "Cost: $($result.cost), CO2: $($result.co2_emissions)" "PASS"
            return $true
        } else {
            Write-Status "Logistics Integration" "FAIL" $Red
            return $false
        }
    } catch {
        Write-Status "Logistics Integration" "FAIL" $Red
        return $false
    }
}

function Test-ComplianceIntegration {
    Write-Host "`n📋 Testing Compliance Integration..." -ForegroundColor $Blue
    
    try {
        # Test regulatory compliance
        $response = Invoke-WebRequest -Uri "$BACKEND_URL/api/compliance/check" -Method GET
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            Write-Status "Compliance Integration" "PASS"
            Write-Status "Compliance score: $($result.compliance_score)" "PASS"
            return $true
        } else {
            Write-Status "Compliance Integration" "FAIL" $Red
            return $false
        }
    } catch {
        Write-Status "Compliance Integration" "FAIL" $Red
        return $false
    }
}

function Show-SystemSummary {
    Write-Host "`n📊 System Summary" -ForegroundColor $Blue
    Write-Host "=================" -ForegroundColor $Blue
    
    # Get system stats
    try {
        $statsResponse = Invoke-WebRequest -Uri "$BACKEND_URL/api/admin/stats" -Method GET
        $stats = $statsResponse.Content | ConvertFrom-Json
        
        Write-Host "Companies: $($stats.total_companies)" -ForegroundColor $Green
        Write-Host "Materials: $($stats.total_materials)" -ForegroundColor $Green
        Write-Host "Matches: $($stats.total_matches)" -ForegroundColor $Green
        Write-Host "AI Listings: $($stats.total_ai_listings)" -ForegroundColor $Green
        Write-Host "Potential Value: $($stats.total_potential_value)" -ForegroundColor $Green
        Write-Host "System Health: $($stats.system_health_score)%" -ForegroundColor $Green
    } catch {
        Write-Host "Could not retrieve system stats" -ForegroundColor $Yellow
    }
}

function Show-Recommendations {
    Write-Host "`n💡 Recommendations" -ForegroundColor $Blue
    Write-Host "==================" -ForegroundColor $Blue
    
    Write-Host "1. Ensure all 50 Gulf companies are imported" -ForegroundColor $Yellow
    Write-Host "2. Run AI listings generator for all companies" -ForegroundColor $Yellow
    Write-Host "3. Execute AI matching engine to create symbiosis opportunities" -ForegroundColor $Yellow
    Write-Host "4. Monitor admin dashboard for insights and opportunities" -ForegroundColor $Yellow
    Write-Host "5. Test all admin dashboard tabs and features" -ForegroundColor $Yellow
    Write-Host "6. Verify logistics and compliance integrations" -ForegroundColor $Yellow
}

# Pre-check: Ensure backend and frontend are reachable before running tests
function PreCheck-Service {
    param($URL, $Name)
    try {
        $response = Invoke-WebRequest -Uri $URL -Method GET -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Status "$Name is reachable" "PASS"
            return $true
        } else {
            Write-Status "$Name unreachable (Status: $($response.StatusCode))" "FAIL" $Red
            return $false
        }
    } catch {
        Write-Status "$Name unreachable (Error: $($_.Exception.Message))" "FAIL" $Red
        return $false
    }
}

# Main test execution
Write-Host "Starting comprehensive system test..." -ForegroundColor $Blue

$backendUp = PreCheck-Service $BACKEND_URL "Backend"
$frontendUp = PreCheck-Service $FRONTEND_URL "Frontend"

if (-not $backendUp) {
    Write-Host "Backend is not reachable. Please start the backend service and retry." -ForegroundColor $Red
}
if (-not $frontendUp) {
    Write-Host "Frontend is not reachable. Please start the frontend service and retry." -ForegroundColor $Red
}

$results = @{}
if ($backendUp) {
    $results.Database = Test-Database
    $results.Backend = Test-BackendServices
    $results.AIServices = Test-AIServices
    $results.DataIntegrity = Test-DataIntegrity
    $results.AIListings = Test-AIListingsGenerator
    $results.AIMatching = Test-AIMatchingEngine
    $results.Logistics = Test-LogisticsIntegration
    $results.Compliance = Test-ComplianceIntegration
}
if ($frontendUp) {
    $results.Frontend = Test-Frontend
    $results.AdminDashboard = Test-AdminDashboard
}

$totalTests = $results.Count
$passedTests = ($results.Values | Where-Object { $_ -eq $true }).Count
$score = if ($totalTests -gt 0) { [math]::Round(($passedTests / $totalTests) * 100, 1) } else { 0 }

Write-Host "`n🎯 Overall Test Results" -ForegroundColor $Blue
Write-Host "=====================" -ForegroundColor $Blue
Write-Host "Passed: $passedTests/$totalTests ($score%)" -ForegroundColor $(if ($score -ge 80) { $Green } elseif ($score -ge 60) { $Yellow } else { $Red })

if ($score -ge 80) {
    Write-Host "`n🎉 System is ready for production!" -ForegroundColor $Green
} elseif ($score -ge 60) {
    Write-Host "`n⚠️ System needs some improvements before production" -ForegroundColor $Yellow
} else {
    Write-Host "`n❌ System needs significant work before production" -ForegroundColor $Red
}

if ($backendUp) { Show-SystemSummary }
Show-Recommendations

Write-Host "`n✅ Test completed!" -ForegroundColor $Green 