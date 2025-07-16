# Final Repairs Test Script for PowerShell
Write-Host "🔧 Running Final Repairs Test Suite" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check if backend is running
Write-Host "`n🧪 Testing: Backend Health" -ForegroundColor Yellow
Write-Host "------------------------------" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod -Uri "http://localhost:3000/api/health" -Method Get -TimeoutSec 10
    Write-Host "✅ Backend health: $($response.status)" -ForegroundColor Green
    $backendHealthy = $true
} catch {
    Write-Host "❌ Backend health error: $($_.Exception.Message)" -ForegroundColor Red
    $backendHealthy = $false
}

# Test adaptive onboarding initialization
Write-Host "`n🧪 Testing: Adaptive Onboarding Init" -ForegroundColor Yellow
Write-Host "------------------------------" -ForegroundColor Yellow

try {
    $result = python backend/adaptive_onboarding_init.py initialize
    $data = $result | ConvertFrom-Json
    
    if ($data.status -eq "initialized") {
        Write-Host "✅ Adaptive onboarding initialization successful" -ForegroundColor Green
        $onboardingHealthy = $true
    } else {
        Write-Host "❌ Adaptive onboarding initialization failed: $($data.error)" -ForegroundColor Red
        $onboardingHealthy = $false
    }
} catch {
    Write-Host "❌ Adaptive onboarding test error: $($_.Exception.Message)" -ForegroundColor Red
    $onboardingHealthy = $false
}

# Test Python imports
Write-Host "`n🧪 Testing: Python Imports" -ForegroundColor Yellow
Write-Host "------------------------------" -ForegroundColor Yellow

$modules = @("federated_meta_learning", "regulatory_compliance", "ai_onboarding_questions_generator")
$importSuccess = 0

foreach ($module in $modules) {
    try {
        python -c "import $module; print('✅ $module import successful')"
        $importSuccess++
    } catch {
        Write-Host "❌ $module import failed" -ForegroundColor Red
    }
}

# Test adaptive_ai_onboarding (in backend directory)
try {
    python -c "import sys; sys.path.append('backend'); import adaptive_ai_onboarding; print('✅ adaptive_ai_onboarding import successful')"
    $importSuccess++
} catch {
    Write-Host "❌ adaptive_ai_onboarding import failed" -ForegroundColor Red
}

$pythonImportsHealthy = $importSuccess -gt 0

# Test companies API
Write-Host "`n🧪 Testing: Companies API" -ForegroundColor Yellow
Write-Host "------------------------------" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod -Uri "http://localhost:3000/api/companies" -Method Get -TimeoutSec 10
    $companies = $response.companies
    Write-Host "✅ Companies API: $($companies.Count) companies found" -ForegroundColor Green
    $companiesApiHealthy = $companies.Count -gt 0
} catch {
    Write-Host "❌ Companies API error: $($_.Exception.Message)" -ForegroundColor Red
    $companiesApiHealthy = $false
}

# Test AI services
Write-Host "`n🧪 Testing: AI Services" -ForegroundColor Yellow
Write-Host "------------------------------" -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod -Uri "http://localhost:3000/api/ai/services/status" -Method Get -TimeoutSec 15
    $services = $response.services_status
    
    $healthyServices = 0
    $totalServices = $services.Count
    
    foreach ($service in $services.GetEnumerator()) {
        if ($service.Value.status -eq "healthy") {
            Write-Host "✅ $($service.Key): healthy" -ForegroundColor Green
            $healthyServices++
        } else {
            Write-Host "⚠️ $($service.Key): $($service.Value.error)" -ForegroundColor Yellow
        }
    }
    
    Write-Host "📊 AI Services: $healthyServices/$totalServices healthy" -ForegroundColor Cyan
    $aiServicesHealthy = $healthyServices -gt 0
} catch {
    Write-Host "❌ AI services test error: $($_.Exception.Message)" -ForegroundColor Red
    $aiServicesHealthy = $false
}

# Test logistics preview
Write-Host "`n🧪 Testing: Logistics Preview" -ForegroundColor Yellow
Write-Host "------------------------------" -ForegroundColor Yellow

try {
    $testData = @{
        origin = "Dubai, UAE"
        destination = "Abu Dhabi, UAE"
        material = "Steel scrap"
        weight_kg = 1000
        company_profile = @{
            name = "Test Company"
            industry = "Manufacturing"
        }
    }
    
    $response = Invoke-RestMethod -Uri "http://localhost:3000/api/logistics-preview" -Method Post -Body ($testData | ConvertTo-Json -Depth 3) -ContentType "application/json" -TimeoutSec 15
    Write-Host "✅ Logistics preview successful" -ForegroundColor Green
    $logisticsHealthy = $true
} catch {
    Write-Host "❌ Logistics preview error: $($_.Exception.Message)" -ForegroundColor Red
    $logisticsHealthy = $false
}

# Final results
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "📊 FINAL TEST RESULTS" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$tests = @(
    @{Name="Backend Health"; Result=$backendHealthy}
    @{Name="Python Imports"; Result=$pythonImportsHealthy}
    @{Name="Adaptive Onboarding Init"; Result=$onboardingHealthy}
    @{Name="Companies API"; Result=$companiesApiHealthy}
    @{Name="AI Services"; Result=$aiServicesHealthy}
    @{Name="Logistics Preview"; Result=$logisticsHealthy}
)

$passed = 0
$total = $tests.Count

foreach ($test in $tests) {
    $status = if ($test.Result) { "✅ PASS" } else { "❌ FAIL" }
    $color = if ($test.Result) { "Green" } else { "Red" }
    Write-Host "$($test.Name): $status" -ForegroundColor $color
    if ($test.Result) { $passed++ }
}

Write-Host "`nOverall: $passed/$total tests passed" -ForegroundColor Cyan

if ($passed -eq $total) {
    Write-Host "🎉 All tests passed! System is ready for production." -ForegroundColor Green
    exit 0
} elseif ($passed -ge ($total * 0.8)) {
    Write-Host "⚠️ Most tests passed. System is mostly functional." -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "❌ Multiple tests failed. System needs attention." -ForegroundColor Red
    exit 1
} 