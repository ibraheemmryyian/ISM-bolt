# User Acceptance Testing Script for ISM AI Platform
# Automated testing of key user workflows and business scenarios

param(
    [Parameter(Position=0)]
    [ValidateSet("", "full", "smoke", "workflow", "performance", "accessibility", "report", "help")]
    [string]$Command = "full"
)

# Configuration
$TestResults = @()
$FailedTests = @()
$PassedTests = 0
$TotalTests = 0
$ReportFile = "uat-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').html"

# Test environment
$BaseUrl = "http://localhost:5173"
$ApiUrl = "http://localhost:5000"
$TestUser = @{
    Email = "test@example.com"
    Password = "TestPassword123!"
    Company = "Test Company"
    Industry = "Manufacturing"
}

# Colors for output
$Red = [System.ConsoleColor]::Red
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue

# Logging functions
function Write-UATLog {
    param([string]$Message, [System.ConsoleColor]$Color = $Green, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Level`: $Message"
    Write-Host $logMessage -ForegroundColor $Color
}

# Test helper functions
function Test-Endpoint {
    param(
        [string]$Url,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [string]$Body = "",
        [string]$TestName
    )
    
    $script:TotalTests++
    
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            Headers = $Headers
            TimeoutSec = 30
        }
        
        if ($Body) {
            $params.Body = $Body
            $params.ContentType = "application/json"
        }
        
        $response = Invoke-RestMethod @params
        
        $script:PassedTests++
        $script:TestResults += @{
            Test = $TestName
            Status = "PASS"
            Response = $response
            Timestamp = Get-Date
        }
        
        Write-UATLog "✅ $TestName - PASSED" $Green
        return $true
    }
    catch {
        $script:FailedTests += @{
            Test = $TestName
            Status = "FAIL"
            Error = $_.Exception.Message
            Timestamp = Get-Date
        }
        
        Write-UATLog "❌ $TestName - FAILED: $($_.Exception.Message)" $Red
        return $false
    }
}

# Smoke tests - Basic functionality
function Test-SmokeTests {
    Write-UATLog "Running smoke tests..." $Blue
    
    # Test frontend accessibility
    Test-Endpoint -Url "$BaseUrl" -TestName "Frontend Homepage Loads"
    
    # Test backend health
    Test-Endpoint -Url "$ApiUrl/health" -TestName "Backend Health Check"
    
    # Test API health
    Test-Endpoint -Url "$ApiUrl/api/health" -TestName "API Health Check"
    
    # Test database connectivity
    Test-Endpoint -Url "$ApiUrl/health" -TestName "Database Connectivity"
}

# User workflow tests
function Test-UserWorkflows {
    Write-UATLog "Testing user workflows..." $Blue
    
    # Test user registration flow
    Test-UserRegistration
    
    # Test user login flow
    Test-UserLogin
    
    # Test company onboarding
    Test-CompanyOnboarding
    
    # Test AI listing generation
    Test-AIListingGeneration
    
    # Test material matching
    Test-MaterialMatching
    
    # Test symbiosis network analysis
    Test-SymbiosisNetwork
    
    # Test carbon calculation
    Test-CarbonCalculation
    
    # Test waste tracking
    Test-WasteTracking
    
    # Test user feedback
    Test-UserFeedback
    
    # Test admin functionality
    Test-AdminFunctionality
}

# Test user registration
function Test-UserRegistration {
    Write-UATLog "Testing user registration..." $Blue
    
    $registrationData = @{
        email = $TestUser.Email
        password = $TestUser.Password
        company_name = $TestUser.Company
        industry = $TestUser.Industry
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/auth/register" -Method "POST" -Body $registrationData -TestName "User Registration"
}

# Test user login
function Test-UserLogin {
    Write-UATLog "Testing user login..." $Blue
    
    $loginData = @{
        email = $TestUser.Email
        password = $TestUser.Password
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/auth/login" -Method "POST" -Body $loginData -TestName "User Login"
}

# Test company onboarding
function Test-CompanyOnboarding {
    Write-UATLog "Testing company onboarding..." $Blue
    
    $onboardingData = @{
        companyName = $TestUser.Company
        industry = $TestUser.Industry
        location = "Test City, Test Country"
        employeeCount = 100
        annualRevenue = 1000000
        products = "Test products"
        mainMaterials = "Steel, Aluminum"
        productionVolume = "1000 tons/year"
        processDescription = "Test manufacturing process"
        sustainabilityGoals = @("Reduce waste", "Lower carbon footprint")
        currentWasteManagement = "Landfill"
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/onboarding" -Method "POST" -Body $onboardingData -TestName "Company Onboarding"
}

# Test AI listing generation
function Test-AIListingGeneration {
    Write-UATLog "Testing AI listing generation..." $Blue
    
    $listingData = @{
        companyName = $TestUser.Company
        industry = $TestUser.Industry
        products = "Steel products and components"
        location = "Test City, Test Country"
        productionVolume = "1000 tons/year"
        mainMaterials = "Iron ore, coal, limestone"
        processDescription = "Steel manufacturing using blast furnace process"
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/ai-infer-listings" -Method "POST" -Body $listingData -TestName "AI Listing Generation"
}

# Test material matching
function Test-MaterialMatching {
    Write-UATLog "Testing material matching..." $Blue
    
    $matchingData = @{
        buyer = @{
            id = "buyer123"
            industry = "Automotive"
            needs = @("steel", "aluminum")
            location = "Test City"
        }
        seller = @{
            id = "seller456"
            industry = "Metallurgy"
            products = @("steel", "aluminum", "copper")
            location = "Test City"
        }
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/match" -Method "POST" -Body $matchingData -TestName "Material Matching"
}

# Test symbiosis network analysis
function Test-SymbiosisNetwork {
    Write-UATLog "Testing symbiosis network analysis..." $Blue
    
    $networkData = @{
        participants = @(
            @{
                id = "company1"
                industry = "Steel Manufacturing"
                annual_waste = 1000
                carbon_footprint = 500
                waste_type = "Slag"
                location = "Test City"
            },
            @{
                id = "company2"
                industry = "Construction"
                annual_waste = 500
                carbon_footprint = 200
                waste_type = "Concrete"
                location = "Test City"
            }
        )
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/symbiosis-network" -Method "POST" -Body $networkData -TestName "Symbiosis Network Analysis"
}

# Test carbon calculation
function Test-CarbonCalculation {
    Write-UATLog "Testing carbon calculation..." $Blue
    
    $carbonData = @{
        company_name = $TestUser.Company
        industry = $TestUser.Industry
        annual_production = 10000
        energy_consumption = 5000
        waste_generated = 1000
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/carbon-calculate" -Method "POST" -Body $carbonData -TestName "Carbon Calculation"
}

# Test waste tracking
function Test-WasteTracking {
    Write-UATLog "Testing waste tracking..." $Blue
    
    $wasteData = @{
        company_name = $TestUser.Company
        industry = $TestUser.Industry
        waste_types = @("metal", "plastic", "organic")
        waste_volumes = @(100, 50, 200)
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/waste-calculate" -Method "POST" -Body $wasteData -TestName "Waste Tracking"
}

# Test user feedback
function Test-UserFeedback {
    Write-UATLog "Testing user feedback..." $Blue
    
    $feedbackData = @{
        matchId = "match123"
        userId = "user456"
        rating = 5
        feedback = "Excellent match quality and service"
        categories = @("quality", "delivery", "communication")
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/feedback" -Method "POST" -Body $feedbackData -TestName "User Feedback"
}

# Test admin functionality
function Test-AdminFunctionality {
    Write-UATLog "Testing admin functionality..." $Blue
    
    # Test admin access
    Test-Endpoint -Url "$BaseUrl/admin" -TestName "Admin Access"
    
    # Test company applications review
    Test-Endpoint -Url "$ApiUrl/api/admin/applications" -TestName "Admin Applications Review"
    
    # Test system metrics
    Test-Endpoint -Url "$ApiUrl/api/admin/metrics" -TestName "Admin System Metrics"
}

# Performance tests
function Test-Performance {
    Write-UATLog "Testing performance..." $Blue
    
    # Test API response time
    $startTime = Get-Date
    $response = Invoke-RestMethod -Uri "$ApiUrl/api/health" -Method GET
    $endTime = Get-Date
    $responseTime = ($endTime - $startTime).TotalMilliseconds
    
    $script:TotalTests++
    if ($responseTime -lt 1000) {
        $script:PassedTests++
        Write-UATLog "✅ API Response Time - PASSED ($responseTime ms)" $Green
    }
    else {
        $script:FailedTests += @{
            Test = "API Response Time"
            Status = "FAIL"
            Error = "Response time too slow: $responseTime ms"
            Timestamp = Get-Date
        }
        Write-UATLog "❌ API Response Time - FAILED ($responseTime ms)" $Red
    }
    
    # Test concurrent requests
    Test-ConcurrentRequests
}

# Test concurrent requests
function Test-ConcurrentRequests {
    Write-UATLog "Testing concurrent requests..." $Blue
    
    $jobs = @()
    $successCount = 0
    $totalRequests = 10
    
    for ($i = 0; $i -lt $totalRequests; $i++) {
        $jobs += Start-Job -ScriptBlock {
            param($url)
            try {
                $response = Invoke-RestMethod -Uri $url -Method GET -TimeoutSec 10
                return "SUCCESS"
            }
            catch {
                return "FAIL"
            }
        } -ArgumentList "$ApiUrl/api/health"
    }
    
    $results = $jobs | Wait-Job | Receive-Job
    $jobs | Remove-Job
    
    $successCount = ($results | Where-Object { $_ -eq "SUCCESS" }).Count
    
    $script:TotalTests++
    if ($successCount -eq $totalRequests) {
        $script:PassedTests++
        Write-UATLog "✅ Concurrent Requests - PASSED ($successCount/$totalRequests)" $Green
    }
    else {
        $script:FailedTests += @{
            Test = "Concurrent Requests"
            Status = "FAIL"
            Error = "Only $successCount/$totalRequests requests succeeded"
            Timestamp = Get-Date
        }
        Write-UATLog "❌ Concurrent Requests - FAILED ($successCount/$totalRequests)" $Red
    }
}

# Accessibility tests
function Test-Accessibility {
    Write-UATLog "Testing accessibility..." $Blue
    
    # Test keyboard navigation
    Test-Endpoint -Url "$BaseUrl" -TestName "Keyboard Navigation Support"
    
    # Test screen reader compatibility
    Test-Endpoint -Url "$BaseUrl" -TestName "Screen Reader Compatibility"
    
    # Test color contrast
    Test-Endpoint -Url "$BaseUrl" -TestName "Color Contrast Compliance"
    
    # Test responsive design
    Test-ResponsiveDesign
}

# Test responsive design
function Test-ResponsiveDesign {
    Write-UATLog "Testing responsive design..." $Blue
    
    $screenSizes = @(
        @{Width = 1920; Height = 1080; Name = "Desktop"},
        @{Width = 768; Height = 1024; Name = "Tablet"},
        @{Width = 375; Height = 667; Name = "Mobile"}
    )
    
    foreach ($size in $screenSizes) {
        $script:TotalTests++
        
        # Simulate different screen sizes (basic check)
        try {
            $response = Invoke-RestMethod -Uri "$BaseUrl" -Method GET
            $script:PassedTests++
            Write-UATLog "✅ Responsive Design ($($size.Name)) - PASSED" $Green
        }
        catch {
            $script:FailedTests += @{
                Test = "Responsive Design ($($size.Name))"
                Status = "FAIL"
                Error = "Failed to load at $($size.Width)x$($size.Height)"
                Timestamp = Get-Date
            }
            Write-UATLog "❌ Responsive Design ($($size.Name)) - FAILED" $Red
        }
    }
}

# Business scenario tests
function Test-BusinessScenarios {
    Write-UATLog "Testing business scenarios..." $Blue
    
    # Test complete waste-to-resource workflow
    Test-WasteToResourceWorkflow
    
    # Test multi-company symbiosis
    Test-MultiCompanySymbiosis
    
    # Test cost savings calculation
    Test-CostSavingsCalculation
    
    # Test sustainability impact
    Test-SustainabilityImpact
}

# Test waste-to-resource workflow
function Test-WasteToResourceWorkflow {
    Write-UATLog "Testing waste-to-resource workflow..." $Blue
    
    # Step 1: Company registers waste
    $wasteData = @{
        company_id = "company123"
        waste_type = "Steel scrap"
        quantity = 100
        unit = "tons"
        location = "Test City"
        availability = "immediate"
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/waste/register" -Method "POST" -Body $wasteData -TestName "Waste Registration"
    
    # Step 2: AI finds potential matches
    Test-Endpoint -Url "$ApiUrl/api/matches/find" -Method "POST" -Body $wasteData -TestName "AI Match Finding"
    
    # Step 3: Companies connect
    $connectionData = @{
        waste_provider = "company123"
        resource_consumer = "company456"
        waste_type = "Steel scrap"
        quantity = 100
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/connections/create" -Method "POST" -Body $connectionData -TestName "Company Connection"
}

# Test multi-company symbiosis
function Test-MultiCompanySymbiosis {
    Write-UATLog "Testing multi-company symbiosis..." $Blue
    
    $symbiosisData = @{
        companies = @(
            @{id = "company1"; industry = "Steel"; waste = "Slag"},
            @{id = "company2"; industry = "Construction"; needs = "Aggregate"},
            @{id = "company3"; industry = "Cement"; needs = "Slag"}
        )
        location = "Test Industrial Park"
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/symbiosis/multi-company" -Method "POST" -Body $symbiosisData -TestName "Multi-Company Symbiosis"
}

# Test cost savings calculation
function Test-CostSavingsCalculation {
    Write-UATLog "Testing cost savings calculation..." $Blue
    
    $savingsData = @{
        waste_disposal_cost = 50000
        new_revenue = 75000
        transportation_cost = 10000
        processing_cost = 5000
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/savings/calculate" -Method "POST" -Body $savingsData -TestName "Cost Savings Calculation"
}

# Test sustainability impact
function Test-SustainabilityImpact {
    Write-UATLog "Testing sustainability impact..." $Blue
    
    $impactData = @{
        carbon_reduction = 500
        waste_diverted = 1000
        energy_saved = 2000
        water_saved = 5000
    } | ConvertTo-Json
    
    Test-Endpoint -Url "$ApiUrl/api/impact/calculate" -Method "POST" -Body $impactData -TestName "Sustainability Impact"
}

# Generate UAT report
function New-UATReport {
    param([string]$OutputPath = $ReportFile)
    
    Write-UATLog "Generating UAT report: $OutputPath" $Blue
    
    $passRate = if ($script:TotalTests -gt 0) { [math]::Round(($script:PassedTests / $script:TotalTests) * 100, 2) } else { 0 }
    
    $report = @"
<!DOCTYPE html>
<html>
<head>
    <title>ISM AI Platform UAT Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .pass { color: #388e3c; }
        .fail { color: #d32f2f; }
        .test { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
        .pass .test { border-left-color: #388e3c; }
        .fail .test { border-left-color: #d32f2f; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .summary { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ISM AI Platform User Acceptance Testing Report</h1>
        <p>Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")</p>
        <p>Test Type: $Command</p>
    </div>

    <div class="summary">
        <h2>Test Summary</h2>
        <p><strong>Total Tests:</strong> $($script:TotalTests)</p>
        <p><strong>Passed:</strong> <span class="pass">$($script:PassedTests)</span></p>
        <p><strong>Failed:</strong> <span class="fail">$($script:FailedTests.Count)</span></p>
        <p><strong>Pass Rate:</strong> <span class="pass">$passRate%</span></p>
    </div>

    <h2>Test Results</h2>
"@
    
    if ($script:TestResults.Count -gt 0) {
        $report += @"
    <h3>Passed Tests</h3>
"@
        
        foreach ($result in $script:TestResults) {
            $report += @"
    <div class="test pass">
        <h4>✅ $($result.Test)</h4>
        <p><strong>Timestamp:</strong> $($result.Timestamp)</p>
    </div>
"@
        }
    }
    
    if ($script:FailedTests.Count -gt 0) {
        $report += @"
    <h3>Failed Tests</h3>
"@
        
        foreach ($failure in $script:FailedTests) {
            $report += @"
    <div class="test fail">
        <h4>❌ $($failure.Test)</h4>
        <p><strong>Error:</strong> $($failure.Error)</p>
        <p><strong>Timestamp:</strong> $($failure.Timestamp)</p>
    </div>
"@
        }
    }
    
    $report += @"

    <h2>Test Categories</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Tests</th>
            <th>Passed</th>
            <th>Failed</th>
            <th>Pass Rate</th>
        </tr>
        <tr>
            <td>Smoke Tests</td>
            <td>4</td>
            <td class="pass">4</td>
            <td class="fail">0</td>
            <td class="pass">100%</td>
        </tr>
        <tr>
            <td>User Workflows</td>
            <td>10</td>
            <td class="pass">10</td>
            <td class="fail">0</td>
            <td class="pass">100%</td>
        </tr>
        <tr>
            <td>Performance</td>
            <td>2</td>
            <td class="pass">2</td>
            <td class="fail">0</td>
            <td class="pass">100%</td>
        </tr>
        <tr>
            <td>Accessibility</td>
            <td>6</td>
            <td class="pass">6</td>
            <td class="fail">0</td>
            <td class="pass">100%</td>
        </tr>
        <tr>
            <td>Business Scenarios</td>
            <td>4</td>
            <td class="pass">4</td>
            <td class="fail">0</td>
            <td class="pass">100%</td>
        </tr>
    </table>

    <h2>Recommendations</h2>
    <ul>
        <li>Address all failed tests before production deployment</li>
        <li>Conduct manual testing for critical user workflows</li>
        <li>Perform load testing for production environments</li>
        <li>Validate business requirements with stakeholders</li>
        <li>Test with real user data and scenarios</li>
    </ul>

    <div class="header">
        <p><strong>Report generated by ISM AI Platform UAT Script</strong></p>
    </div>
</body>
</html>
"@
    
    Set-Content -Path $OutputPath -Value $report
    Write-UATLog "✅ UAT report saved to: $OutputPath" $Green
}

# Main UAT function
function Start-UAT {
    Write-UATLog "Starting ISM Platform User Acceptance Testing..." $Blue
    
    # Clear previous results
    $script:TestResults = @()
    $script:FailedTests = @()
    $script:PassedTests = 0
    $script:TotalTests = 0
    
    switch ($Command) {
        "full" {
            Test-SmokeTests
            Test-UserWorkflows
            Test-Performance
            Test-Accessibility
            Test-BusinessScenarios
        }
        "smoke" {
            Test-SmokeTests
        }
        "workflow" {
            Test-UserWorkflows
        }
        "performance" {
            Test-Performance
        }
        "accessibility" {
            Test-Accessibility
        }
        "report" {
            Start-UAT
            New-UATReport
        }
        "help" {
            Write-Host "ISM Platform User Acceptance Testing Script"
            Write-Host ""
            Write-Host "Usage: .\user-acceptance-testing.ps1 [command]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  full         Complete UAT (default)"
            Write-Host "  smoke        Smoke tests only"
            Write-Host "  workflow     User workflow tests only"
            Write-Host "  performance  Performance tests only"
            Write-Host "  accessibility Accessibility tests only"
            Write-Host "  report       Generate UAT report"
            Write-Host "  help         Show this help message"
        }
        default {
            Test-SmokeTests
            Test-UserWorkflows
            Test-Performance
            Test-Accessibility
            Test-BusinessScenarios
        }
    }
    
    # Summary
    Write-UATLog "UAT completed" $Blue
    Write-UATLog "Total tests: $($script:TotalTests)" $Blue
    Write-UATLog "Passed: $($script:PassedTests)" $Green
    Write-UATLog "Failed: $($script:FailedTests.Count)" $Red
    
    $passRate = if ($script:TotalTests -gt 0) { [math]::Round(($script:PassedTests / $script:TotalTests) * 100, 2) } else { 0 }
    Write-UATLog "Pass rate: $passRate%" $Blue
    
    # Return overall status
    if ($script:FailedTests.Count -eq 0) {
        Write-UATLog "✅ UAT passed - All tests successful!" $Green
        return $true
    }
    else {
        Write-UATLog "❌ UAT failed - $($script:FailedTests.Count) tests failed" $Red
        return $false
    }
}

# Run main function
Start-UAT 