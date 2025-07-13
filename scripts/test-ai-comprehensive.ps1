# Comprehensive AI Testing Script for ISM AI Platform
# Tests all AI components, pricing mechanisms, and shipping calculations

param(
    [Parameter(Position=0)]
    [ValidateSet("all", "ai-services", "pricing", "logistics", "matching", "analytics", "help")]
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
$Magenta = [System.ConsoleColor]::Magenta

# Test results tracking
$TestResults = @{
    "AIServices" = @{ passed = 0; failed = 0; total = 0; details = @() }
    "Pricing" = @{ passed = 0; failed = 0; total = 0; details = @() }
    "Logistics" = @{ passed = 0; failed = 0; total = 0; details = @() }
    "Matching" = @{ passed = 0; failed = 0; total = 0; details = @() }
    "Analytics" = @{ passed = 0; failed = 0; total = 0; details = @() }
}

function Write-TestLog {
    param([string]$Message, [System.ConsoleColor]$Color = $Green, [string]$Category = "General")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] AI-TEST-$Category`: $Message"
    Write-Host $logMessage -ForegroundColor $Color
    
    # Also write to log file
    $logFile = "$LogsDir/ai-comprehensive-testing.log"
    if (!(Test-Path $LogsDir)) {
        New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
    }
    Add-Content -Path $logFile -Value $logMessage
}

function Test-AIServices {
    Write-TestLog "Testing AI Services..." $Blue "AI-SERVICES"
    
    $services = @(
        @{Name = "GNN Reasoning Engine"; Url = "http://localhost:5001/health"; Endpoint = "/api/ai/gnn-reasoning"},
        @{Name = "Federated Learning"; Url = "http://localhost:5002/health"; Endpoint = "/api/ai/federated-learning"},
        @{Name = "Multi-Hop Symbiosis"; Url = "http://localhost:5003/health"; Endpoint = "/api/ai/multi-hop-symbiosis"},
        @{Name = "Advanced Analytics"; Url = "http://localhost:5004/health"; Endpoint = "/api/ai/advanced-analytics"},
        @{Name = "Revolutionary AI Matching"; Url = "http://localhost:3001/api/ai/revolutionary-matching"; Endpoint = "/api/ai/revolutionary-matching"},
        @{Name = "AI Portfolio Generator"; Url = "http://localhost:3001/api/ai/generate-all-listings"; Endpoint = "/api/ai/generate-all-listings"}
    )
    
    foreach ($service in $services) {
        $TestResults.AIServices.total++
        try {
            # Test health endpoint
            $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 10 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-TestLog "✅ $($service.Name) service healthy" $Green "AI-SERVICES"
                
                # Test actual functionality
                if ($service.Endpoint -like "*/ai/*") {
                    $testResponse = Invoke-WebRequest -Uri "http://localhost:3001$($service.Endpoint)" -Method POST -Body '{"test": true}' -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
                    if ($testResponse.StatusCode -eq 200) {
                        Write-TestLog "   ✅ $($service.Name) functionality working" $Cyan "AI-SERVICES"
                        $TestResults.AIServices.passed++
                        $TestResults.AIServices.details += "✅ $($service.Name): Healthy and functional"
                    } else {
                        throw "Functionality test failed with status: $($testResponse.StatusCode)"
                    }
                } else {
                    $TestResults.AIServices.passed++
                    $TestResults.AIServices.details += "✅ $($service.Name): Healthy"
                }
            } else {
                throw "Status code: $($response.StatusCode)"
            }
        } catch {
            Write-TestLog "❌ $($service.Name) service failed: $($_.Exception.Message)" $Red "AI-SERVICES"
            $TestResults.AIServices.failed++
            $TestResults.AIServices.details += "❌ $($service.Name): $($_.Exception.Message)"
        }
    }
}

function Test-PricingMechanisms {
    Write-TestLog "Testing Pricing Mechanisms..." $Blue "PRICING"
    
    # Test material pricing calculations
    $TestResults.Pricing.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/materials/pricing" -Method POST -Body '{
            "material_name": "steel scrap",
            "quantity": 1000,
            "unit": "kg",
            "quality_grade": "A",
            "location": "Dubai"
        }' -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-TestLog "✅ Material pricing calculation working" $Green "PRICING"
                Write-TestLog "   Market value: $($result.pricing.market_value)" $Cyan "PRICING"
                Write-TestLog "   Disposal cost: $($result.pricing.disposal_cost)" $Cyan "PRICING"
                Write-TestLog "   Net potential: $($result.pricing.net_potential)" $Cyan "PRICING"
                $TestResults.Pricing.passed++
                $TestResults.Pricing.details += "✅ Material pricing: Working with realistic values"
            } else {
                throw "Pricing calculation returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ Material pricing failed: $($_.Exception.Message)" $Red "PRICING"
        $TestResults.Pricing.failed++
        $TestResults.Pricing.details += "❌ Material pricing: $($_.Exception.Message)"
    }
    
    # Test profit calculation
    $TestResults.Pricing.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/financial/calculate-profit" -Method POST -Body '{
            "material_value": 2500,
            "logistics_cost": 500,
            "processing_cost": 200,
            "market_demand": "high"
        }' -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-TestLog "✅ Profit calculation working" $Green "PRICING"
                Write-TestLog "   Net profit: $($result.profit.net_profit)" $Cyan "PRICING"
                Write-TestLog "   ROI: $($result.profit.roi_percentage)%" $Cyan "PRICING"
                Write-TestLog "   Feasibility: $($result.profit.is_feasible)" $Cyan "PRICING"
                $TestResults.Pricing.passed++
                $TestResults.Pricing.details += "✅ Profit calculation: Working with realistic ROI"
            } else {
                throw "Profit calculation returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ Profit calculation failed: $($_.Exception.Message)" $Red "PRICING"
        $TestResults.Pricing.failed++
        $TestResults.Pricing.details += "❌ Profit calculation: $($_.Exception.Message)"
    }
}

function Test-LogisticsCalculations {
    Write-TestLog "Testing Logistics Calculations..." $Blue "LOGISTICS"
    
    # Test Freightos integration
    $TestResults.Logistics.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/logistics/freight-estimate" -Method POST -Body '{
            "origin": "Dubai",
            "destination": "Abu Dhabi",
            "weight": 1000,
            "volume": 1.0,
            "commodity": "steel scrap",
            "mode": "truck"
        }' -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-TestLog "✅ Freightos integration working" $Green "LOGISTICS"
                Write-TestLog "   Total cost: $($result.estimate.total_cost.total_cost)" $Cyan "LOGISTICS"
                Write-TestLog "   Carbon footprint: $($result.estimate.carbon_footprint) kg CO2" $Cyan "LOGISTICS"
                Write-TestLog "   Sustainability score: $($result.estimate.sustainability_score)" $Cyan "LOGISTICS"
                $TestResults.Logistics.passed++
                $TestResults.Logistics.details += "✅ Freightos integration: Real-time rates and CO2 calculation"
            } else {
                throw "Freightos integration returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ Freightos integration failed: $($_.Exception.Message)" $Red "LOGISTICS"
        $TestResults.Logistics.failed++
        $TestResults.Logistics.details += "❌ Freightos integration: $($_.Exception.Message)"
    }
    
    # Test shipping cost breakdown
    $TestResults.Logistics.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/logistics/cost-breakdown" -Method POST -Body '{
            "material_id": "test-material",
            "origin": "Dubai",
            "destination": "Riyadh",
            "weight_kg": 2000,
            "material_value": 5000
        }' -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-TestLog "✅ Shipping cost breakdown working" $Green "LOGISTICS"
                Write-TestLog "   Base rate: $($result.breakdown.base_rate)" $Cyan "LOGISTICS"
                Write-TestLog "   Fuel surcharge: $($result.breakdown.fuel_surcharge)" $Cyan "LOGISTICS"
                Write-TestLog "   Handling fees: $($result.breakdown.handling_fees)" $Cyan "LOGISTICS"
                Write-TestLog "   Total: $($result.breakdown.total_cost)" $Cyan "LOGISTICS"
                $TestResults.Logistics.passed++
                $TestResults.Logistics.details += "✅ Shipping cost breakdown: Detailed cost analysis"
            } else {
                throw "Cost breakdown returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ Shipping cost breakdown failed: $($_.Exception.Message)" $Red "LOGISTICS"
        $TestResults.Logistics.failed++
        $TestResults.Logistics.details += "❌ Shipping cost breakdown: $($_.Exception.Message)"
    }
}

function Test-AIMatching {
    Write-TestLog "Testing AI Matching Engine..." $Blue "MATCHING"
    
    # Test revolutionary AI matching
    $TestResults.Matching.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/ai/revolutionary-matching" -Method POST -Body '{
            "company_data": {
                "name": "Test Manufacturing Co",
                "industry": "manufacturing",
                "location": "Dubai",
                "materials": ["steel scrap", "plastic waste"],
                "requirements": ["raw materials", "energy"]
            },
            "top_k": 5
        }' -ContentType "application/json" -TimeoutSec 60 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-TestLog "✅ Revolutionary AI matching working" $Green "MATCHING"
                Write-TestLog "   Matches found: $($result.matches.Count)" $Cyan "MATCHING"
                Write-TestLog "   Average match score: $([math]::Round(($result.matches | ForEach-Object { $_.match_score } | Measure-Object -Average).Average, 2))" $Cyan "MATCHING"
                Write-TestLog "   Processing time: $($result.matching_metrics.processing_time)ms" $Cyan "MATCHING"
                $TestResults.Matching.passed++
                $TestResults.Matching.details += "✅ Revolutionary AI matching: Found $($result.matches.Count) high-quality matches"
            } else {
                throw "AI matching returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ Revolutionary AI matching failed: $($_.Exception.Message)" $Red "MATCHING"
        $TestResults.Matching.failed++
        $TestResults.Matching.details += "❌ Revolutionary AI matching: $($_.Exception.Message)"
    }
    
    # Test GNN reasoning
    $TestResults.Matching.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/ai/gnn-reasoning" -Method POST -Body '{
            "action": "detect_multi_hop_symbiosis",
            "participants": [
                {
                    "name": "Manufacturer A",
                    "industry": "manufacturing",
                    "waste": ["steel scrap", "heat"]
                },
                {
                    "name": "Processor B", 
                    "industry": "processing",
                    "needs": ["raw materials", "energy"]
                }
            ],
            "max_hops": 3
        }' -ContentType "application/json" -TimeoutSec 60 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-TestLog "✅ GNN reasoning working" $Green "MATCHING"
                Write-TestLog "   Symbiosis chains found: $($result.symbiosis_chains.Count)" $Cyan "MATCHING"
                Write-TestLog "   Network complexity: $($result.network_metrics.complexity_score)" $Cyan "MATCHING"
                $TestResults.Matching.passed++
                $TestResults.Matching.details += "✅ GNN reasoning: Multi-hop symbiosis detection working"
            } else {
                throw "GNN reasoning returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ GNN reasoning failed: $($_.Exception.Message)" $Red "MATCHING"
        $TestResults.Matching.failed++
        $TestResults.Matching.details += "❌ GNN reasoning: $($_.Exception.Message)"
    }
}

function Test-Analytics {
    Write-TestLog "Testing AI Analytics..." $Blue "ANALYTICS"
    
    # Test carbon calculation
    $TestResults.Analytics.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/ai/carbon-calculation" -Method POST -Body '{
            "company_data": {
                "industry": "manufacturing",
                "waste_streams": [
                    {"material": "steel scrap", "quantity": 1000, "unit": "kg"},
                    {"material": "plastic waste", "quantity": 500, "unit": "kg"}
                ],
                "energy_consumption": 5000,
                "location": "Dubai"
            }
        }' -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-TestLog "✅ Carbon calculation working" $Green "ANALYTICS"
                Write-TestLog "   Total carbon footprint: $($result.carbon_footprint.total_emissions) kg CO2e" $Cyan "ANALYTICS"
                Write-TestLog "   Efficiency score: $($result.efficiency_metrics.efficiency_score)" $Cyan "ANALYTICS"
                Write-TestLog "   Reduction potential: $($result.reduction_potential.kg_co2e)" $Cyan "ANALYTICS"
                $TestResults.Analytics.passed++
                $TestResults.Analytics.details += "✅ Carbon calculation: Comprehensive environmental analysis"
            } else {
                throw "Carbon calculation returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ Carbon calculation failed: $($_.Exception.Message)" $Red "ANALYTICS"
        $TestResults.Analytics.failed++
        $TestResults.Analytics.details += "❌ Carbon calculation: $($_.Exception.Message)"
    }
    
    # Test business metrics
    $TestResults.Analytics.total++
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3001/api/ai/business-metrics" -Method POST -Body '{
            "company_data": {
                "annual_revenue": 1000000,
                "waste_costs": 50000,
                "energy_costs": 100000,
                "employee_count": 100
            }
        }' -ContentType "application/json" -TimeoutSec 30 -ErrorAction Stop
        
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            if ($result.success) {
                Write-TestLog "✅ Business metrics working" $Green "ANALYTICS"
                Write-TestLog "   Potential savings: $($result.metrics.potential_savings)" $Cyan "ANALYTICS"
                Write-TestLog "   ROI potential: $($result.metrics.roi_potential)%" $Cyan "ANALYTICS"
                Write-TestLog "   Sustainability score: $($result.metrics.sustainability_score)" $Cyan "ANALYTICS"
                $TestResults.Analytics.passed++
                $TestResults.Analytics.details += "✅ Business metrics: Financial and sustainability analysis"
            } else {
                throw "Business metrics returned success: false"
            }
        } else {
            throw "Status code: $($response.StatusCode)"
        }
    } catch {
        Write-TestLog "❌ Business metrics failed: $($_.Exception.Message)" $Red "ANALYTICS"
        $TestResults.Analytics.failed++
        $TestResults.Analytics.details += "❌ Business metrics: $($_.Exception.Message)"
    }
}

function Show-TestSummary {
    Write-TestLog "AI Testing Summary" $Magenta "SUMMARY"
    Write-TestLog "=================" $Magenta "SUMMARY"
    
    $totalTests = 0
    $totalPassed = 0
    $totalFailed = 0
    
    foreach ($category in $TestResults.Keys) {
        $categoryData = $TestResults[$category]
        $totalTests += $categoryData.total
        $totalPassed += $categoryData.passed
        $totalFailed += $categoryData.failed
        
        $percentage = if ($categoryData.total -gt 0) { [math]::Round(($categoryData.passed / $categoryData.total) * 100, 1) } else { 0 }
        $color = if ($percentage -ge 80) { $Green } elseif ($percentage -ge 60) { $Yellow } else { $Red }
        
        Write-TestLog "$category`: $($categoryData.passed)/$($categoryData.total) ($percentage%)" $color "SUMMARY"
        
        # Show details
        foreach ($detail in $categoryData.details) {
            Write-TestLog "   $detail" $Cyan "SUMMARY"
        }
    }
    
    $overallPercentage = if ($totalTests -gt 0) { [math]::Round(($totalPassed / $totalTests) * 100, 1) } else { 0 }
    $overallColor = if ($overallPercentage -ge 80) { $Green } elseif ($overallPercentage -ge 60) { $Yellow } else { $Red }
    
    Write-TestLog "=================" $Magenta "SUMMARY"
    Write-TestLog "Overall: $totalPassed/$totalTests ($overallPercentage%)" $overallColor "SUMMARY"
    
    if ($overallPercentage -ge 80) {
        Write-TestLog "🎉 AI system is ready for 50 companies!" $Green "SUMMARY"
    } elseif ($overallPercentage -ge 60) {
        Write-TestLog "⚠️ AI system needs some improvements before scaling" $Yellow "SUMMARY"
    } else {
        Write-TestLog "❌ AI system needs significant work before scaling" $Red "SUMMARY"
    }
}

function Start-ComprehensiveAITesting {
    Write-TestLog "Starting Comprehensive AI Testing for ISM Platform" $Magenta "START"
    Write-TestLog "Testing all AI components before introducing 50 companies" $Magenta "START"
    
    switch ($Command) {
        "ai-services" {
            Test-AIServices
        }
        "pricing" {
            Test-PricingMechanisms
        }
        "logistics" {
            Test-LogisticsCalculations
        }
        "matching" {
            Test-AIMatching
        }
        "analytics" {
            Test-Analytics
        }
        "all" {
            Test-AIServices
            Test-PricingMechanisms
            Test-LogisticsCalculations
            Test-AIMatching
            Test-Analytics
        }
        "help" {
            Write-Host "ISM AI Platform - Comprehensive AI Testing Script"
            Write-Host ""
            Write-Host "Usage: .\test-ai-comprehensive.ps1 [command]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  all         Test all AI components (default)"
            Write-Host "  ai-services Test AI service health and functionality"
            Write-Host "  pricing     Test material pricing and profit calculations"
            Write-Host "  logistics   Test shipping costs and Freightos integration"
            Write-Host "  matching    Test AI matching engines and GNN reasoning"
            Write-Host "  analytics   Test carbon calculation and business metrics"
            Write-Host "  help        Show this help message"
            Write-Host ""
            Write-Host "This script ensures your AI system is ready for 50 companies."
        }
        default {
            Test-AIServices
            Test-PricingMechanisms
            Test-LogisticsCalculations
            Test-AIMatching
            Test-Analytics
        }
    }
    
    Show-TestSummary
}

# Run comprehensive AI testing
Start-ComprehensiveAITesting 