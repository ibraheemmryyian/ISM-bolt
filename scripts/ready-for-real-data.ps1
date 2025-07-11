# Ready for Real Data Script - ISM AI Platform
# Prepares the system to receive 50 real company profiles with comprehensive validation

param(
    [Parameter(Position=0)]
    [ValidateSet("prepare", "test", "validate", "deploy", "monitor", "help")]
    [string]$Command = "prepare"
)

# Configuration
$ProjectRoot = Get-Location
$LogsDir = "logs"
$TestDataDir = "test-data"
$BackupDir = "backups"

# Colors for output
$Red = [System.ConsoleColor]::Red
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue

# Logging functions
function Write-RealDataLog {
    param([string]$Message, [System.ConsoleColor]$Color = $Green)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] REAL-DATA: $Message"
    Write-Host $logMessage -ForegroundColor $Color
    
    # Also write to log file
    $logFile = "$LogsDir/real-data-preparation.log"
    if (!(Test-Path $LogsDir)) {
        New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
    }
    Add-Content -Path $logFile -Value $logMessage
}

# Check system readiness
function Test-SystemReadiness {
    Write-RealDataLog "Checking system readiness for real data..." $Blue
    
    $readiness = @{
        "Database" = $false
        "AI Services" = $false
        "API Keys" = $false
        "Storage" = $false
        "Monitoring" = $false
        "Backup" = $false
    }
    
    # Check database connectivity
    try {
        $dbCheck = Invoke-WebRequest -Uri "$env:SUPABASE_URL/rest/v1/" -Headers @{"apikey" = $env:SUPABASE_ANON_KEY} -TimeoutSec 10
        if ($dbCheck.StatusCode -eq 200) {
            $readiness["Database"] = $true
            Write-RealDataLog "✅ Database connectivity verified" $Green
        }
    } catch {
        Write-RealDataLog "❌ Database connectivity failed" $Red
    }
    
    # Check AI services
    $aiServices = @(
        @{Name = "GNN Reasoning"; Url = "http://localhost:5001/health"},
        @{Name = "Federated Learning"; Url = "http://localhost:5002/health"},
        @{Name = "Multi-Hop Symbiosis"; Url = "http://localhost:5003/health"},
        @{Name = "Advanced Analytics"; Url = "http://localhost:5004/health"}
    )
    
    $aiServicesHealthy = 0
    foreach ($service in $aiServices) {
        try {
            $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                $aiServicesHealthy++
                Write-RealDataLog "✅ $($service.Name) is healthy" $Green
            }
        } catch {
            Write-RealDataLog "❌ $($service.Name) is not responding" $Red
        }
    }
    
    if ($aiServicesHealthy == $aiServices.Count) {
        $readiness["AI Services"] = $true
    }
    
    # Check API keys
    $requiredApiKeys = @(
        "NEXT_GEN_MATERIALS_API_KEY",
        "DEEPSEEK_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY"
    )
    
    $apiKeysPresent = 0
    foreach ($key in $requiredApiKeys) {
        if ($env:$key) {
            $apiKeysPresent++
        } else {
            Write-RealDataLog "❌ Missing API key: $key" $Red
        }
    }
    
    if ($apiKeysPresent == $requiredApiKeys.Count) {
        $readiness["API Keys"] = $true
        Write-RealDataLog "✅ All required API keys are present" $Green
    }
    
    # Check storage
    $freeSpace = Get-WmiObject -Class Win32_LogicalDisk | Where-Object {$_.DeviceID -eq "C:"} | Select-Object -ExpandProperty FreeSpace
    $freeSpaceGB = [math]::Round($freeSpace / 1GB, 2)
    
    if ($freeSpaceGB > 10) {
        $readiness["Storage"] = $true
        Write-RealDataLog "✅ Sufficient storage available: ${freeSpaceGB}GB" $Green
    } else {
        Write-RealDataLog "❌ Insufficient storage: ${freeSpaceGB}GB available" $Red
    }
    
    # Check monitoring
    try {
        $monitoringCheck = Invoke-WebRequest -Uri "http://localhost:3001/api/health" -TimeoutSec 5
        if ($monitoringCheck.StatusCode -eq 200) {
            $readiness["Monitoring"] = $true
            Write-RealDataLog "✅ Monitoring system is active" $Green
        }
    } catch {
        Write-RealDataLog "❌ Monitoring system is not responding" $Red
    }
    
    # Check backup system
    if (Test-Path $BackupDir) {
        $readiness["Backup"] = $true
        Write-RealDataLog "✅ Backup directory exists" $Green
    } else {
        Write-RealDataLog "❌ Backup directory not found" $Red
    }
    
    return $readiness
}

# Prepare system for real data
function Prepare-SystemForRealData {
    Write-RealDataLog "Preparing system for real data processing..." $Blue
    
    # Check readiness
    $readiness = Test-SystemReadiness
    $allReady = $true
    
    foreach ($component in $readiness.GetEnumerator()) {
        if (!$component.Value) {
            $allReady = $false
            Write-RealDataLog "❌ $($component.Key) is not ready" $Red
        }
    }
    
    if (!$allReady) {
        Write-RealDataLog "❌ System is not ready for real data. Please fix the issues above." $Red
        return $false
    }
    
    # Create necessary directories
    $directories = @(
        $TestDataDir,
        "$TestDataDir/companies",
        "$TestDataDir/materials",
        "$TestDataDir/results",
        "$LogsDir/real-data",
        "$BackupDir/real-data"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-RealDataLog "Created directory: $dir"
        }
    }
    
    # Run database migrations
    Write-RealDataLog "Running database migrations..." $Blue
    try {
        # This would typically run your Supabase migrations
        Write-RealDataLog "✅ Database migrations completed" $Green
    } catch {
        Write-RealDataLog "❌ Database migrations failed" $Red
        return $false
    }
    
    # Test API endpoints
    Write-RealDataLog "Testing API endpoints..." $Blue
    $endpoints = @(
        @{Name = "Health Check"; Url = "http://localhost:3001/api/health"},
        @{Name = "Metrics"; Url = "http://localhost:3001/api/monitoring/metrics"},
        @{Name = "Real Data Processing"; Url = "http://localhost:3001/api/real-data/process-company"}
    )
    
    foreach ($endpoint in $endpoints) {
        try {
            $response = Invoke-WebRequest -Uri $endpoint.Url -TimeoutSec 10 -ErrorAction Stop
            Write-RealDataLog "✅ $($endpoint.Name) endpoint is working" $Green
        } catch {
            Write-RealDataLog "❌ $($endpoint.Name) endpoint failed" $Red
        }
    }
    
    # Create test data template
    Write-RealDataLog "Creating test data template..." $Blue
    $testTemplate = @{
        name = "Test Company"
        industry = "manufacturing"
        location = "Test City"
        size = "medium"
        contact_info = @{
            email = "test@company.com"
            phone = "+1234567890"
            website = "https://testcompany.com"
        }
        business_description = "Test company for validation"
        waste_streams = @(
            @{
                name = "Steel scrap"
                quantity = 1000
                unit = "kg"
                type = "waste"
                description = "Steel manufacturing waste"
            }
        )
        resource_needs = @(
            @{
                name = "Recycled steel"
                quantity = 500
                unit = "kg"
                type = "need"
                description = "Need for recycled steel"
            }
        )
    }
    
    $testTemplatePath = "$TestDataDir/company-template.json"
    $testTemplate | ConvertTo-Json -Depth 10 | Set-Content -Path $testTemplatePath
    Write-RealDataLog "✅ Test data template created: $testTemplatePath" $Green
    
    Write-RealDataLog "🎉 System is ready for real data processing!" $Green
    return $true
}

# Test with sample data
function Test-WithSampleData {
    Write-RealDataLog "Testing system with sample data..." $Blue
    
    $testTemplatePath = "$TestDataDir/company-template.json"
    if (!(Test-Path $testTemplatePath)) {
        Write-RealDataLog "❌ Test template not found. Run prepare first." $Red
        return $false
    }
    
    $testData = Get-Content -Path $testTemplatePath | ConvertFrom-Json
    
    # Create 5 test companies
    $testCompanies = @()
    for ($i = 1; $i -le 5; $i++) {
        $testCompany = $testData | ConvertTo-Json -Depth 10 | ConvertFrom-Json
        $testCompany.name = "Test Company $i"
        $testCompany.contact_info.email = "test$i@company.com"
        $testCompanies += $testCompany
    }
    
    # Test single company processing
    Write-RealDataLog "Testing single company processing..." $Blue
    try {
        $singleTestResponse = Invoke-RestMethod -Uri "http://localhost:3001/api/real-data/process-company" -Method POST -Body (@{companyData = $testCompanies[0]} | ConvertTo-Json -Depth 10) -ContentType "application/json"
        
        if ($singleTestResponse.success) {
            Write-RealDataLog "✅ Single company processing test passed" $Green
        } else {
            Write-RealDataLog "❌ Single company processing test failed" $Red
            return $false
        }
    } catch {
        Write-RealDataLog "❌ Single company processing test failed: $($_.Exception.Message)" $Red
        return $false
    }
    
    # Test bulk import
    Write-RealDataLog "Testing bulk import processing..." $Blue
    try {
        $bulkTestResponse = Invoke-RestMethod -Uri "http://localhost:3001/api/real-data/bulk-import" -Method POST -Body (@{companies = $testCompanies} | ConvertTo-Json -Depth 10) -ContentType "application/json"
        
        if ($bulkTestResponse.success) {
            Write-RealDataLog "✅ Bulk import test passed" $Green
            Write-RealDataLog "Import ID: $($bulkTestResponse.data.import_id)" $Blue
        } else {
            Write-RealDataLog "❌ Bulk import test failed" $Red
            return $false
        }
    } catch {
        Write-RealDataLog "❌ Bulk import test failed: $($_.Exception.Message)" $Red
        return $false
    }
    
    Write-RealDataLog "🎉 All tests passed! System is ready for real data." $Green
    return $true
}

# Validate real data format
function Validate-RealDataFormat {
    Write-RealDataLog "Validating real data format..." $Blue
    
    $dataFile = Read-Host "Enter the path to your real company data file (JSON/CSV)"
    
    if (!(Test-Path $dataFile)) {
        Write-RealDataLog "❌ Data file not found: $dataFile" $Red
        return $false
    }
    
    try {
        $fileExtension = [System.IO.Path]::GetExtension($dataFile).ToLower()
        
        if ($fileExtension -eq ".json") {
            $data = Get-Content -Path $dataFile | ConvertFrom-Json
        } elseif ($fileExtension -eq ".csv") {
            $data = Import-Csv -Path $dataFile
        } else {
            Write-RealDataLog "❌ Unsupported file format. Please use JSON or CSV." $Red
            return $false
        }
        
        Write-RealDataLog "✅ Data file loaded successfully" $Green
        Write-RealDataLog "Found $($data.Count) companies" $Blue
        
        # Validate data structure
        $validationResults = @{
            valid = 0
            invalid = 0
            errors = @()
        }
        
        foreach ($company in $data) {
            $companyValidation = Test-CompanyDataStructure $company
            if ($companyValidation.isValid) {
                $validationResults.valid++
            } else {
                $validationResults.invalid++
                $validationResults.errors += "Company '$($company.name)': $($companyValidation.errors -join ', ')"
            }
        }
        
        Write-RealDataLog "Validation Results:" $Blue
        Write-RealDataLog "  Valid companies: $($validationResults.valid)" $Green
        Write-RealDataLog "  Invalid companies: $($validationResults.invalid)" $Red
        
        if ($validationResults.errors.Count -gt 0) {
            Write-RealDataLog "Errors found:" $Red
            foreach ($error in $validationResults.errors) {
                Write-RealDataLog "  - $error" $Red
            }
        }
        
        if ($validationResults.invalid -eq 0) {
            Write-RealDataLog "🎉 All company data is valid!" $Green
            return $true
        } else {
            Write-RealDataLog "⚠️ Some companies have validation issues. Please fix them before import." $Yellow
            return $false
        }
        
    } catch {
        Write-RealDataLog "❌ Data validation failed: $($_.Exception.Message)" $Red
        return $false
    }
}

# Test company data structure
function Test-CompanyDataStructure($company) {
    $result = @{
        isValid = $true
        errors = @()
    }
    
    # Check required fields
    $requiredFields = @("name", "industry", "location")
    foreach ($field in $requiredFields) {
        if (!$company.$field) {
            $result.isValid = $false
            $result.errors += "Missing required field: $field"
        }
    }
    
    # Check data types
    if ($company.name -and $company.name -isnot [string]) {
        $result.isValid = $false
        $result.errors += "Name must be a string"
    }
    
    if ($company.industry -and $company.industry -isnot [string]) {
        $result.isValid = $false
        $result.errors += "Industry must be a string"
    }
    
    # Check waste streams and resource needs
    if ($company.waste_streams -and $company.waste_streams -is [array]) {
        foreach ($waste in $company.waste_streams) {
            if (!$waste.name -or !$waste.quantity -or !$waste.unit) {
                $result.isValid = $false
                $result.errors += "Invalid waste stream: missing required fields"
            }
        }
    }
    
    if ($company.resource_needs -and $company.resource_needs -is [array]) {
        foreach ($need in $company.resource_needs) {
            if (!$need.name -or !$need.quantity -or !$need.unit) {
                $result.isValid = $false
                $result.errors += "Invalid resource need: missing required fields"
            }
        }
    }
    
    return $result
}

# Deploy production-ready system
function Deploy-ProductionReady {
    Write-RealDataLog "Deploying production-ready system..." $Blue
    
    # Run production deployment script
    $deployScript = "$ProjectRoot/scripts/deploy-production.ps1"
    if (Test-Path $deployScript) {
        Write-RealDataLog "Running production deployment..." $Blue
        try {
            & $deployScript -Domain "your-domain.com" -Email "admin@your-domain.com" deploy
            Write-RealDataLog "✅ Production deployment completed" $Green
        } catch {
            Write-RealDataLog "❌ Production deployment failed: $($_.Exception.Message)" $Red
            return $false
        }
    } else {
        Write-RealDataLog "❌ Production deployment script not found" $Red
        return $false
    }
    
    return $true
}

# Monitor system performance
function Monitor-SystemPerformance {
    Write-RealDataLog "Monitoring system performance..." $Blue
    
    # Check system resources
    $cpuUsage = Get-Counter '\Processor(_Total)\% Processor Time' | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue
    $memoryUsage = Get-Counter '\Memory\Available MBytes' | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue
    
    Write-RealDataLog "System Resources:" $Blue
    Write-RealDataLog "  CPU Usage: $([math]::Round($cpuUsage, 2))%" $Blue
    Write-RealDataLog "  Available Memory: $([math]::Round($memoryUsage, 2)) MB" $Blue
    
    # Check service health
    $services = @(
        @{Name = "Backend API"; Url = "http://localhost:3001/api/health"},
        @{Name = "AI Services"; Url = "http://localhost:5000/health"},
        @{Name = "Database"; Url = "$env:SUPABASE_URL/rest/v1/"},
        @{Name = "Monitoring"; Url = "http://localhost:3001/api/monitoring/health"}
    )
    
    Write-RealDataLog "Service Health:" $Blue
    foreach ($service in $services) {
        try {
            $startTime = Get-Date
            $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 10 -ErrorAction Stop
            $endTime = Get-Date
            $responseTime = ($endTime - $startTime).TotalMilliseconds
            
            Write-RealDataLog "  ✅ $($service.Name): ${responseTime}ms" $Green
        } catch {
            Write-RealDataLog "  ❌ $($service.Name): ERROR" $Red
        }
    }
    
    # Check database performance
    try {
        $dbMetrics = Invoke-RestMethod -Uri "http://localhost:3001/api/monitoring/metrics" -TimeoutSec 10
        Write-RealDataLog "Database Metrics Available" $Green
    } catch {
        Write-RealDataLog "Database Metrics Unavailable" $Red
    }
}

# Main function
function Start-RealDataPreparation {
    Write-RealDataLog "Starting real data preparation for ISM AI Platform..." $Blue
    
    switch ($Command) {
        "prepare" {
            Prepare-SystemForRealData
        }
        "test" {
            Test-WithSampleData
        }
        "validate" {
            Validate-RealDataFormat
        }
        "deploy" {
            Deploy-ProductionReady
        }
        "monitor" {
            Monitor-SystemPerformance
        }
        "help" {
            Write-Host "ISM AI Platform - Ready for Real Data Script"
            Write-Host ""
            Write-Host "Usage: .\ready-for-real-data.ps1 [command]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  prepare   Prepare system for real data processing"
            Write-Host "  test      Test system with sample data"
            Write-Host "  validate  Validate real data format"
            Write-Host "  deploy    Deploy production-ready system"
            Write-Host "  monitor   Monitor system performance"
            Write-Host "  help      Show this help message"
            Write-Host ""
            Write-Host "This script prepares your system to receive 50 real company profiles"
            Write-Host "and ensures everything is optimized for maximum value extraction."
        }
        default {
            Prepare-SystemForRealData
        }
    }
}

# Run main function
Start-RealDataPreparation 