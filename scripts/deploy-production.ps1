# Production Deployment Script for ISM AI Platform
# Handles complete production deployment with SSL, monitoring, and optimization

param(
    [Parameter(Mandatory=$true)]
    [string]$Domain,
    
    [Parameter(Mandatory=$true)]
    [string]$Email,
    
    [Parameter(Position=0)]
    [ValidateSet("deploy", "update", "rollback", "status", "monitor", "help")]
    [string]$Command = "deploy"
)

# Configuration
$ProjectRoot = Get-Location
$DeploymentDir = "deployment"
$LogsDir = "logs"
$BackupDir = "backups"

# Colors for output
$Red = [System.ConsoleColor]::Red
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue

# Logging functions
function Write-DeploymentLog {
    param([string]$Message, [System.ConsoleColor]$Color = $Green)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] DEPLOYMENT: $Message"
    Write-Host $logMessage -ForegroundColor $Color
    
    # Also write to log file
    $logFile = "$LogsDir/deployment.log"
    if (!(Test-Path $LogsDir)) {
        New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
    }
    Add-Content -Path $logFile -Value $logMessage
}

# Check prerequisites
function Test-DeploymentPrerequisites {
    Write-DeploymentLog "Checking deployment prerequisites..." $Blue
    
    $prerequisites = @{
        "Docker" = $false
        "Docker Compose" = $false
        "Domain DNS" = $false
        "Environment Variables" = $false
        "Database" = $false
    }
    
    # Check Docker
    try {
        $dockerVersion = docker --version 2>$null
        if ($dockerVersion) {
            $prerequisites["Docker"] = $true
            Write-DeploymentLog "✅ Docker is available" $Green
        }
    } catch {
        Write-DeploymentLog "❌ Docker not found" $Red
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker-compose --version 2>$null
        if ($composeVersion) {
            $prerequisites["Docker Compose"] = $true
            Write-DeploymentLog "✅ Docker Compose is available" $Green
        }
    } catch {
        Write-DeploymentLog "❌ Docker Compose not found" $Red
    }
    
    # Check domain DNS
    try {
        $dnsResult = Resolve-DnsName $Domain -ErrorAction Stop
        if ($dnsResult) {
            $prerequisites["Domain DNS"] = $true
            Write-DeploymentLog "✅ Domain $Domain resolves to $($dnsResult.IPAddress)" $Green
        }
    } catch {
        Write-DeploymentLog "❌ Domain $Domain does not resolve" $Red
    }
    
    # Check environment variables
    $requiredEnvVars = @(
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY", 
        "SUPABASE_SERVICE_ROLE_KEY",
        "JWT_SECRET",
        "NEXT_GEN_MATERIALS_API_KEY",
        "DEEPSEEK_API_KEY"
    )
    
    $missingEnvVars = @()
    foreach ($var in $requiredEnvVars) {
        if (!(Get-Variable -Name $var -ErrorAction SilentlyContinue) -and !$env:$var) {
            $missingEnvVars += $var
        }
    }
    
    if ($missingEnvVars.Count -eq 0) {
        $prerequisites["Environment Variables"] = $true
        Write-DeploymentLog "✅ All required environment variables are set" $Green
    } else {
        Write-DeploymentLog "❌ Missing environment variables: $($missingEnvVars -join ', ')" $Red
    }
    
    # Check database connectivity
    try {
        $dbCheck = Invoke-WebRequest -Uri "$env:SUPABASE_URL/rest/v1/" -Headers @{"apikey" = $env:SUPABASE_ANON_KEY} -TimeoutSec 10
        if ($dbCheck.StatusCode -eq 200) {
            $prerequisites["Database"] = $true
            Write-DeploymentLog "✅ Database connectivity verified" $Green
        }
    } catch {
        Write-DeploymentLog "❌ Database connectivity failed" $Red
    }
    
    return $prerequisites
}

# Create deployment structure
function New-DeploymentStructure {
    Write-DeploymentLog "Creating deployment directory structure..." $Blue
    
    $directories = @(
        $DeploymentDir,
        $LogsDir,
        $BackupDir,
        "$DeploymentDir/ssl",
        "$DeploymentDir/monitoring",
        "$DeploymentDir/configs"
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-DeploymentLog "Created directory: $dir"
        }
    }
}

# Setup SSL certificates
function Install-SSLCertificates {
    Write-DeploymentLog "Setting up SSL certificates..." $Blue
    
    try {
        $sslScript = "$ProjectRoot/scripts/setup-ssl.ps1"
        if (Test-Path $sslScript) {
            & $sslScript -Domain $Domain -Email $Email install
            Write-DeploymentLog "✅ SSL certificates installed" $Green
        } else {
            Write-DeploymentLog "❌ SSL setup script not found" $Red
            return $false
        }
    } catch {
        Write-DeploymentLog "❌ SSL certificate installation failed: $($_.Exception.Message)" $Red
        return $false
    }
    
    return $true
}

# Setup monitoring stack
function Install-MonitoringStack {
    Write-DeploymentLog "Setting up monitoring stack..." $Blue
    
    try {
        $monitoringScript = "$ProjectRoot/scripts/setup-monitoring.ps1"
        if (Test-Path $monitoringScript) {
            & $monitoringScript install
            Write-DeploymentLog "✅ Monitoring stack installed" $Green
        } else {
            Write-DeploymentLog "❌ Monitoring setup script not found" $Red
            return $false
        }
    } catch {
        Write-DeploymentLog "❌ Monitoring stack installation failed: $($_.Exception.Message)" $Red
        return $false
    }
    
    return $true
}

# Build and deploy application
function Deploy-Application {
    Write-DeploymentLog "Building and deploying application..." $Blue
    
    try {
        # Build frontend
        Write-DeploymentLog "Building frontend..." $Blue
        Set-Location "$ProjectRoot/frontend"
        docker-compose -f ../docker-compose.prod.yml build frontend
        
        # Build backend
        Write-DeploymentLog "Building backend..." $Blue
        Set-Location "$ProjectRoot/backend"
        docker-compose -f ../docker-compose.prod.yml build backend
        
        # Build AI services
        Write-DeploymentLog "Building AI services..." $Blue
        Set-Location "$ProjectRoot/ai_service_flask"
        docker-compose -f ../docker-compose.prod.yml build ai-services
        
        # Deploy all services
        Write-DeploymentLog "Deploying all services..." $Blue
        Set-Location $ProjectRoot
        docker-compose -f docker-compose.prod.yml up -d
        
        # Wait for services to start
        Start-Sleep -Seconds 30
        
        # Check service health
        $healthChecks = @(
            @{Name = "Frontend"; Url = "https://$Domain/health"},
            @{Name = "Backend"; Url = "https://api.$Domain/api/health"},
            @{Name = "AI Services"; Url = "https://api.$Domain/ai/health"}
        )
        
        foreach ($service in $healthChecks) {
            try {
                $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 10 -ErrorAction Stop
                if ($response.StatusCode -eq 200) {
                    Write-DeploymentLog "✅ $($service.Name) is healthy" $Green
                } else {
                    Write-DeploymentLog "⚠️ $($service.Name) returned status $($response.StatusCode)" $Yellow
                }
            } catch {
                Write-DeploymentLog "❌ $($service.Name) health check failed: $($_.Exception.Message)" $Red
            }
        }
        
        Write-DeploymentLog "✅ Application deployment completed" $Green
        return $true
    } catch {
        Write-DeploymentLog "❌ Application deployment failed: $($_.Exception.Message)" $Red
        return $false
    }
}

# Update application
function Update-Application {
    Write-DeploymentLog "Updating application..." $Blue
    
    try {
        # Create backup
        $backupName = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        New-Backup -Name $backupName
        
        # Pull latest changes
        git pull origin main
        
        # Rebuild and redeploy
        Deploy-Application
        
        Write-DeploymentLog "✅ Application update completed" $Green
        return $true
    } catch {
        Write-DeploymentLog "❌ Application update failed: $($_.Exception.Message)" $Red
        return $false
    }
}

# Create backup
function New-Backup {
    param([string]$Name)
    
    Write-DeploymentLog "Creating backup: $Name" $Blue
    
    try {
        $backupPath = "$BackupDir/$Name"
        New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
        
        # Backup configuration files
        Copy-Item -Path "docker-compose.prod.yml" -Destination "$backupPath/" -Force
        Copy-Item -Path "backend/env.example" -Destination "$backupPath/" -Force
        Copy-Item -Path "frontend/nginx.conf" -Destination "$backupPath/" -Force
        
        # Backup database (if possible)
        # This would typically involve database dumps
        
        Write-DeploymentLog "✅ Backup created: $backupPath" $Green
    } catch {
        Write-DeploymentLog "❌ Backup creation failed: $($_.Exception.Message)" $Red
    }
}

# Rollback to previous version
function Rollback-Application {
    Write-DeploymentLog "Rolling back application..." $Blue
    
    try {
        # Get list of backups
        $backups = Get-ChildItem -Path $BackupDir -Directory | Sort-Object LastWriteTime -Descending
        
        if ($backups.Count -eq 0) {
            Write-DeploymentLog "❌ No backups found for rollback" $Red
            return $false
        }
        
        $latestBackup = $backups[0]
        Write-DeploymentLog "Rolling back to: $($latestBackup.Name)" $Blue
        
        # Stop current services
        docker-compose -f docker-compose.prod.yml down
        
        # Restore from backup
        Copy-Item -Path "$BackupDir/$($latestBackup.Name)/*" -Destination $ProjectRoot -Recurse -Force
        
        # Restart services
        docker-compose -f docker-compose.prod.yml up -d
        
        Write-DeploymentLog "✅ Rollback completed" $Green
        return $true
    } catch {
        Write-DeploymentLog "❌ Rollback failed: $($_.Exception.Message)" $Red
        return $false
    }
}

# Check deployment status
function Get-DeploymentStatus {
    Write-DeploymentLog "Checking deployment status..." $Blue
    
    $services = @(
        @{Name = "Frontend"; Container = "ism-frontend"},
        @{Name = "Backend"; Container = "ism-backend"},
        @{Name = "AI Services"; Container = "ism-ai-services"},
        @{Name = "Redis"; Container = "ism-redis"},
        @{Name = "Prometheus"; Container = "ism-prometheus"},
        @{Name = "Grafana"; Container = "ism-grafana"}
    )
    
    foreach ($service in $services) {
        try {
            $containerStatus = docker inspect --format='{{.State.Status}}' $service.Container 2>$null
            if ($containerStatus -eq "running") {
                Write-DeploymentLog "✅ $($service.Name): Running" $Green
            } else {
                Write-DeploymentLog "❌ $($service.Name): $containerStatus" $Red
            }
        } catch {
            Write-DeploymentLog "❌ $($service.Name): Not found" $Red
        }
    }
    
    # Check SSL certificate
    try {
        $sslCheck = Invoke-WebRequest -Uri "https://$Domain" -TimeoutSec 10 -ErrorAction Stop
        if ($sslCheck.StatusCode -eq 200) {
            Write-DeploymentLog "✅ SSL Certificate: Valid" $Green
        }
    } catch {
        Write-DeploymentLog "❌ SSL Certificate: Invalid or missing" $Red
    }
}

# Monitor application performance
function Monitor-Application {
    Write-DeploymentLog "Monitoring application performance..." $Blue
    
    # Check system resources
    $cpuUsage = Get-Counter '\Processor(_Total)\% Processor Time' | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue
    $memoryUsage = Get-Counter '\Memory\Available MBytes' | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue
    
    Write-DeploymentLog "System Resources:" $Blue
    Write-DeploymentLog "  CPU Usage: $([math]::Round($cpuUsage, 2))%" $Blue
    Write-DeploymentLog "  Available Memory: $([math]::Round($memoryUsage, 2)) MB" $Blue
    
    # Check Docker resources
    $dockerStats = docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    Write-DeploymentLog "Docker Container Resources:" $Blue
    Write-DeploymentLog $dockerStats $Blue
    
    # Check application endpoints
    $endpoints = @(
        @{Name = "Frontend"; Url = "https://$Domain"},
        @{Name = "API"; Url = "https://api.$Domain/api/health"},
        @{Name = "Monitoring"; Url = "http://localhost:3001"}
    )
    
    foreach ($endpoint in $endpoints) {
        try {
            $startTime = Get-Date
            $response = Invoke-WebRequest -Uri $endpoint.Url -TimeoutSec 10 -ErrorAction Stop
            $endTime = Get-Date
            $responseTime = ($endTime - $startTime).TotalMilliseconds
            
            Write-DeploymentLog "  $($endpoint.Name): $($response.StatusCode) (${responseTime}ms)" $Green
        } catch {
            Write-DeploymentLog "  $($endpoint.Name): ERROR - $($_.Exception.Message)" $Red
        }
    }
}

# Main deployment function
function Start-ProductionDeployment {
    Write-DeploymentLog "Starting production deployment for domain: $Domain" $Blue
    
    switch ($Command) {
        "deploy" {
            # Check prerequisites
            $prerequisites = Test-DeploymentPrerequisites
            $allMet = $true
            foreach ($prereq in $prerequisites.GetEnumerator()) {
                if (!$prereq.Value) {
                    $allMet = $false
                }
            }
            
            if (!$allMet) {
                Write-DeploymentLog "❌ Prerequisites not met. Please fix the issues above." $Red
                return $false
            }
            
            # Create deployment structure
            New-DeploymentStructure
            
            # Setup SSL certificates
            if (!(Install-SSLCertificates)) {
                Write-DeploymentLog "❌ SSL setup failed" $Red
                return $false
            }
            
            # Setup monitoring
            if (!(Install-MonitoringStack)) {
                Write-DeploymentLog "❌ Monitoring setup failed" $Red
                return $false
            }
            
            # Deploy application
            if (!(Deploy-Application)) {
                Write-DeploymentLog "❌ Application deployment failed" $Red
                return $false
            }
            
            Write-DeploymentLog "🎉 Production deployment completed successfully!" $Green
            Write-DeploymentLog "" $Blue
            Write-DeploymentLog "Your ISM AI Platform is now live at:" $Blue
            Write-DeploymentLog "  - Frontend: https://$Domain" $Blue
            Write-DeploymentLog "  - API: https://api.$Domain" $Blue
            Write-DeploymentLog "  - Monitoring: http://localhost:3001 (admin/admin)" $Blue
            Write-DeploymentLog "" $Blue
            Write-DeploymentLog "Next steps:" $Blue
            Write-DeploymentLog "  1. Configure your domain DNS to point to this server" $Blue
            Write-DeploymentLog "  2. Set up monitoring alerts" $Blue
            Write-DeploymentLog "  3. Import your 50 companies data" $Blue
            Write-DeploymentLog "  4. Start your targeted outreach campaign" $Blue
            
            return $true
        }
        "update" {
            Update-Application
        }
        "rollback" {
            Rollback-Application
        }
        "status" {
            Get-DeploymentStatus
        }
        "monitor" {
            Monitor-Application
        }
        "help" {
            Write-Host "ISM AI Platform Production Deployment Script"
            Write-Host ""
            Write-Host "Usage: .\deploy-production.ps1 -Domain <domain> -Email <email> [command]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  deploy     Complete production deployment (SSL + Monitoring + App)"
            Write-Host "  update     Update existing deployment"
            Write-Host "  rollback   Rollback to previous version"
            Write-Host "  status     Check deployment status"
            Write-Host "  monitor    Monitor application performance"
            Write-Host "  help       Show this help message"
            Write-Host ""
            Write-Host "Example:"
            Write-Host "  .\deploy-production.ps1 -Domain ism.yourdomain.com -Email admin@yourdomain.com deploy"
        }
        default {
            Start-ProductionDeployment -Command "deploy"
        }
    }
}

# Run main function
Start-ProductionDeployment 