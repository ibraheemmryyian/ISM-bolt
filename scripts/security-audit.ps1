# Security Audit Script for ISM AI Platform
# Comprehensive security review and vulnerability assessment

param(
    [Parameter(Position=0)]
    [ValidateSet("", "full", "quick", "dependencies", "code", "infrastructure", "report", "fix", "help")]
    [string]$Command = "full"
)

# Configuration
$SecurityIssues = @()
$Recommendations = @()
$Vulnerabilities = @()
$ReportFile = "security-audit-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').html"

# Colors for output
$Red = [System.ConsoleColor]::Red
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue

# Logging functions
function Write-SecurityLog {
    param([string]$Message, [System.ConsoleColor]$Color = $Green, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Level`: $Message"
    Write-Host $logMessage -ForegroundColor $Color
}

# Check for security tools
function Test-SecurityTools {
    Write-SecurityLog "Checking security tools availability..." $Blue
    
    $tools = @{
        "npm audit" = $false
        "snyk" = $false
        "trivy" = $false
        "bandit" = $false
        "safety" = $false
    }
    
    # Check npm audit
    try {
        $null = npm audit --version 2>$null
        $tools["npm audit"] = $true
        Write-SecurityLog "✅ npm audit available"
    }
    catch {
        Write-SecurityLog "❌ npm audit not available" $Yellow
    }
    
    # Check Snyk
    try {
        $null = snyk --version 2>$null
        $tools["snyk"] = $true
        Write-SecurityLog "✅ Snyk available"
    }
    catch {
        Write-SecurityLog "❌ Snyk not available" $Yellow
    }
    
    # Check Trivy
    try {
        $null = trivy --version 2>$null
        $tools["trivy"] = $true
        Write-SecurityLog "✅ Trivy available"
    }
    catch {
        Write-SecurityLog "❌ Trivy not available" $Yellow
    }
    
    return $tools
}

# Dependency security audit
function Test-DependencySecurity {
    Write-SecurityLog "Auditing dependencies for vulnerabilities..." $Blue
    
    # Backend dependencies
    if (Test-Path "backend/package.json") {
        Write-SecurityLog "Checking backend dependencies..."
        Set-Location backend
        
        try {
            $auditResult = npm audit --json 2>$null
            if ($auditResult) {
                $auditData = $auditResult | ConvertFrom-Json
                
                if ($auditData.metadata.vulnerabilities) {
                    $auditData.metadata.vulnerabilities
                    $vulnCount = $auditData.metadata.vulnerabilities
                    Write-SecurityLog "Found $($vulnCount.critical) critical, $($vulnCount.high) high, $($vulnCount.moderate) moderate vulnerabilities" $Red
                    
                    $script:Vulnerabilities += @{
                        Type = "npm"
                        Component = "backend"
                        Critical = $vulnCount.critical
                        High = $vulnCount.high
                        Moderate = $vulnCount.moderate
                        Low = $vulnCount.low
                    }
                }
                else {
                    Write-SecurityLog "✅ No vulnerabilities found in backend dependencies" $Green
                }
            }
        }
        catch {
            Write-SecurityLog "⚠️ Could not run npm audit: $($_.Exception.Message)" $Yellow
        }
        
        Set-Location ..
    }
    
    # Frontend dependencies
    if (Test-Path "frontend/package.json") {
        Write-SecurityLog "Checking frontend dependencies..."
        Set-Location frontend
        
        try {
            $auditResult = npm audit --json 2>$null
            if ($auditResult) {
                $auditData = $auditResult | ConvertFrom-Json
                
                if ($auditData.metadata.vulnerabilities) {
                    $vulnCount = $auditData.metadata.vulnerabilities
                    Write-SecurityLog "Found $($vulnCount.critical) critical, $($vulnCount.high) high, $($vulnCount.moderate) moderate vulnerabilities" $Red
                    
                    $script:Vulnerabilities += @{
                        Type = "npm"
                        Component = "frontend"
                        Critical = $vulnCount.critical
                        High = $vulnCount.high
                        Moderate = $vulnCount.moderate
                        Low = $vulnCount.low
                    }
                }
                else {
                    Write-SecurityLog "✅ No vulnerabilities found in frontend dependencies" $Green
                }
            }
        }
        catch {
            Write-SecurityLog "⚠️ Could not run npm audit: $($_.Exception.Message)" $Yellow
        }
        
        Set-Location ..
    }
    
    # Python dependencies
    if (Test-Path "requirements.txt") {
        Write-SecurityLog "Checking Python dependencies..."
        
        try {
            # Check if safety is available
            $safetyResult = safety check --json 2>$null
            if ($safetyResult) {
                $safetyData = $safetyResult | ConvertFrom-Json
                
                if ($safetyData.length -gt 0) {
                    Write-SecurityLog "Found $($safetyData.length) Python security issues" $Red
                    
                    $script:Vulnerabilities += @{
                        Type = "python"
                        Component = "ai-services"
                        Issues = $safetyData.length
                        Details = $safetyData
                    }
                }
                else {
                    Write-SecurityLog "✅ No vulnerabilities found in Python dependencies" $Green
                }
            }
        }
        catch {
            Write-SecurityLog "⚠️ Could not run safety check: $($_.Exception.Message)" $Yellow
        }
    }
}

# Code security audit
function Test-CodeSecurity {
    Write-SecurityLog "Auditing code for security issues..." $Blue
    
    # Check for hardcoded secrets
    $secretPatterns = @(
        "password.*=.*['\""][^'\""]+['\""]",
        "secret.*=.*['\""][^'\""]+['\""]",
        "api_key.*=.*['\""][^'\""]+['\""]",
        "token.*=.*['\""][^'\""]+['\""]",
        "private_key.*=.*['\""][^'\""]+['\""]"
    )
    
    $codeFiles = Get-ChildItem -Recurse -Include "*.js", "*.ts", "*.py", "*.json", "*.yml", "*.yaml" | Where-Object { $_.FullName -notmatch "node_modules|\.git|dist|build" }
    
    foreach ($file in $codeFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content) {
            foreach ($pattern in $secretPatterns) {
                if ($content -match $pattern) {
                    $script:SecurityIssues += @{
                        Type = "Hardcoded Secret"
                        File = $file.FullName
                        Pattern = $pattern
                        Severity = "Critical"
                        Description = "Potential hardcoded secret found"
                    }
                    Write-SecurityLog "❌ Hardcoded secret found in $($file.Name)" $Red
                }
            }
        }
    }
    
    # Check for SQL injection vulnerabilities
    $sqlPatterns = @(
        "SELECT.*\+.*\$",
        "INSERT.*\+.*\$",
        "UPDATE.*\+.*\$",
        "DELETE.*\+.*\$"
    )
    
    foreach ($file in $codeFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content) {
            foreach ($pattern in $sqlPatterns) {
                if ($content -match $pattern) {
                    $script:SecurityIssues += @{
                        Type = "SQL Injection"
                        File = $file.FullName
                        Pattern = $pattern
                        Severity = "Critical"
                        Description = "Potential SQL injection vulnerability"
                    }
                    Write-SecurityLog "❌ Potential SQL injection in $($file.Name)" $Red
                }
            }
        }
    }
    
    # Check for XSS vulnerabilities
    $xssPatterns = @(
        "innerHTML.*=.*\$",
        "outerHTML.*=.*\$",
        "document\.write.*\$"
    )
    
    foreach ($file in $codeFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content) {
            foreach ($pattern in $xssPatterns) {
                if ($content -match $pattern) {
                    $script:SecurityIssues += @{
                        Type = "XSS"
                        File = $file.FullName
                        Pattern = $pattern
                        Severity = "High"
                        Description = "Potential XSS vulnerability"
                    }
                    Write-SecurityLog "❌ Potential XSS in $($file.Name)" $Red
                }
            }
        }
    }
}

# Infrastructure security audit
function Test-InfrastructureSecurity {
    Write-SecurityLog "Auditing infrastructure security..." $Blue
    
    # Check Docker configurations
    $dockerFiles = Get-ChildItem -Recurse -Include "Dockerfile", "docker-compose*.yml" | Where-Object { $_.FullName -notmatch "node_modules|\.git" }
    
    foreach ($file in $dockerFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content) {
            # Check for root user
            if ($content -match "USER root") {
                $script:SecurityIssues += @{
                    Type = "Docker Security"
                    File = $file.FullName
                    Issue = "Running as root user"
                    Severity = "High"
                    Description = "Container should not run as root"
                }
                Write-SecurityLog "❌ Container running as root in $($file.Name)" $Red
            }
            
            # Check for exposed ports
            if ($content -match "EXPOSE.*80|EXPOSE.*22|EXPOSE.*21") {
                $script:SecurityIssues += @{
                    Type = "Docker Security"
                    File = $file.FullName
                    Issue = "Exposed sensitive ports"
                    Severity = "Medium"
                    Description = "Avoid exposing ports 80, 22, 21 directly"
                }
                Write-SecurityLog "⚠️ Sensitive ports exposed in $($file.Name)" $Yellow
            }
        }
    }
    
    # Check Kubernetes configurations
    $k8sFiles = Get-ChildItem -Recurse -Include "*.yaml", "*.yml" | Where-Object { $_.FullName -match "k8s" -and $_.FullName -notmatch "node_modules|\.git" }
    
    foreach ($file in $k8sFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content) {
            # Check for privileged containers
            if ($content -match "privileged:\s*true") {
                $script:SecurityIssues += @{
                    Type = "Kubernetes Security"
                    File = $file.FullName
                    Issue = "Privileged container"
                    Severity = "Critical"
                    Description = "Container running in privileged mode"
                }
                Write-SecurityLog "❌ Privileged container in $($file.Name)" $Red
            }
            
            # Check for host network
            if ($content -match "hostNetwork:\s*true") {
                $script:SecurityIssues += @{
                    Type = "Kubernetes Security"
                    File = $file.FullName
                    Issue = "Host network access"
                    Severity = "High"
                    Description = "Pod using host network"
                }
                Write-SecurityLog "❌ Host network access in $($file.Name)" $Red
            }
        }
    }
}

# Environment security audit
function Test-EnvironmentSecurity {
    Write-SecurityLog "Auditing environment security..." $Blue
    
    # Check environment files
    $envFiles = Get-ChildItem -Recurse -Include ".env*" | Where-Object { $_.FullName -notmatch "node_modules|\.git" }
    
    foreach ($file in $envFiles) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content) {
            # Check for weak secrets
            $weakPatterns = @(
                "password.*=.*(123|admin|password|changeme)",
                "secret.*=.*(123|admin|password|changeme)",
                "key.*=.*(123|admin|password|changeme)"
            )
            
            foreach ($pattern in $weakPatterns) {
                if ($content -match $pattern) {
                    $script:SecurityIssues += @{
                        Type = "Weak Secret"
                        File = $file.FullName
                        Pattern = $pattern
                        Severity = "High"
                        Description = "Weak secret detected"
                    }
                    Write-SecurityLog "❌ Weak secret in $($file.Name)" $Red
                }
            }
            
            # Check for production secrets in development
            if ($file.Name -eq ".env" -and $content -match "production") {
                $script:SecurityIssues += @{
                    Type = "Environment Mixing"
                    File = $file.FullName
                    Issue = "Production settings in development"
                    Severity = "Medium"
                    Description = "Production configuration in development environment"
                }
                Write-SecurityLog "⚠️ Production settings in development $($file.Name)" $Yellow
            }
        }
    }
}

# Generate security recommendations
function Get-SecurityRecommendations {
    Write-SecurityLog "Generating security recommendations..." $Blue
    
    $script:Recommendations = @(
        "Implement automated dependency scanning in CI/CD pipeline",
        "Use secrets management (AWS Secrets Manager, Azure Key Vault)",
        "Enable container scanning with Trivy or similar tools",
        "Implement network policies in Kubernetes",
        "Use non-root users in containers",
        "Enable pod security policies",
        "Implement proper RBAC in Kubernetes",
        "Use HTTPS everywhere",
        "Implement rate limiting",
        "Enable audit logging",
        "Use security headers (Helmet.js)",
        "Implement input validation and sanitization",
        "Use prepared statements for database queries",
        "Enable CORS properly",
        "Implement proper session management",
        "Use strong password policies",
        "Enable multi-factor authentication",
        "Implement proper error handling (no information disclosure)",
        "Use environment-specific configurations",
        "Regular security training for developers"
    )
}

# Generate security report
function New-SecurityReport {
    param([string]$OutputPath = $ReportFile)
    
    Write-SecurityLog "Generating security report: $OutputPath" $Blue
    
    $report = @"
<!DOCTYPE html>
<html>
<head>
    <title>ISM AI Platform Security Audit Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .critical { color: #d32f2f; }
        .high { color: #f57c00; }
        .medium { color: #fbc02d; }
        .low { color: #388e3c; }
        .issue { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
        .critical .issue { border-left-color: #d32f2f; }
        .high .issue { border-left-color: #f57c00; }
        .medium .issue { border-left-color: #fbc02d; }
        .low .issue { border-left-color: #388e3c; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ISM AI Platform Security Audit Report</h1>
        <p>Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")</p>
        <p>Audit Type: $Command</p>
    </div>

    <h2>Executive Summary</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Count</th>
        </tr>
        <tr>
            <td>Critical Issues</td>
            <td class="critical">$($SecurityIssues | Where-Object { $_.Severity -eq "Critical" } | Measure-Object | Select-Object -ExpandProperty Count)</td>
        </tr>
        <tr>
            <td>High Issues</td>
            <td class="high">$($SecurityIssues | Where-Object { $_.Severity -eq "High" } | Measure-Object | Select-Object -ExpandProperty Count)</td>
        </tr>
        <tr>
            <td>Medium Issues</td>
            <td class="medium">$($SecurityIssues | Where-Object { $_.Severity -eq "Medium" } | Measure-Object | Select-Object -ExpandProperty Count)</td>
        </tr>
        <tr>
            <td>Low Issues</td>
            <td class="low">$($SecurityIssues | Where-Object { $_.Severity -eq "Low" } | Measure-Object | Select-Object -ExpandProperty Count)</td>
        </tr>
        <tr>
            <td>Vulnerabilities</td>
            <td>$($Vulnerabilities.Count)</td>
        </tr>
    </table>

    <h2>Dependency Vulnerabilities</h2>
"@
    
    if ($Vulnerabilities.Count -gt 0) {
        $report += @"
    <table>
        <tr>
            <th>Type</th>
            <th>Component</th>
            <th>Critical</th>
            <th>High</th>
            <th>Moderate</th>
            <th>Low</th>
        </tr>
"@
        
        foreach ($vuln in $Vulnerabilities) {
            $report += @"
        <tr>
            <td>$($vuln.Type)</td>
            <td>$($vuln.Component)</td>
            <td>$($vuln.Critical)</td>
            <td>$($vuln.High)</td>
            <td>$($vuln.Moderate)</td>
            <td>$($vuln.Low)</td>
        </tr>
"@
        }
        
        $report += "</table>"
    }
    else {
        $report += "<p>✅ No dependency vulnerabilities found.</p>"
    }
    
    $report += @"

    <h2>Security Issues</h2>
"@
    
    if ($SecurityIssues.Count -gt 0) {
        foreach ($severity in @("Critical", "High", "Medium", "Low")) {
            $issues = $SecurityIssues | Where-Object { $_.Severity -eq $severity }
            if ($issues.Count -gt 0) {
                $report += "<h3 class='$($severity.ToLower())'>$severity Issues</h3>"
                
                foreach ($issue in $issues) {
                    $report += @"
    <div class="issue $($severity.ToLower())">
        <h4>$($issue.Type)</h4>
        <p><strong>File:</strong> $($issue.File)</p>
        <p><strong>Description:</strong> $($issue.Description)</p>
"@
                    
                    if ($issue.Pattern) {
                        $report += "<p><strong>Pattern:</strong> $($issue.Pattern)</p>"
                    }
                    
                    if ($issue.Issue) {
                        $report += "<p><strong>Issue:</strong> $($issue.Issue)</p>"
                    }
                    
                    $report += "</div>"
                }
            }
        }
    }
    else {
        $report += "<p>✅ No security issues found.</p>"
    }
    
    $report += @"

    <h2>Recommendations</h2>
    <ul>
"@
    
    foreach ($rec in $Recommendations) {
        $report += "<li>$rec</li>"
    }
    
    $report += @"
    </ul>

    <h2>Next Steps</h2>
    <ol>
        <li>Address all Critical and High severity issues immediately</li>
        <li>Update vulnerable dependencies</li>
        <li>Implement security recommendations</li>
        <li>Schedule regular security audits</li>
        <li>Train development team on security best practices</li>
    </ol>

    <div class="header">
        <p><strong>Report generated by ISM AI Platform Security Audit Script</strong></p>
    </div>
</body>
</html>
"@
    
    Set-Content -Path $OutputPath -Value $report
    Write-SecurityLog "✅ Security report saved to: $OutputPath" $Green
}

# Fix common security issues
function Repair-SecurityIssues {
    Write-SecurityLog "Attempting to fix common security issues..." $Blue
    
    $fixed = 0
    
    # Fix npm vulnerabilities
    if (Test-Path "backend/package.json") {
        Set-Location backend
        Write-SecurityLog "Fixing backend vulnerabilities..."
        npm audit fix --force 2>$null
        $fixed++
        Set-Location ..
    }
    
    if (Test-Path "frontend/package.json") {
        Set-Location frontend
        Write-SecurityLog "Fixing frontend vulnerabilities..."
        npm audit fix --force 2>$null
        $fixed++
        Set-Location ..
    }
    
    # Update Python dependencies
    if (Test-Path "requirements.txt") {
        Write-SecurityLog "Updating Python dependencies..."
        pip install --upgrade -r requirements.txt 2>$null
        $fixed++
    }
    
    Write-SecurityLog "Fixed $fixed security issues" $Green
    return $fixed
}

# Main security audit function
function Start-SecurityAudit {
    Write-SecurityLog "Starting ISM Platform security audit..." $Blue
    
    # Clear previous results
    $script:SecurityIssues = @()
    $script:Recommendations = @()
    $script:Vulnerabilities = @()
    
    switch ($Command) {
        "full" {
            Test-SecurityTools
            Test-DependencySecurity
            Test-CodeSecurity
            Test-InfrastructureSecurity
            Test-EnvironmentSecurity
        }
        "quick" {
            Test-DependencySecurity
            Test-EnvironmentSecurity
        }
        "dependencies" {
            Test-DependencySecurity
        }
        "code" {
            Test-CodeSecurity
        }
        "infrastructure" {
            Test-InfrastructureSecurity
        }
        "report" {
            Start-SecurityAudit
            Get-SecurityRecommendations
            New-SecurityReport
        }
        "fix" {
            Repair-SecurityIssues
        }
        "help" {
            Write-Host "ISM Platform Security Audit Script"
            Write-Host ""
            Write-Host "Usage: .\security-audit.ps1 [command]"
            Write-Host ""
            Write-Host "Commands:"
            Write-Host "  full         Complete security audit (default)"
            Write-Host "  quick        Quick security check"
            Write-Host "  dependencies Audit dependencies only"
            Write-Host "  code         Audit code security only"
            Write-Host "  infrastructure Audit infrastructure security only"
            Write-Host "  report       Generate security report"
            Write-Host "  fix          Fix common security issues"
            Write-Host "  help         Show this help message"
        }
        default {
            Test-SecurityTools
            Test-DependencySecurity
            Test-CodeSecurity
            Test-InfrastructureSecurity
            Test-EnvironmentSecurity
        }
    }
    
    # Generate recommendations
    Get-SecurityRecommendations
    
    # Summary
    Write-SecurityLog "Security audit completed" $Blue
    Write-SecurityLog "Security issues found: $($SecurityIssues.Count)" $Blue
    Write-SecurityLog "Vulnerabilities found: $($Vulnerabilities.Count)" $Blue
    Write-SecurityLog "Recommendations: $($Recommendations.Count)" $Blue
    
    # Return overall status
    $criticalIssues = ($SecurityIssues | Where-Object { $_.Severity -eq "Critical" }).Count
    $highIssues = ($SecurityIssues | Where-Object { $_.Severity -eq "High" }).Count
    
    if ($criticalIssues -gt 0 -or $highIssues -gt 0) {
        Write-SecurityLog "❌ Security audit failed - Critical/High issues found" $Red
        return $false
    }
    else {
        Write-SecurityLog "✅ Security audit passed" $Green
        return $true
    }
}

# Run main function
Start-SecurityAudit 