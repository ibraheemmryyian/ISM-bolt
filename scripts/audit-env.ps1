# Environment Variable Auditing Script for ISM AI Platform
# Checks for missing, insecure, and misconfigured environment variables

param(
    [Parameter(Position=0)]
    [ValidateSet("", "check", "fix", "report", "help")]
    [string]$Command = "check"
)

# Configuration
$EnvFiles = @(
    "backend/.env",
    "backend/env.example",
    "frontend/.env",
    "frontend/.env.example"
)

$RequiredVars = @{
    "backend" = @(
        "PORT",
        "NODE_ENV",
        "FRONTEND_URL",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "JWT_SECRET",
        "SESSION_SECRET"
    )
    "frontend" = @(
        "VITE_SUPABASE_URL",
        "VITE_SUPABASE_ANON_KEY",
        "VITE_API_URL"
    )
}

$SensitiveVars = @(
    "JWT_SECRET",
    "SESSION_SECRET",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "DEEPSEEK_API_KEY"
)

$SecurityIssues = @()
$MissingVars = @()
$Recommendations = @()

# Logging functions
function Write-AuditLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Level`: $Message"
    
    switch ($Level) {
        "ERROR" { Write-Host $logMessage -ForegroundColor Red }
        "WARN"  { Write-Host $logMessage -ForegroundColor Yellow }
        "INFO"  { Write-Host $logMessage -ForegroundColor Blue }
        default { Write-Host $logMessage -ForegroundColor Green }
    }
}

# Check if environment file exists
function Test-EnvFile {
    param([string]$FilePath)
    
    if (Test-Path $FilePath) {
        Write-AuditLog "Found environment file: $FilePath"
        return $true
    }
    else {
        Write-AuditLog "Missing environment file: $FilePath" "WARN"
        return $false
    }
}

# Parse environment file
function Get-EnvVariables {
    param([string]$FilePath)
    
    if (!(Test-Path $FilePath)) {
        return @{}
    }
    
    $envVars = @{}
    $content = Get-Content $FilePath -ErrorAction SilentlyContinue
    
    foreach ($line in $content) {
        $line = $line.Trim()
        
        # Skip comments and empty lines
        if ($line -match '^#' -or $line -eq '') {
            continue
        }
        
        # Parse key=value pairs
        if ($line -match '^([^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            
            # Remove quotes if present
            if ($value -match '^["\'](.*)["\']$') {
                $value = $matches[1]
            }
            
            $envVars[$key] = $value
        }
    }
    
    return $envVars
}

# Check for missing required variables
function Test-RequiredVariables {
    param([string]$EnvType, [hashtable]$EnvVars)
    
    Write-AuditLog "Checking required variables for $EnvType"
    
    $required = $RequiredVars[$EnvType]
    $missing = @()
    
    foreach ($var in $required) {
        if (!$EnvVars.ContainsKey($var) -or $EnvVars[$var] -eq '') {
            $missing += $var
        }
    }
    
    if ($missing.Count -gt 0) {
        Write-AuditLog "Missing required variables for $EnvType`: $($missing -join ', ')" "ERROR"
        $script:MissingVars += @{
            Type = $EnvType
            Variables = $missing
        }
        return $false
    }
    else {
        Write-AuditLog "All required variables present for $EnvType"
        return $true
    }
}

# Check for security issues
function Test-SecurityIssues {
    param([hashtable]$EnvVars, [string]$FilePath)
    
    Write-AuditLog "Checking security issues in $FilePath"
    
    foreach ($sensitiveVar in $SensitiveVars) {
        if ($EnvVars.ContainsKey($sensitiveVar)) {
            $value = $EnvVars[$sensitiveVar]
            
            # Check for default/placeholder values
            if ($value -match '^(your-|changeme|placeholder|default)') {
                $script:SecurityIssues += @{
                    File = $FilePath
                    Variable = $sensitiveVar
                    Issue = "Using default/placeholder value"
                    Severity = "HIGH"
                }
                Write-AuditLog "Security issue: $sensitiveVar using default value" "ERROR"
            }
            
            # Check for weak secrets
            if ($sensitiveVar -match 'SECRET' -and $value.Length -lt 32) {
                $script:SecurityIssues += @{
                    File = $FilePath
                    Variable = $sensitiveVar
                    Issue = "Weak secret (less than 32 characters)"
                    Severity = "HIGH"
                }
                Write-AuditLog "Security issue: $sensitiveVar is too weak" "ERROR"
            }
            
            # Check for hardcoded production values in development
            if ($value -match '^sk-|^pk_' -and $FilePath -match '\.env$') {
                $script:SecurityIssues += @{
                    File = $FilePath
                    Variable = $sensitiveVar
                    Issue = "Production API key in .env file"
                    Severity = "CRITICAL"
                }
                Write-AuditLog "Security issue: Production API key in .env file" "ERROR"
            }
        }
    }
}

# Check for configuration issues
function Test-ConfigurationIssues {
    param([hashtable]$EnvVars, [string]$FilePath)
    
    Write-AuditLog "Checking configuration issues in $FilePath"
    
    # Check for development settings in production
    if ($EnvVars.ContainsKey("NODE_ENV") -and $EnvVars["NODE_ENV"] -eq "production") {
        if ($EnvVars.ContainsKey("FRONTEND_URL") -and $EnvVars["FRONTEND_URL"] -match 'localhost') {
            $script:SecurityIssues += @{
                File = $FilePath
                Variable = "FRONTEND_URL"
                Issue = "Production environment using localhost URL"
                Severity = "MEDIUM"
            }
            Write-AuditLog "Configuration issue: Production using localhost URL" "WARN"
        }
    }
    
    # Check for missing CORS configuration
    if ($EnvVars.ContainsKey("FRONTEND_URL") -and $EnvVars["FRONTEND_URL"] -eq '') {
        $script:SecurityIssues += @{
            File = $FilePath
            Variable = "FRONTEND_URL"
            Issue = "Missing CORS configuration"
            Severity = "MEDIUM"
        }
        Write-AuditLog "Configuration issue: Missing CORS configuration" "WARN"
    }
    
    # Check for insecure ports
    if ($EnvVars.ContainsKey("PORT") -and $EnvVars["PORT"] -eq "3000") {
        $script:Recommendations += "Consider using a non-standard port for production"
    }
}

# Generate recommendations
function Get-Recommendations {
    Write-AuditLog "Generating recommendations"
    
    $script:Recommendations += @(
        "Use environment-specific .env files (.env.development, .env.production)",
        "Implement secrets management (AWS Secrets Manager, Azure Key Vault)",
        "Use strong, randomly generated secrets (32+ characters)",
        "Enable environment variable encryption in production",
        "Implement regular secret rotation",
        "Use .env.example files for documentation",
        "Add environment variable validation at startup",
        "Implement proper CORS configuration for production"
    )
}

# Generate security report
function New-SecurityReport {
    param([string]$OutputPath = "security-audit-report.txt")
    
    Write-AuditLog "Generating security report: $OutputPath"
    
    $report = @"
# Environment Variable Security Audit Report
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Summary
- Files Audited: $($EnvFiles.Count)
- Security Issues Found: $($SecurityIssues.Count)
- Missing Variables: $($MissingVars.Count)
- Recommendations: $($Recommendations.Count)

## Security Issues
"@
    
    if ($SecurityIssues.Count -gt 0) {
        foreach ($issue in $SecurityIssues) {
            $report += @"

### $($issue.Severity) - $($issue.Variable)
- File: $($issue.File)
- Issue: $($issue.Issue)
"@
        }
    }
    else {
        $report += @"

No security issues found.
"@
    }
    
    $report += @"

## Missing Required Variables
"@
    
    if ($MissingVars.Count -gt 0) {
        foreach ($missing in $MissingVars) {
            $report += @"

### $($missing.Type)
- Missing: $($missing.Variables -join ', ')
"@
        }
    }
    else {
        $report += @"

No missing required variables.
"@
    }
    
    $report += @"

## Recommendations
"@
    
    foreach ($rec in $Recommendations) {
        $report += @"
- $rec
"@
    }
    
    $report += @"

## Next Steps
1. Address all HIGH and CRITICAL security issues immediately
2. Set up missing required variables
3. Implement recommended security measures
4. Schedule regular security audits
5. Document environment variable requirements

---
Report generated by ISM AI Platform Security Audit Script
"@
    
    Set-Content -Path $OutputPath -Value $report
    Write-AuditLog "Security report saved to: $OutputPath"
}

# Fix common issues
function Repair-EnvironmentFiles {
    Write-AuditLog "Attempting to fix common issues"
    
    $fixed = 0
    
    # Create missing .env files from examples
    foreach ($envFile in $EnvFiles) {
        if ($envFile -match '\.env$' -and !(Test-Path $envFile)) {
            $exampleFile = $envFile -replace '\.env$', '.env.example'
            if (Test-Path $exampleFile) {
                Copy-Item $exampleFile $envFile
                Write-AuditLog "Created $envFile from example"
                $fixed++
            }
        }
    }
    
    # Generate strong secrets for weak ones
    foreach ($envFile in $EnvFiles) {
        if (Test-Path $envFile) {
            $envVars = Get-EnvVariables $envFile
            $content = Get-Content $envFile
            
            foreach ($sensitiveVar in $SensitiveVars) {
                if ($envVars.ContainsKey($sensitiveVar)) {
                    $value = $envVars[$sensitiveVar]
                    
                    # Replace weak secrets
                    if ($value -match '^(your-|changeme|placeholder|default)' -or 
                        ($sensitiveVar -match 'SECRET' -and $value.Length -lt 32)) {
                        
                        $newSecret = -join ((33..126) | Get-Random -Count 64 | ForEach-Object {[char]$_})
                        $content = $content -replace "$sensitiveVar=.*", "$sensitiveVar=`"$newSecret`""
                        
                        Write-AuditLog "Generated strong secret for $sensitiveVar"
                        $fixed++
                    }
                }
            }
            
            Set-Content -Path $envFile -Value $content
        }
    }
    
    Write-AuditLog "Fixed $fixed issues"
    return $fixed
}

# Main audit function
function Start-EnvironmentAudit {
    Write-AuditLog "Starting environment variable audit"
    
    # Clear previous results
    $script:SecurityIssues = @()
    $script:MissingVars = @()
    $script:Recommendations = @()
    
    # Audit each environment file
    foreach ($envFile in $EnvFiles) {
        if (Test-EnvFile $envFile) {
            $envVars = Get-EnvVariables $envFile
            
            # Determine environment type
            $envType = if ($envFile -match 'backend') { "backend" } else { "frontend" }
            
            # Run checks
            Test-RequiredVariables $envType $envVars
            Test-SecurityIssues $envVars $envFile
            Test-ConfigurationIssues $envVars $envFile
        }
    }
    
    # Generate recommendations
    Get-Recommendations
    
    # Summary
    Write-AuditLog "Audit completed"
    Write-AuditLog "Security issues found: $($SecurityIssues.Count)"
    Write-AuditLog "Missing variables: $($MissingVars.Count)"
    Write-AuditLog "Recommendations: $($Recommendations.Count)"
    
    # Return overall status
    return ($SecurityIssues.Count -eq 0 -and $MissingVars.Count -eq 0)
}

# Handle script arguments
switch ($Command) {
    "check" {
        $success = Start-EnvironmentAudit
        if ($success) {
            Write-AuditLog "Environment audit passed - no issues found"
            exit 0
        }
        else {
            Write-AuditLog "Environment audit failed - issues found" "ERROR"
            exit 1
        }
    }
    "fix" {
        Start-EnvironmentAudit
        $fixed = Repair-EnvironmentFiles
        Write-AuditLog "Repair completed - $fixed issues fixed"
    }
    "report" {
        Start-EnvironmentAudit
        New-SecurityReport
    }
    "help" {
        Write-Host "Environment Variable Audit Script"
        Write-Host ""
        Write-Host "Usage: .\audit-env.ps1 [command]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  check   Run security audit (default)"
        Write-Host "  fix     Attempt to fix common issues"
        Write-Host "  report  Generate detailed security report"
        Write-Host "  help    Show this help message"
        Write-Host ""
        Write-Host "The script checks for:"
        Write-Host "- Missing required environment variables"
        Write-Host "- Security issues (weak secrets, default values)"
        Write-Host "- Configuration problems"
        Write-Host "- Best practice violations"
    }
    default {
        Start-EnvironmentAudit
    }
} 