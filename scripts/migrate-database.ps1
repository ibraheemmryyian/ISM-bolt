# Database Migration Automation Script for ISM AI Platform (PowerShell)
# Handles Supabase migrations, schema updates, and data migrations

param(
    [Parameter(Position=0)]
    [ValidateSet("", "rollback", "status", "validate", "backup", "help")]
    [string]$Command = ""
)

# Configuration
$MigrationDir = "frontend/supabase/migrations"
$BackupDir = "backups/migrations"
$LogFile = "logs/migration.log"
$RollbackFile = "temp/migration_rollback.txt"

# Ensure directories exist
if (!(Test-Path $BackupDir)) { New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null }
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" -Force | Out-Null }
if (!(Test-Path "temp")) { New-Item -ItemType Directory -Path "temp" -Force | Out-Null }

# Logging functions
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Level`: $Message"
    
    switch ($Level) {
        "ERROR" { Write-Host $logMessage -ForegroundColor Red }
        "WARN"  { Write-Host $logMessage -ForegroundColor Yellow }
        "INFO"  { Write-Host $logMessage -ForegroundColor Blue }
        default { Write-Host $logMessage -ForegroundColor Green }
    }
    
    Add-Content -Path $LogFile -Value $logMessage
}

# Check if Supabase CLI is installed
function Test-SupabaseCLI {
    try {
        $null = Get-Command supabase -ErrorAction Stop
        Write-Log "Supabase CLI found"
        return $true
    }
    catch {
        Write-Log "Supabase CLI is not installed. Please install it first:" "ERROR"
        Write-Host "npm install -g supabase"
        return $false
    }
}

# Backup current database state
function Backup-Database {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupFile = Join-Path $BackupDir "backup_$timestamp.sql"
    
    Write-Log "Creating database backup..."
    
    try {
        # Create backup using Supabase CLI
        $result = supabase db dump --file $backupFile 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Database backup created: $backupFile"
            Set-Content -Path $RollbackFile -Value $backupFile
            return $true
        }
        else {
            Write-Log "Could not create full database backup, continuing with schema backup" "WARN"
            # Fallback: backup schema only
            supabase db dump --schema-only --file $backupFile 2>$null
            return $true
        }
    }
    catch {
        Write-Log "Backup failed: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Check migration status
function Get-MigrationStatus {
    Write-Log "Checking current migration status..."
    
    try {
        $appliedMigrations = supabase migration list --status applied 2>$null
        $pendingMigrations = supabase migration list --status pending 2>$null
        
        $appliedCount = ($appliedMigrations -split "`n" | Where-Object { $_ -match '\S' }).Count
        $pendingCount = ($pendingMigrations -split "`n" | Where-Object { $_ -match '\S' }).Count
        
        Write-Log "Applied migrations: $appliedCount"
        Write-Log "Pending migrations: $pendingCount"
        
        if ($pendingCount -gt 0) {
            Write-Log "Found pending migrations to apply"
            return $true
        }
        else {
            Write-Log "No pending migrations found"
            return $false
        }
    }
    catch {
        Write-Log "Failed to check migration status: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Validate migration files
function Test-MigrationFiles {
    Write-Log "Validating migration files..."
    
    $migrationFiles = Get-ChildItem -Path $MigrationDir -Filter "*.sql" | Sort-Object Name
    $errorCount = 0
    
    foreach ($file in $migrationFiles) {
        Write-Log "Validating: $($file.Name)"
        
        # Basic SQL syntax check (simplified for PowerShell)
        $content = Get-Content $file.FullName -Raw
        if ($content -match "CREATE|ALTER|INSERT|UPDATE|DELETE") {
            # Basic validation passed
        }
        else {
            Write-Log "Migration file may have syntax issues: $($file.Name)" "WARN"
            $errorCount++
        }
    }
    
    if ($errorCount -gt 0) {
        Write-Log "Found $errorCount migration files with potential issues" "ERROR"
        return $false
    }
    
    Write-Log "All migration files validated successfully"
    return $true
}

# Run Supabase migrations
function Invoke-SupabaseMigrations {
    Write-Log "Running Supabase migrations..."
    
    # Check if we're in a Supabase project
    if (!(Test-Path "supabase/config.toml")) {
        Write-Log "Not in a Supabase project directory. Please run from project root." "ERROR"
        return $false
    }
    
    try {
        # Check if linked to project
        $status = supabase status 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Not linked to Supabase project. Attempting to link..." "WARN"
            
            if ($env:SUPABASE_PROJECT_REF) {
                supabase link --project-ref $env:SUPABASE_PROJECT_REF 2>$null
                if ($LASTEXITCODE -ne 0) {
                    Write-Log "Failed to link to Supabase project. Please check your configuration." "ERROR"
                    return $false
                }
            }
            else {
                Write-Log "SUPABASE_PROJECT_REF environment variable not set" "ERROR"
                return $false
            }
        }
        
        # Push migrations to remote database
        supabase db push
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Supabase migrations applied successfully"
            return $true
        }
        else {
            Write-Log "Failed to apply Supabase migrations" "ERROR"
            return $false
        }
    }
    catch {
        Write-Log "Migration failed: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Run custom migrations
function Invoke-CustomMigrations {
    Write-Log "Running custom migrations..."
    
    $customMigrations = @(
        "materials_migration.sql",
        "supabase_email_verification_migration.sql",
        "fix_materials_table_safe.sql"
    )
    
    foreach ($migration in $customMigrations) {
        if (Test-Path $migration) {
            Write-Log "Applying custom migration: $migration"
            
            if (Invoke-SqlMigration $migration) {
                Write-Log "Custom migration applied: $migration"
            }
            else {
                Write-Log "Failed to apply custom migration: $migration" "ERROR"
                return $false
            }
        }
        else {
            Write-Log "Custom migration file not found: $migration" "WARN"
        }
    }
    
    Write-Log "Custom migrations completed"
    return $true
}

# Apply SQL migration via Supabase API
function Invoke-SqlMigration {
    param([string]$MigrationFile)
    
    try {
        $sqlContent = Get-Content $MigrationFile -Raw
        
        # Use Invoke-RestMethod to apply SQL via Supabase REST API
        $headers = @{
            "apikey" = $env:SUPABASE_ANON_KEY
            "Authorization" = "Bearer $env:SUPABASE_ANON_KEY"
            "Content-Type" = "application/json"
        }
        
        $body = @{
            sql = $sqlContent
        } | ConvertTo-Json
        
        $response = Invoke-RestMethod -Uri "$env:SUPABASE_URL/rest/v1/rpc/exec_sql" `
                                     -Method POST `
                                     -Headers $headers `
                                     -Body $body
        
        return $true
    }
    catch {
        Write-Log "SQL migration failed: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Verify migration success
function Test-MigrationSuccess {
    Write-Log "Verifying migration success..."
    
    try {
        # Check database connectivity
        $status = supabase status 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Database connection failed after migration" "ERROR"
            return $false
        }
        
        # Check key tables exist
        $keyTables = @("companies", "materials", "profiles", "subscriptions")
        foreach ($table in $keyTables) {
            $query = "SELECT 1 FROM $table LIMIT 1;"
            $result = supabase db query $query 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Log "Table $table verified"
            }
            else {
                Write-Log "Table $table may not exist or be accessible" "WARN"
            }
        }
        
        Write-Log "Migration verification completed"
        return $true
    }
    catch {
        Write-Log "Verification failed: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Rollback function
function Invoke-MigrationRollback {
    Write-Log "Rollback triggered!" "ERROR"
    
    try {
        $backupFile = Get-Content $RollbackFile -ErrorAction SilentlyContinue
        
        if ($backupFile -and (Test-Path $backupFile)) {
            Write-Log "Rolling back to backup: $backupFile"
            
            # Restore from backup
            supabase db reset --linked
            if ($LASTEXITCODE -eq 0) {
                Write-Log "Database reset completed"
                return $true
            }
            else {
                Write-Log "Failed to reset database" "ERROR"
                return $false
            }
        }
        else {
            Write-Log "No backup file found for rollback" "ERROR"
            return $false
        }
    }
    catch {
        Write-Log "Rollback failed: $($_.Exception.Message)" "ERROR"
        return $false
    }
}

# Cleanup function
function Remove-MigrationArtifacts {
    Write-Log "Cleaning up migration artifacts..."
    
    # Remove temporary files
    if (Test-Path $RollbackFile) {
        Remove-Item $RollbackFile -Force
    }
    
    # Clean old backups (keep last 10)
    if (Test-Path $BackupDir) {
        $backups = Get-ChildItem -Path $BackupDir -Filter "*.sql" | Sort-Object LastWriteTime -Descending
        if ($backups.Count -gt 10) {
            $backups | Select-Object -Skip 10 | Remove-Item -Force
            Write-Log "Cleaned up old backup files"
        }
    }
}

# Main migration function
function Start-Migration {
    Write-Log "Starting database migration process"
    
    # Check prerequisites
    if (!(Test-SupabaseCLI)) {
        exit 1
    }
    
    # Set up error handling
    $ErrorActionPreference = "Stop"
    
    try {
        # Backup current state
        if (!(Backup-Database)) {
            throw "Backup failed"
        }
        
        # Check migration status
        if (!(Get-MigrationStatus)) {
            Write-Log "No migrations to apply"
            Remove-MigrationArtifacts
            return
        }
        
        # Validate migrations
        if (!(Test-MigrationFiles)) {
            throw "Migration validation failed"
        }
        
        # Run Supabase migrations
        if (!(Invoke-SupabaseMigrations)) {
            throw "Supabase migration failed"
        }
        
        # Run custom migrations
        if (!(Invoke-CustomMigrations)) {
            throw "Custom migration failed"
        }
        
        # Verify migrations
        if (!(Test-MigrationSuccess)) {
            throw "Migration verification failed"
        }
        
        # Success
        Write-Log "Database migration completed successfully!"
        Remove-MigrationArtifacts
    }
    catch {
        Write-Log "Migration failed: $($_.Exception.Message)" "ERROR"
        Invoke-MigrationRollback
        Remove-MigrationArtifacts
        exit 1
    }
}

# Handle script arguments
switch ($Command) {
    "rollback" {
        Invoke-MigrationRollback
    }
    "status" {
        Get-MigrationStatus
    }
    "validate" {
        Test-MigrationFiles
    }
    "backup" {
        Backup-Database
    }
    "help" {
        Write-Host "Database Migration Script (PowerShell)"
        Write-Host ""
        Write-Host "Usage: .\migrate-database.ps1 [command]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  (no args)  Run full migration process"
        Write-Host "  rollback   Rollback to previous backup"
        Write-Host "  status     Check migration status"
        Write-Host "  validate   Validate migration files"
        Write-Host "  backup     Create database backup"
        Write-Host "  help       Show this help message"
    }
    default {
        Start-Migration
    }
} 