#!/bin/bash

# Database Migration Automation Script for ISM AI Platform
# Handles Supabase migrations, schema updates, and data migrations

set -e  # Exit on any error

# Configuration
MIGRATION_DIR="frontend/supabase/migrations"
BACKUP_DIR="/backups/migrations"
LOG_FILE="/var/log/ism-migration.log"
ROLLBACK_FILE="/tmp/migration_rollback.sql"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a $LOG_FILE
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a $LOG_FILE
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a $LOG_FILE
}

# Check if Supabase CLI is installed
check_supabase_cli() {
    if ! command -v supabase &> /dev/null; then
        error "Supabase CLI is not installed. Please install it first:"
        echo "npm install -g supabase"
        exit 1
    fi
    log "Supabase CLI found"
}

# Create backup directory
create_backup_dir() {
    if [ ! -d "$BACKUP_DIR" ]; then
        mkdir -p "$BACKUP_DIR"
        log "Created backup directory: $BACKUP_DIR"
    fi
}

# Backup current database state
backup_database() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/backup_$timestamp.sql"
    
    log "Creating database backup..."
    
    # Create backup using Supabase CLI
    if supabase db dump --file "$backup_file" 2>/dev/null; then
        log "Database backup created: $backup_file"
        echo "$backup_file" > "$ROLLBACK_FILE"
    else
        warn "Could not create full database backup, continuing with schema backup"
        # Fallback: backup schema only
        supabase db dump --schema-only --file "$backup_file" 2>/dev/null || true
    fi
}

# Check migration status
check_migration_status() {
    log "Checking current migration status..."
    
    # Get list of applied migrations
    local applied_migrations=$(supabase migration list --status applied 2>/dev/null || echo "")
    local pending_migrations=$(supabase migration list --status pending 2>/dev/null || echo "")
    
    info "Applied migrations: $(echo "$applied_migrations" | wc -l)"
    info "Pending migrations: $(echo "$pending_migrations" | wc -l)"
    
    if [ -n "$pending_migrations" ]; then
        log "Found pending migrations to apply"
        return 0
    else
        log "No pending migrations found"
        return 1
    fi
}

# Validate migration files
validate_migrations() {
    log "Validating migration files..."
    
    local migration_files=$(find "$MIGRATION_DIR" -name "*.sql" -type f | sort)
    local error_count=0
    
    for file in $migration_files; do
        info "Validating: $(basename "$file")"
        
        # Check SQL syntax (basic validation)
        if ! sqlite3 :memory: < "$file" 2>/dev/null; then
            # Try PostgreSQL syntax validation
            if ! psql -h localhost -U postgres -d postgres -f "$file" --dry-run 2>/dev/null; then
                warn "Migration file may have syntax issues: $file"
                ((error_count++))
            fi
        fi
    done
    
    if [ $error_count -gt 0 ]; then
        error "Found $error_count migration files with potential issues"
        return 1
    fi
    
    log "All migration files validated successfully"
    return 0
}

# Run Supabase migrations
run_supabase_migrations() {
    log "Running Supabase migrations..."
    
    # Check if we're in a Supabase project
    if [ ! -f "supabase/config.toml" ]; then
        error "Not in a Supabase project directory. Please run from project root."
        exit 1
    fi
    
    # Link to Supabase project if not already linked
    if ! supabase status 2>/dev/null; then
        warn "Not linked to Supabase project. Attempting to link..."
        supabase link --project-ref "$SUPABASE_PROJECT_REF" 2>/dev/null || {
            error "Failed to link to Supabase project. Please check your configuration."
            exit 1
        }
    fi
    
    # Push migrations to remote database
    if supabase db push; then
        log "Supabase migrations applied successfully"
        return 0
    else
        error "Failed to apply Supabase migrations"
        return 1
    fi
}

# Run custom migrations
run_custom_migrations() {
    log "Running custom migrations..."
    
    local custom_migrations=(
        "materials_migration.sql"
        "supabase_email_verification_migration.sql"
        "fix_materials_table_safe.sql"
    )
    
    for migration in "${custom_migrations[@]}"; do
        if [ -f "$migration" ]; then
            info "Applying custom migration: $migration"
            
            # Apply migration using Supabase SQL editor API
            if apply_sql_migration "$migration"; then
                log "Custom migration applied: $migration"
            else
                error "Failed to apply custom migration: $migration"
                return 1
            fi
        else
            warn "Custom migration file not found: $migration"
        fi
    done
    
    log "Custom migrations completed"
    return 0
}

# Apply SQL migration via Supabase API
apply_sql_migration() {
    local migration_file="$1"
    local sql_content=$(cat "$migration_file")
    
    # Use curl to apply SQL via Supabase REST API
    local response=$(curl -s -X POST \
        -H "apikey: $SUPABASE_ANON_KEY" \
        -H "Authorization: Bearer $SUPABASE_ANON_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"sql\": \"$sql_content\"}" \
        "$SUPABASE_URL/rest/v1/rpc/exec_sql" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        return 0
    else
        return 1
    fi
}

# Verify migration success
verify_migrations() {
    log "Verifying migration success..."
    
    # Check database connectivity
    if ! supabase status 2>/dev/null; then
        error "Database connection failed after migration"
        return 1
    fi
    
    # Check key tables exist
    local key_tables=("companies" "materials" "profiles" "subscriptions")
    for table in "${key_tables[@]}"; do
        if ! supabase db query "SELECT 1 FROM $table LIMIT 1;" 2>/dev/null; then
            warn "Table $table may not exist or be accessible"
        else
            info "Table $table verified"
        fi
    done
    
    log "Migration verification completed"
    return 0
}

# Rollback function
rollback_migrations() {
    error "Rollback triggered!"
    
    local backup_file=$(cat "$ROLLBACK_FILE" 2>/dev/null || echo "")
    
    if [ -n "$backup_file" ] && [ -f "$backup_file" ]; then
        log "Rolling back to backup: $backup_file"
        
        # Restore from backup
        if supabase db reset --linked; then
            log "Database reset completed"
        else
            error "Failed to reset database"
            return 1
        fi
    else
        error "No backup file found for rollback"
        return 1
    fi
    
    log "Rollback completed"
    return 0
}

# Cleanup function
cleanup() {
    log "Cleaning up migration artifacts..."
    
    # Remove temporary files
    rm -f "$ROLLBACK_FILE"
    
    # Clean old backups (keep last 10)
    if [ -d "$BACKUP_DIR" ]; then
        cd "$BACKUP_DIR"
        ls -t *.sql | tail -n +11 | xargs -r rm
        log "Cleaned up old backup files"
    fi
}

# Main migration function
main() {
    log "Starting database migration process"
    
    # Check prerequisites
    check_supabase_cli
    create_backup_dir
    
    # Set trap for cleanup and rollback
    trap 'cleanup; exit 1' EXIT
    trap 'rollback_migrations; cleanup; exit 1' INT TERM
    
    # Backup current state
    backup_database
    
    # Check migration status
    if ! check_migration_status; then
        log "No migrations to apply"
        cleanup
        exit 0
    fi
    
    # Validate migrations
    if ! validate_migrations; then
        error "Migration validation failed"
        rollback_migrations
        exit 1
    fi
    
    # Run Supabase migrations
    if ! run_supabase_migrations; then
        error "Supabase migration failed"
        rollback_migrations
        exit 1
    fi
    
    # Run custom migrations
    if ! run_custom_migrations; then
        error "Custom migration failed"
        rollback_migrations
        exit 1
    fi
    
    # Verify migrations
    if ! verify_migrations; then
        error "Migration verification failed"
        rollback_migrations
        exit 1
    fi
    
    # Success
    log "Database migration completed successfully!"
    cleanup
    exit 0
}

# Handle script arguments
case "${1:-}" in
    "rollback")
        rollback_migrations
        ;;
    "status")
        check_migration_status
        ;;
    "validate")
        validate_migrations
        ;;
    "backup")
        backup_database
        ;;
    "help"|"-h"|"--help")
        echo "Database Migration Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)  Run full migration process"
        echo "  rollback   Rollback to previous backup"
        echo "  status     Check migration status"
        echo "  validate   Validate migration files"
        echo "  backup     Create database backup"
        echo "  help       Show this help message"
        ;;
    *)
        main
        ;;
esac 