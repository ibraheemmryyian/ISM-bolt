const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

// Load environment variables
require('dotenv').config({ path: path.join(__dirname, 'backend', '.env') });

// Get Supabase configuration from environment
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
    console.error('❌ Supabase configuration not found!');
    console.error('Please ensure your .env file contains:');
    console.error('SUPABASE_URL=your-supabase-url');
    console.error('SUPABASE_ANON_KEY=your-supabase-anon-key');
    process.exit(1);
}

// Create Supabase client
const supabase = createClient(supabaseUrl, supabaseKey);

async function runMigration() {
    try {
        console.log('🚀 Starting Supabase Migration...');
        console.log('📊 Reading migration file...');
        
        // Read the SQL migration file
        const migrationFile = path.join(__dirname, 'database_migration_for_real_data.sql');
        
        if (!fs.existsSync(migrationFile)) {
            console.error('❌ Migration file not found:', migrationFile);
            process.exit(1);
        }
        
        const migrationSQL = fs.readFileSync(migrationFile, 'utf8');
        console.log('✅ Migration file loaded successfully');
        
        // Split the migration into individual statements
        const statements = migrationSQL
            .split(';')
            .map(stmt => stmt.trim())
            .filter(stmt => stmt.length > 0 && !stmt.startsWith('--') && !stmt.startsWith('/*'))
            .filter(stmt => !stmt.toLowerCase().includes('do $$'));
        
        console.log(`📊 Found ${statements.length} SQL statements to execute`);
        console.log('');
        
        let successCount = 0;
        let errorCount = 0;
        
        // Execute each statement
        for (let i = 0; i < statements.length; i++) {
            const statement = statements[i];
            if (statement.trim()) {
                try {
                    console.log(`🔄 Executing statement ${i + 1}/${statements.length}...`);
                    
                    // Use rpc to execute SQL (requires exec_sql function in Supabase)
                    const { error } = await supabase.rpc('exec_sql', { sql: statement });
                    
                    if (error) {
                        console.warn(`⚠️  Statement ${i + 1} warning: ${error.message}`);
                        errorCount++;
                    } else {
                        console.log(`✅ Statement ${i + 1} executed successfully`);
                        successCount++;
                    }
                } catch (err) {
                    console.warn(`⚠️  Statement ${i + 1} error: ${err.message}`);
                    errorCount++;
                }
            }
        }
        
        console.log('');
        console.log('🎉 Migration Summary:');
        console.log(`✅ Successful: ${successCount}`);
        console.log(`⚠️  Errors/Warnings: ${errorCount}`);
        console.log(`📊 Total: ${statements.length}`);
        
        if (errorCount === 0) {
            console.log('🎉 All statements executed successfully!');
        } else {
            console.log('⚠️  Some statements had issues. Check the output above.');
        }
        
    } catch (error) {
        console.error('❌ Migration failed:', error);
        process.exit(1);
    }
}

// Alternative method using direct SQL execution
async function runMigrationAlternative() {
    try {
        console.log('🚀 Starting Supabase Migration (Alternative Method)...');
        console.log('📊 Reading migration file...');
        
        const migrationFile = path.join(__dirname, 'database_migration_for_real_data.sql');
        const migrationSQL = fs.readFileSync(migrationFile, 'utf8');
        
        console.log('✅ Migration file loaded successfully');
        console.log('');
        console.log('📋 Instructions for manual execution:');
        console.log('');
        console.log('1. Go to your Supabase Dashboard');
        console.log('2. Navigate to SQL Editor');
        console.log('3. Copy and paste the following SQL:');
        console.log('');
        console.log('='.repeat(80));
        console.log(migrationSQL);
        console.log('='.repeat(80));
        console.log('');
        console.log('4. Click "Run" to execute the migration');
        console.log('');
        console.log('✅ Migration will create 20+ tables for real data collection');
        
    } catch (error) {
        console.error('❌ Failed to read migration file:', error);
        process.exit(1);
    }
}

// Check if exec_sql function exists
async function checkExecSqlFunction() {
    try {
        const { data, error } = await supabase.rpc('exec_sql', { sql: 'SELECT 1' });
        return !error;
    } catch (err) {
        return false;
    }
}

// Main execution
async function main() {
    console.log('🔍 Checking Supabase configuration...');
    
    const hasExecSql = await checkExecSqlFunction();
    
    if (hasExecSql) {
        console.log('✅ exec_sql function found. Running automated migration...');
        await runMigration();
    } else {
        console.log('⚠️  exec_sql function not found. Using alternative method...');
        await runMigrationAlternative();
    }
}

// Run the migration
main().catch(console.error); 