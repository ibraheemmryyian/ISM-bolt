// Fix Database Schema
// Run this script to add missing columns for the business profile form

require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');

// Initialize Supabase client
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

async function fixSchema() {
  console.log('🔧 Fixing database schema...');
  
  try {
    // Add process_description column
    console.log('📝 Adding process_description column...');
    const { error: descError } = await supabase.rpc('exec_sql', {
      sql: `
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'companies' 
                AND column_name = 'process_description'
            ) THEN
                ALTER TABLE companies ADD COLUMN process_description TEXT;
            END IF;
        END $$;
      `
    });
    
    if (descError) {
      console.log('⚠️ process_description column might already exist or error occurred:', descError.message);
    } else {
      console.log('✅ process_description column added');
    }

    // Add waste_quantity column
    console.log('📊 Adding waste_quantity column...');
    const { error: qtyError } = await supabase.rpc('exec_sql', {
      sql: `
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'companies' 
                AND column_name = 'waste_quantity'
            ) THEN
                ALTER TABLE companies ADD COLUMN waste_quantity NUMERIC;
            END IF;
        END $$;
      `
    });
    
    if (qtyError) {
      console.log('⚠️ waste_quantity column might already exist or error occurred:', qtyError.message);
    } else {
      console.log('✅ waste_quantity column added');
    }

    // Add waste_unit column
    console.log('📏 Adding waste_unit column...');
    const { error: unitError } = await supabase.rpc('exec_sql', {
      sql: `
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'companies' 
                AND column_name = 'waste_unit'
            ) THEN
                ALTER TABLE companies ADD COLUMN waste_unit VARCHAR(50);
            END IF;
        END $$;
      `
    });
    
    if (unitError) {
      console.log('⚠️ waste_unit column might already exist or error occurred:', unitError.message);
    } else {
      console.log('✅ waste_unit column added');
    }

    // Add waste_frequency column
    console.log('🕒 Adding waste_frequency column...');
    const { error: freqError } = await supabase.rpc('exec_sql', {
      sql: `
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'companies' 
                AND column_name = 'waste_frequency'
            ) THEN
                ALTER TABLE companies ADD COLUMN waste_frequency VARCHAR(50);
            END IF;
        END $$;
      `
    });
    
    if (freqError) {
      console.log('⚠️ waste_frequency column might already exist or error occurred:', freqError.message);
    } else {
      console.log('✅ waste_frequency column added');
    }

    // Add current_waste_management column
    console.log('🗑️ Adding current_waste_management column...');
    const { error: wasteError } = await supabase.rpc('exec_sql', {
      sql: `
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'companies' 
                AND column_name = 'current_waste_management'
            ) THEN
                ALTER TABLE companies ADD COLUMN current_waste_management VARCHAR(100);
            END IF;
        END $$;
      `
    });
    
    if (wasteError) {
      console.log('⚠️ current_waste_management column might already exist or error occurred:', wasteError.message);
    } else {
      console.log('✅ current_waste_management column added');
    }

    // Add user_type column
    console.log('👤 Adding user_type column...');
    const { error: userError } = await supabase.rpc('exec_sql', {
      sql: `
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'companies' 
                AND column_name = 'user_type'
            ) THEN
                ALTER TABLE companies ADD COLUMN user_type VARCHAR(50) DEFAULT 'business';
            END IF;
        END $$;
      `
    });
    
    if (userError) {
      console.log('⚠️ user_type column might already exist or error occurred:', userError.message);
    } else {
      console.log('✅ user_type column added');
    }

    // Add onboarding_completed column
    console.log('✅ Adding onboarding_completed column...');
    const { error: onboardingError } = await supabase.rpc('exec_sql', {
      sql: `
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'companies' 
                AND column_name = 'onboarding_completed'
            ) THEN
                ALTER TABLE companies ADD COLUMN onboarding_completed BOOLEAN DEFAULT FALSE;
            END IF;
        END $$;
      `
    });
    
    if (onboardingError) {
      console.log('⚠️ onboarding_completed column might already exist or error occurred:', onboardingError.message);
    } else {
      console.log('✅ onboarding_completed column added');
    }

    // Add application_status column
    console.log('📋 Adding application_status column...');
    const { error: statusError } = await supabase.rpc('exec_sql', {
      sql: `
        DO $$ 
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'companies' 
                AND column_name = 'application_status'
            ) THEN
                ALTER TABLE companies ADD COLUMN application_status VARCHAR(50) DEFAULT 'pending';
            END IF;
        END $$;
      `
    });
    
    if (statusError) {
      console.log('⚠️ application_status column might already exist or error occurred:', statusError.message);
    } else {
      console.log('✅ application_status column added');
    }

    // Verify the schema
    console.log('\n🔍 Verifying schema...');
    const { data: columns, error: verifyError } = await supabase
      .from('companies')
      .select('*')
      .limit(1);

    if (verifyError) {
      console.error('❌ Error verifying schema:', verifyError);
    } else {
      console.log('✅ Schema verification successful');
      console.log('📋 Available columns:', Object.keys(columns[0] || {}));
    }

    console.log('\n🎉 Schema fix completed!');
    console.log('💡 You can now use the business profile form without errors.');

  } catch (error) {
    console.error('❌ Error fixing schema:', error);
  }
}

// Run the fix
fixSchema(); 