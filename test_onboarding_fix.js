// Test Onboarding Fix
// Run this to verify the database schema and form issues are resolved

import { supabase } from './frontend/src/lib/supabase.js';

async function testOnboardingFix() {
  console.log('🧪 Testing Onboarding Fix...');
  
  try {
    // 1. Test if current_waste_management column exists
    console.log('📋 Checking database schema...');
    const { data: schemaData, error: schemaError } = await supabase
      .from('companies')
      .select('current_waste_management, waste_quantity, waste_unit, waste_frequency, user_type')
      .limit(1);
    
    if (schemaError) {
      console.error('❌ Schema error:', schemaError.message);
      if (schemaError.message.includes('current_waste_management')) {
        console.log('💡 Run the fix_onboarding_issues.sql script in Supabase SQL editor');
        return false;
      }
    } else {
      console.log('✅ Database schema is correct');
    }
    
    // 2. Test inserting a test company record
    console.log('📝 Testing form submission...');
    const testCompany = {
      id: 'test-' + Date.now(),
      name: 'Test Company',
      email: 'test@example.com',
      location: 'Test City',
      user_type: 'business',
      current_waste_management: 'plastic waste',
      waste_quantity: '1000',
      waste_unit: 'kg',
      waste_frequency: 'monthly',
      process_description: 'Test waste: plastic waste, Quantity: 1000 kg per monthly',
      onboarding_completed: true
    };
    
    const { error: insertError } = await supabase
      .from('companies')
      .insert([testCompany]);
    
    if (insertError) {
      console.error('❌ Insert error:', insertError.message);
      return false;
    } else {
      console.log('✅ Form submission works correctly');
    }
    
    // 3. Clean up test data
    const { error: deleteError } = await supabase
      .from('companies')
      .delete()
      .eq('id', testCompany.id);
    
    if (deleteError) {
      console.warn('⚠️ Could not clean up test data:', deleteError.message);
    } else {
      console.log('🧹 Test data cleaned up');
    }
    
    console.log('🎉 All tests passed! Onboarding should work correctly now.');
    return true;
    
  } catch (error) {
    console.error('❌ Test failed:', error);
    return false;
  }
}

// Run the test
testOnboardingFix().then(success => {
  if (success) {
    console.log('\n✅ Onboarding fix verification complete');
    console.log('📝 The form should now work correctly with:');
    console.log('   - Separate fields for quantity, unit, and frequency');
    console.log('   - Proper database schema support');
    console.log('   - Clear user interface');
  } else {
    console.log('\n❌ Onboarding fix verification failed');
    console.log('🔧 Please run the fix_onboarding_issues.sql script first');
  }
}); 