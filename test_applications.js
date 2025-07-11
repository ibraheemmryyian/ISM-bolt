// Test script for Request Access applications
// Run this in your browser console to test the system

console.log('=== REQUEST ACCESS APPLICATIONS TEST ===');

// Test 1: Check if Supabase is accessible
async function testSupabaseConnection() {
  console.log('1. Testing Supabase connection...');
  try {
    const { data, error } = await supabase.from('company_applications').select('count').limit(1);
    if (error) throw error;
    console.log('✅ Supabase connection: OK');
    return true;
  } catch (err) {
    console.error('❌ Supabase connection failed:', err);
    return false;
  }
}

// Test 2: Check if we can read applications
async function testReadApplications() {
  console.log('2. Testing read applications...');
  try {
    const { data, error } = await supabase.from('company_applications').select('*').limit(5);
    if (error) throw error;
    console.log('✅ Read applications: OK');
    console.log('   Found', data.length, 'applications');
    return data.length;
  } catch (err) {
    console.error('❌ Read applications failed:', err);
    return 0;
  }
}

// Test 3: Test submitting a new application
async function testSubmitApplication() {
  console.log('3. Testing submit application...');
  try {
    const testData = {
      company_name: 'Test Company ' + Date.now(),
      contact_email: 'test' + Date.now() + '@example.com',
      contact_name: 'Test Contact',
      application_answers: { motivation: 'Test motivation for automated testing' },
      status: 'pending'
    };
    
    const { data, error } = await supabase.from('company_applications').insert(testData).select();
    if (error) throw error;
    console.log('✅ Submit application: OK');
    console.log('   Created application ID:', data[0].id);
    return data[0].id;
  } catch (err) {
    console.error('❌ Submit application failed:', err);
    return null;
  }
}

// Test 4: Test admin dashboard loading
async function testAdminDashboard() {
  console.log('4. Testing admin dashboard...');
  try {
    // Simulate what the admin dashboard does
    const { data: applications, error: appsError } = await supabase
      .from('company_applications')
      .select('*')
      .order('created_at', { ascending: false });
    
    if (appsError) throw appsError;
    
    const { data: companies, error: companiesError } = await supabase
      .from('companies')
      .select('*')
      .order('created_at', { ascending: false });
    
    if (companiesError) throw companiesError;
    
    console.log('✅ Admin dashboard data loading: OK');
    console.log('   Applications:', applications.length);
    console.log('   Companies:', companies.length);
    return { applications: applications.length, companies: companies.length };
  } catch (err) {
    console.error('❌ Admin dashboard failed:', err);
    return null;
  }
}

// Test 5: Test RLS policies
async function testRLSPolicies() {
  console.log('5. Testing RLS policies...');
  try {
    // Test if we can insert without authentication
    const { error: insertError } = await supabase.from('company_applications').insert({
      company_name: 'RLS Test Company',
      contact_email: 'rls@test.com',
      contact_name: 'RLS Test',
      application_answers: { motivation: 'RLS test' },
      status: 'pending'
    });
    
    if (insertError) {
      console.log('❌ Insert blocked by RLS:', insertError.message);
      return false;
    } else {
      console.log('✅ Insert allowed by RLS');
      return true;
    }
  } catch (err) {
    console.error('❌ RLS test failed:', err);
    return false;
  }
}

// Run all tests
async function runAllTests() {
  console.log('Starting comprehensive test...\n');
  
  const results = {
    supabase: await testSupabaseConnection(),
    read: await testReadApplications(),
    submit: await testSubmitApplication(),
    admin: await testAdminDashboard(),
    rls: await testRLSPolicies()
  };
  
  console.log('\n=== TEST RESULTS ===');
  console.log('Supabase Connection:', results.supabase ? '✅ PASS' : '❌ FAIL');
  console.log('Read Applications:', results.read > 0 ? '✅ PASS' : '❌ FAIL');
  console.log('Submit Application:', results.submit ? '✅ PASS' : '❌ FAIL');
  console.log('Admin Dashboard:', results.admin ? '✅ PASS' : '❌ FAIL');
  console.log('RLS Policies:', results.rls ? '✅ PASS' : '❌ FAIL');
  
  const allPassed = Object.values(results).every(r => r !== false && r !== null);
  console.log('\nOverall Result:', allPassed ? '✅ ALL TESTS PASSED' : '❌ SOME TESTS FAILED');
  
  return results;
}

// Export for manual testing
window.testApplications = runAllTests;
console.log('Run testApplications() to execute all tests'); 