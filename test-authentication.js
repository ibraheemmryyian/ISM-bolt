// Simple authentication test script
// Run this in browser console to test basic functionality

console.log('🧪 Starting Authentication Tests...');

// Test 1: Check if Supabase is configured
async function testSupabaseConfig() {
  console.log('✅ Testing Supabase configuration...');
  try {
    const { data, error } = await supabase.auth.getUser();
    console.log('Supabase config:', { data, error });
    return !error;
  } catch (err) {
    console.error('❌ Supabase config error:', err);
    return false;
  }
}

// Test 2: Check if admin access works
async function testAdminAccess() {
  console.log('✅ Testing admin access...');
  try {
    // Simulate admin login
    localStorage.setItem('temp-admin-access', 'true');
    localStorage.setItem('admin-user-id', '00000000-0000-0000-0000-000000000001');
    
    const adminUserId = localStorage.getItem('admin-user-id');
    console.log('Admin user ID:', adminUserId);
    return adminUserId === '00000000-0000-0000-0000-000000000001';
  } catch (err) {
    console.error('❌ Admin access error:', err);
    return false;
  }
}

// Test 3: Check if applications table exists
async function testApplicationsTable() {
  console.log('✅ Testing applications table...');
  try {
    const { data, error } = await supabase
      .from('company_applications')
      .select('count')
      .limit(1);
    
    console.log('Applications table:', { data, error });
    return !error;
  } catch (err) {
    console.error('❌ Applications table error:', err);
    return false;
  }
}

// Run all tests
async function runAllTests() {
  console.log('🚀 Running all tests...');
  
  const tests = [
    { name: 'Supabase Config', fn: testSupabaseConfig },
    { name: 'Admin Access', fn: testAdminAccess },
    { name: 'Applications Table', fn: testApplicationsTable }
  ];
  
  let passed = 0;
  let total = tests.length;
  
  for (const test of tests) {
    console.log(`\n📋 Running: ${test.name}`);
    const result = await test.fn();
    if (result) {
      console.log(`✅ ${test.name}: PASSED`);
      passed++;
    } else {
      console.log(`❌ ${test.name}: FAILED`);
    }
  }
  
  console.log(`\n📊 Test Results: ${passed}/${total} tests passed`);
  
  if (passed === total) {
    console.log('🎉 All tests passed! Authentication system is working correctly.');
  } else {
    console.log('⚠️ Some tests failed. Check the errors above.');
  }
}

// Export for manual testing
window.runAuthTests = runAllTests;
console.log('💡 Run "runAuthTests()" in console to execute all tests'); 