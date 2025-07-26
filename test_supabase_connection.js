// Test Supabase connection and environment variables
const { createClient } = require('@supabase/supabase-js');

// Check environment variables
console.log('🔍 Checking environment variables...');
console.log('VITE_SUPABASE_URL:', process.env.VITE_SUPABASE_URL ? '✅ Set' : '❌ Missing');
console.log('VITE_SUPABASE_ANON_KEY:', process.env.VITE_SUPABASE_ANON_KEY ? '✅ Set' : '❌ Missing');

// Use the same values as in the test_authentication.js
const supabaseUrl = 'https://jifkiwbxnttrkdrdcose.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIzNjM5MTQsImV4cCI6MjA2NzkzOTkxNH0.4PE6Zu0RaMhz3QkocYCQsENS9Tv19avtfXSe_ChHcLA';

console.log('\n🔍 Testing Supabase connection...');

const supabase = createClient(supabaseUrl, supabaseAnonKey);

async function testConnection() {
  try {
    console.log('Testing basic connection...');
    
    // Test 1: Basic query
    const { data, error } = await supabase
      .from('companies')
      .select('count', { count: 'exact', head: true });
    
    if (error) {
      console.error('❌ Basic query failed:', error);
      return false;
    }
    
    console.log('✅ Basic query successful');
    
    // Test 2: Auth service
    console.log('Testing auth service...');
    const { data: authData, error: authError } = await supabase.auth.signUp({
      email: 'test@example.com',
      password: 'testpassword123'
    });
    
    if (authError) {
      if (authError.message.includes('User already registered')) {
        console.log('✅ Auth service working (user already exists)');
      } else {
        console.error('❌ Auth service failed:', authError.message);
        return false;
      }
    } else {
      console.log('✅ Auth service working');
    }
    
    // Test 3: Check if companies table exists and has data
    console.log('Testing companies table...');
    const { data: companies, error: companiesError } = await supabase
      .from('companies')
      .select('id, name, email')
      .limit(1);
    
    if (companiesError) {
      console.error('❌ Companies table query failed:', companiesError);
      return false;
    }
    
    console.log('✅ Companies table accessible');
    console.log(`   Found ${companies?.length || 0} companies`);
    
    return true;
  } catch (error) {
    console.error('❌ Connection test error:', error.message);
    return false;
  }
}

async function runTest() {
  console.log('🚀 Starting Supabase Connection Test\n');
  
  const success = await testConnection();
  
  console.log('\n📊 Test Results:');
  if (success) {
    console.log('🎉 All tests passed! Supabase connection is working correctly.');
    console.log('\n🔧 If you\'re still seeing "load failed" errors:');
    console.log('   1. Check your browser console for specific error messages');
    console.log('   2. Clear browser cache and try again');
    console.log('   3. Try incognito/private browsing mode');
    console.log('   4. Check if your frontend is using the correct environment variables');
  } else {
    console.log('❌ Some tests failed. Check the error messages above.');
    console.log('\n🔧 Troubleshooting:');
    console.log('   1. Verify Supabase project is active and not paused');
    console.log('   2. Check if the API keys are correct');
    console.log('   3. Ensure the companies table exists in your database');
    console.log('   4. Check network connectivity');
  }
}

runTest().catch(console.error); 