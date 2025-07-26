// Test script to diagnose authentication issues
const { createClient } = require('@supabase/supabase-js');

// Test configuration
const supabaseUrl = 'https://jifkiwbxnttrkdrdcose.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImppZmtpd2J4bnR0cmtkcmRjb3NlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIzNjM5MTQsImV4cCI6MjA2NzkzOTkxNH0.4PE6Zu0RaMhz3QkocYCQsENS9Tv19avtfXSe_ChHcLA';

const supabase = createClient(supabaseUrl, supabaseAnonKey);

async function testConnection() {
  console.log('🔍 Testing Supabase connection...');
  
  try {
    // Test basic connection
    const { data, error } = await supabase
      .from('companies')
      .select('count', { count: 'exact', head: true });
    
    if (error) {
      console.error('❌ Connection failed:', error);
      return false;
    }
    
    console.log('✅ Connection successful');
    return true;
  } catch (error) {
    console.error('❌ Connection error:', error.message);
    return false;
  }
}

async function testAuth() {
  console.log('\n🔐 Testing authentication...');
  
  try {
    // Test sign up (this will fail if email already exists, which is expected)
    const { data, error } = await supabase.auth.signUp({
      email: 'test@example.com',
      password: 'testpassword123'
    });
    
    if (error) {
      if (error.message.includes('User already registered')) {
        console.log('✅ Authentication service working (user already exists)');
        return true;
      } else {
        console.error('❌ Authentication failed:', error.message);
        return false;
      }
    }
    
    console.log('✅ Authentication service working');
    return true;
  } catch (error) {
    console.error('❌ Authentication error:', error.message);
    return false;
  }
}

async function testDatabase() {
  console.log('\n💾 Testing database access...');
  
  try {
    // Test reading from companies table
    const { data, error } = await supabase
      .from('companies')
      .select('id, name, email')
      .limit(1);
    
    if (error) {
      console.error('❌ Database access failed:', error);
      return false;
    }
    
    console.log('✅ Database access successful');
    console.log(`   Found ${data?.length || 0} companies`);
    return true;
  } catch (error) {
    console.error('❌ Database error:', error.message);
    return false;
  }
}

async function runTests() {
  console.log('🚀 Starting SymbioFlows Authentication Diagnostics\n');
  
  const connectionOk = await testConnection();
  const authOk = await testAuth();
  const dbOk = await testDatabase();
  
  console.log('\n📊 Test Results:');
  console.log(`   Connection: ${connectionOk ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`   Authentication: ${authOk ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`   Database: ${dbOk ? '✅ PASS' : '❌ FAIL'}`);
  
  if (connectionOk && authOk && dbOk) {
    console.log('\n🎉 All tests passed! The authentication system should work correctly.');
  } else {
    console.log('\n⚠️  Some tests failed. Check the error messages above for details.');
    console.log('\n🔧 Troubleshooting tips:');
    console.log('   1. Check your internet connection');
    console.log('   2. Verify Supabase URL and API key are correct');
    console.log('   3. Ensure Supabase project is active and not paused');
    console.log('   4. Check if there are any CORS issues');
  }
}

// Run the tests
runTests().catch(console.error); 