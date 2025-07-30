#!/usr/bin/env node

// Load environment variables
require('dotenv').config();

console.log('🔧 Backend Test Script');
console.log('======================');

// Test environment variables
console.log('\n📋 Environment Variables:');
console.log('SUPABASE_URL:', process.env.SUPABASE_URL ? 'Available' : 'Missing');
console.log('SUPABASE_ANON_KEY:', process.env.SUPABASE_ANON_KEY ? 'Available' : 'Missing');
console.log('DEEPSEEK_API_KEY:', process.env.DEEPSEEK_API_KEY ? 'Available' : 'Missing');
console.log('OPENAI_API_KEY:', process.env.OPENAI_API_KEY ? 'Available' : 'Missing');

// Test basic imports
console.log('\n📦 Testing Imports:');
try {
  const express = require('express');
  console.log('✅ Express: OK');
} catch (e) {
  console.log('❌ Express:', e.message);
}

try {
  const { createClient } = require('@supabase/supabase-js');
  console.log('✅ Supabase client: OK');
} catch (e) {
  console.log('❌ Supabase client:', e.message);
}

// Test Supabase connection
console.log('\n🔌 Testing Supabase Connection:');
try {
  const { supabase } = require('./supabase');
  console.log('✅ Supabase config: OK');
  
  // Test a simple query
  supabase
    .from('users')
    .select('count')
    .limit(1)
    .then(({ data, error }) => {
      if (error) {
        console.log('❌ Supabase query error:', error.message);
      } else {
        console.log('✅ Supabase connection: OK');
      }
    })
    .catch(err => {
      console.log('❌ Supabase connection failed:', err.message);
    });
} catch (e) {
  console.log('❌ Supabase config error:', e.message);
}

// Test basic server creation
console.log('\n🚀 Testing Basic Server:');
try {
  const express = require('express');
  const app = express();
  
  app.get('/test', (req, res) => {
    res.json({ status: 'OK', message: 'Test endpoint working' });
  });
  
  const server = app.listen(3001, () => {
    console.log('✅ Basic server started on port 3001');
    server.close(() => {
      console.log('✅ Basic server test completed');
    });
  });
} catch (e) {
  console.log('❌ Basic server error:', e.message);
}

console.log('\n✅ Backend test completed!');