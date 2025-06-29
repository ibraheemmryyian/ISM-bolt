const { createClient } = require('@supabase/supabase-js');
const axios = require('axios');

// Supabase configuration
let supabaseUrl = process.env.SUPABASE_URL || 'https://your-project.supabase.co';
const supabaseKey = process.env.SUPABASE_ANON_KEY || 'your-anon-key';

// Remove trailing slash if present
supabaseUrl = supabaseUrl.replace(/\/$/, '');

// Validate configuration
if (!supabaseUrl || supabaseUrl === 'https://your-project.supabase.co') {
  console.error('❌ SUPABASE_URL not configured. Please set the SUPABASE_URL environment variable.');
}

if (!supabaseKey || supabaseKey === 'your-anon-key') {
  console.error('❌ SUPABASE_ANON_KEY not configured. Please set the SUPABASE_ANON_KEY environment variable.');
}

// Debug: Show what we're using
console.log('🔧 Supabase URL:', supabaseUrl);
console.log('🔧 Supabase Key (first 20 chars):', supabaseKey.substring(0, 20) + '...');

// Create custom fetch function using axios
const customFetch = async (url, options = {}) => {
  try {
    console.log('🌐 Making request to:', url);
    
    // Ensure proper headers for Supabase
    const headers = {
      'Content-Type': 'application/json',
      'User-Agent': 'ISM-AI-Backend/1.0.0',
      'apikey': supabaseKey,
      'Authorization': `Bearer ${supabaseKey}`,
      ...options.headers,
    };
    
    console.log('🔑 Using headers:', Object.keys(headers));
    
    const response = await axios({
      method: options.method || 'GET',
      url: url,
      headers: headers,
      data: options.body,
      timeout: 10000, // 10 second timeout
    });

    return {
      ok: response.status >= 200 && response.status < 300,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
      json: async () => response.data,
      text: async () => JSON.stringify(response.data),
    };
  } catch (error) {
    console.error('❌ Axios error:', error.response?.status, error.response?.data);
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout');
    }
    throw error;
  }
};

// Create Supabase client with axios-based fetch
const supabase = createClient(supabaseUrl, supabaseKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: false
  },
  global: {
    fetch: customFetch
  }
});

// Test the connection with better error handling
async function testConnection() {
  try {
    console.log('Testing Supabase connection with axios...');
    const { data, error } = await supabase
      .from('companies')
      .select('count', { count: 'exact', head: true });
    
    if (error) {
      console.error('❌ Supabase connection failed:', error);
      return false;
    }
    
    console.log('✅ Supabase connection successful');
    return true;
  } catch (error) {
    console.error('❌ Supabase connection failed:', error.message);
    return false;
  }
}

// Run connection test
testConnection();

module.exports = { supabase }; 