require('dotenv').config();
const { supabase } = require('./supabase');

async function testSupabaseConnection() {
  console.log('Testing Supabase connection...');
  console.log('SUPABASE_URL:', process.env.SUPABASE_URL);
  console.log('SUPABASE_ANON_KEY:', process.env.SUPABASE_ANON_KEY ? 'Set' : 'Not set');
  
  try {
    // Test basic connection
    const { data, error } = await supabase
      .from('companies')
      .select('count', { count: 'exact', head: true });
    
    if (error) {
      console.error('❌ Supabase connection failed:', error);
      return;
    }
    
    console.log('✅ Basic connection successful');
    
    // Test inserting a company
    const testCompany = {
      id: 'test-' + Date.now(),
      name: 'Test Company',
      email: 'test@example.com',
      role: 'user',
      created_at: new Date().toISOString()
    };
    
    console.log('Testing company insertion...');
    const { data: insertData, error: insertError } = await supabase
      .from('companies')
      .insert(testCompany)
      .select()
      .single();
    
    if (insertError) {
      console.error('❌ Company insertion failed:', insertError);
      return;
    }
    
    console.log('✅ Company insertion successful:', insertData);
    
    // Clean up test data
    await supabase
      .from('companies')
      .delete()
      .eq('id', testCompany.id);
    
    console.log('✅ Test completed successfully');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

testSupabaseConnection(); 