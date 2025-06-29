const { supabase } = require('./supabase');

async function testMaterialsTable() {
  console.log('🧪 Testing materials table...');
  
  try {
    // Test 1: Check if table exists
    console.log('1. Checking if materials table exists...');
    const { data: existingMaterials, error: selectError } = await supabase
      .from('materials')
      .select('*')
      .limit(1);
    
    if (selectError) {
      console.error('❌ Materials table error:', selectError);
      return;
    }
    
    console.log('✅ Materials table exists');
    
    // Test 2: Try to insert a test material
    console.log('2. Testing material insertion...');
    const testMaterial = {
      material_name: 'Test Material',
      type: 'waste',
      quantity: '100',
      unit: 'tons',
      description: 'Test material for database validation',
      company_id: 'test-company-id',
      created_at: new Date().toISOString()
    };
    
    const { data: insertedMaterial, error: insertError } = await supabase
      .from('materials')
      .insert(testMaterial)
      .select()
      .single();
    
    if (insertError) {
      console.error('❌ Material insertion error:', insertError);
      return;
    }
    
    console.log('✅ Material inserted successfully:', insertedMaterial.id);
    
    // Test 3: Clean up test material
    console.log('3. Cleaning up test material...');
    const { error: deleteError } = await supabase
      .from('materials')
      .delete()
      .eq('id', insertedMaterial.id);
    
    if (deleteError) {
      console.error('❌ Material deletion error:', deleteError);
      return;
    }
    
    console.log('✅ Test material cleaned up');
    console.log('🎉 All tests passed! Materials table is working correctly.');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
  }
}

// Run the test
testMaterialsTable(); 