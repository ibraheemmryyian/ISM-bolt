// Materials Quality Analysis Script
// Run this in browser console to test your 1,400 materials

console.log('🔍 Starting Materials Quality Analysis...');

// Test 1: Check total materials count
async function testMaterialsCount() {
  console.log('✅ Testing materials count...');
  try {
    const { data, error } = await supabase
      .from('materials')
      .select('count')
      .limit(1);
    
    if (error) throw error;
    
    const { count } = await supabase
      .from('materials')
      .select('*', { count: 'exact', head: true });
    
    console.log(`📊 Total materials in database: ${count}`);
    return count;
  } catch (err) {
    console.error('❌ Error counting materials:', err);
    return 0;
  }
}

// Test 2: Analyze material types distribution
async function testMaterialTypes() {
  console.log('✅ Testing material types distribution...');
  try {
    const { data, error } = await supabase
      .from('materials')
      .select('type, material_name, quantity, unit, description');
    
    if (error) throw error;
    
    const wasteCount = data.filter(m => m.type === 'waste').length;
    const requirementCount = data.filter(m => m.type === 'requirement').length;
    
    console.log(`📦 Waste materials: ${wasteCount}`);
    console.log(`📥 Requirements: ${requirementCount}`);
    console.log(`📊 Ratio: ${(wasteCount/requirementCount).toFixed(2)}:1`);
    
    return { waste: wasteCount, requirements: requirementCount };
  } catch (err) {
    console.error('❌ Error analyzing types:', err);
    return { waste: 0, requirements: 0 };
  }
}

// Test 3: Check material quality indicators
async function testMaterialQuality() {
  console.log('✅ Testing material quality...');
  try {
    const { data, error } = await supabase
      .from('materials')
      .select('material_name, description, quantity, unit')
      .limit(50);
    
    if (error) throw error;
    
    let qualityScore = 0;
    let totalChecked = 0;
    
    data.forEach(material => {
      totalChecked++;
      
      // Check description quality
      if (material.description && material.description.length > 20) qualityScore++;
      
      // Check quantity validity
      if (material.quantity && material.quantity > 0) qualityScore++;
      
      // Check unit validity
      if (material.unit && material.unit !== 'units') qualityScore++;
      
      // Check name quality
      if (material.material_name && material.material_name.length > 3) qualityScore++;
    });
    
    const avgQuality = (qualityScore / (totalChecked * 4)) * 100;
    console.log(`📊 Average quality score: ${avgQuality.toFixed(1)}%`);
    
    return avgQuality;
  } catch (err) {
    console.error('❌ Error checking quality:', err);
    return 0;
  }
}

// Test 4: Test matching algorithm
async function testMatchingAlgorithm() {
  console.log('✅ Testing matching algorithm...');
  try {
    // Get a sample waste material
    const { data: wasteData, error: wasteError } = await supabase
      .from('materials')
      .select('*')
      .eq('type', 'waste')
      .limit(1);
    
    if (wasteError || !wasteData.length) throw new Error('No waste materials found');
    
    const sampleWaste = wasteData[0];
    console.log(`🔍 Testing matches for: ${sampleWaste.material_name}`);
    
    // Find potential matches (requirements that might use this waste)
    const { data: matches, error: matchError } = await supabase
      .from('materials')
      .select('*')
      .eq('type', 'requirement')
      .ilike('material_name', `%${sampleWaste.material_name.split(' ')[0]}%`)
      .limit(5);
    
    if (matchError) throw matchError;
    
    console.log(`🎯 Found ${matches.length} potential matches:`);
    matches.forEach((match, i) => {
      console.log(`  ${i+1}. ${match.material_name} (${match.quantity} ${match.unit})`);
    });
    
    return matches.length;
  } catch (err) {
    console.error('❌ Error testing matching:', err);
    return 0;
  }
}

// Test 5: Check company distribution
async function testCompanyDistribution() {
  console.log('✅ Testing company distribution...');
  try {
    const { data, error } = await supabase
      .from('materials')
      .select('company_id');
    
    if (error) throw error;
    
    const companyCounts = {};
    data.forEach(material => {
      companyCounts[material.company_id] = (companyCounts[material.company_id] || 0) + 1;
    });
    
    const companies = Object.keys(companyCounts).length;
    const avgMaterialsPerCompany = data.length / companies;
    
    console.log(`🏢 Companies with materials: ${companies}`);
    console.log(`📊 Average materials per company: ${avgMaterialsPerCompany.toFixed(1)}`);
    
    return { companies, avgMaterials: avgMaterialsPerCompany };
  } catch (err) {
    console.error('❌ Error checking company distribution:', err);
    return { companies: 0, avgMaterials: 0 };
  }
}

// Run all tests
async function runQualityTests() {
  console.log('🚀 Running Materials Quality Tests...');
  
  const tests = [
    { name: 'Materials Count', fn: testMaterialsCount },
    { name: 'Type Distribution', fn: testMaterialTypes },
    { name: 'Quality Score', fn: testMaterialQuality },
    { name: 'Matching Algorithm', fn: testMatchingAlgorithm },
    { name: 'Company Distribution', fn: testCompanyDistribution }
  ];
  
  const results = {};
  
  for (const test of tests) {
    console.log(`\n📋 Running: ${test.name}`);
    try {
      results[test.name] = await test.fn();
    } catch (err) {
      console.error(`❌ ${test.name} failed:`, err);
      results[test.name] = null;
    }
  }
  
  console.log('\n📊 Final Results:');
  console.log(JSON.stringify(results, null, 2));
  
  // Overall assessment
  const totalMaterials = results['Materials Count'] || 0;
  const qualityScore = results['Quality Score'] || 0;
  const matchCount = results['Matching Algorithm'] || 0;
  
  console.log('\n🎯 Overall Assessment:');
  if (totalMaterials >= 1000) console.log('✅ Excellent: 1000+ materials generated');
  if (qualityScore >= 80) console.log('✅ Excellent: High quality materials');
  if (matchCount > 0) console.log('✅ Good: Matching algorithm working');
  
  if (totalMaterials < 500) console.log('⚠️ Low: Few materials generated');
  if (qualityScore < 60) console.log('⚠️ Poor: Low quality materials');
  if (matchCount === 0) console.log('⚠️ Issue: No matches found');
}

// Export for manual testing
window.runQualityTests = runQualityTests;
console.log('💡 Run "runQualityTests()" in console to analyze your materials'); 