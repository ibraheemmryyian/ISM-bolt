const fetch = require('node-fetch');

async function testBackendPerformance() {
  console.log('🚀 Testing Backend Performance...\n');
  
  const testData = {
    companyName: 'Performance Test Company',
    industry: 'Electronics Manufacturing',
    products: 'Smartphones, tablets, electronic components',
    location: 'San Francisco, CA',
    productionVolume: '10,000 units/month',
    mainMaterials: 'Silicon wafers, lithium-ion batteries, copper wiring',
    processDescription: 'PCB printing → Component assembly → Firmware installation → Quality control'
  };
  
  console.log('📤 Sending test data:', JSON.stringify(testData, null, 2));
  
  const startTime = Date.now();
  
  try {
    const response = await fetch('http://localhost:5000/api/ai-infer-listings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testData)
    });
    
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    console.log(`⏱️  Response time: ${duration}ms`);
    console.log(`📊 Status: ${response.status}`);
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ Success!');
      console.log(`📋 Generated ${data.listings?.length || 0} listings`);
      console.log(`🤖 AI Model: ${data.ai_model_version}`);
      console.log(`💬 Message: ${data.message}`);
      
      if (data.listings && data.listings.length > 0) {
        console.log('\n📝 Sample listings:');
        data.listings.slice(0, 3).forEach((listing, index) => {
          console.log(`  ${index + 1}. ${listing.material_name} (${listing.material_type}) - ${listing.quantity} ${listing.unit}`);
        });
      }
    } else {
      const errorData = await response.json().catch(() => ({}));
      console.log('❌ Error:', errorData);
    }
    
  } catch (error) {
    console.log('❌ Network error:', error.message);
  }
  
  console.log('\n🎯 Performance Analysis:');
  if (duration < 2000) {
    console.log('✅ Excellent performance (< 2 seconds)');
  } else if (duration < 5000) {
    console.log('⚠️  Acceptable performance (2-5 seconds)');
  } else {
    console.log('❌ Slow performance (> 5 seconds) - needs optimization');
  }
}

testBackendPerformance(); 