const fetch = require('node-fetch');

async function testAIInference() {
  const testData = {
    companyName: "Test Steel Corp",
    industry: "Manufacturing",
    products: "Steel products, metal alloys",
    location: "Pittsburgh, USA",
    productionVolume: "50000 tons/year",
    mainMaterials: "Iron ore, coal, limestone",
    processDescription: "Steel manufacturing using blast furnace process"
  };

  try {
    console.log('Testing AI inference endpoint...');
    console.log('Test data:', JSON.stringify(testData, null, 2));

    const response = await fetch('http://localhost:5000/api/ai-infer-listings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(testData)
    });

    console.log('Response status:', response.status);
    console.log('Response headers:', response.headers);

    if (response.ok) {
      const data = await response.json();
      console.log('✅ AI inference successful!');
      console.log('Response:', JSON.stringify(data, null, 2));
    } else {
      const errorText = await response.text();
      console.log('❌ AI inference failed!');
      console.log('Error response:', errorText);
    }
  } catch (error) {
    console.log('❌ Connection error:', error.message);
    console.log('Make sure the backend server is running on port 5000');
  }
}

// Test health endpoint first
async function testHealth() {
  try {
    console.log('Testing health endpoint...');
    const response = await fetch('http://localhost:5000/api/health');
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ Backend is running!');
      console.log('Health status:', data);
      return true;
    } else {
      console.log('❌ Backend health check failed');
      return false;
    }
  } catch (error) {
    console.log('❌ Cannot connect to backend:', error.message);
    console.log('Please start the backend server with: cd backend && npm start');
    return false;
  }
}

async function runTests() {
  console.log('🚀 Testing Backend API...\n');
  
  const isHealthy = await testHealth();
  
  if (isHealthy) {
    console.log('\n--- Testing AI Inference ---');
    await testAIInference();
  }
  
  console.log('\n✨ Test completed!');
}

runTests(); 