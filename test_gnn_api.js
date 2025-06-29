// Test GNN API endpoints
const axios = require('axios');

const testCompanies = [
  {
    id: "test_company_1",
    name: "Test Steel Company",
    industry: "Steel Manufacturing",
    location: "Pittsburgh, PA",
    materials_offered: ["steel_slag", "iron_oxide"],
    materials_needed: ["limestone", "coke"]
  },
  {
    id: "test_company_2",
    name: "Test Cement Company",
    industry: "Cement Production", 
    location: "Portland, OR",
    materials_offered: ["cement_dust", "fly_ash"],
    materials_needed: ["limestone", "clay"]
  }
];

async function testGnnApi() {
  console.log('🧠 Testing GNN API Endpoints...\n');
  
  try {
    // Test 1: GNN Info endpoint
    console.log('1️⃣ Testing GNN Info endpoint...');
    const infoResponse = await axios.get('http://localhost:3001/api/ai-gnn-info');
    console.log('✅ GNN Info endpoint working');
    console.log(`   Available models: ${infoResponse.data.gnnInfo.availableModels.length}`);
    console.log();
    
    // Test 2: GNN Links endpoint
    console.log('2️⃣ Testing GNN Links endpoint...');
    const linksResponse = await axios.post('http://localhost:3001/api/ai-gnn-links', {
      participants: testCompanies,
      modelType: 'gcn',
      topN: 3
    });
    console.log('✅ GNN Links endpoint working');
    console.log(`   Found ${linksResponse.data.gnnResults.links?.length || 0} links`);
    console.log();
    
    // Test 3: GNN Compare endpoint
    console.log('3️⃣ Testing GNN Compare endpoint...');
    const compareResponse = await axios.post('http://localhost:3001/api/ai-gnn-compare', {
      participants: testCompanies,
      models: ['gcn', 'sage'],
      topN: 3
    });
    console.log('✅ GNN Compare endpoint working');
    console.log(`   Compared ${Object.keys(compareResponse.data.comparisonResults).length} models`);
    console.log();
    
    console.log('🎉 All GNN API endpoints are working correctly!');
    
  } catch (error) {
    console.error('❌ GNN API test failed:');
    if (error.response) {
      console.error(`   Status: ${error.response.status}`);
      console.error(`   Data: ${JSON.stringify(error.response.data, null, 2)}`);
    } else {
      console.error(`   Error: ${error.message}`);
    }
  }
}

testGnnApi(); 