// Simple GNN API Test (no external dependencies)
const http = require('http');

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

function makeRequest(method, path, data = null) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'localhost',
      port: 5000,
      path: path,
      method: method,
      headers: {
        'Content-Type': 'application/json',
      }
    };

    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => {
        body += chunk;
      });
      res.on('end', () => {
        try {
          const result = JSON.parse(body);
          resolve({ status: res.statusCode, data: result });
        } catch (e) {
          resolve({ status: res.statusCode, data: body });
        }
      });
    });

    req.on('error', (err) => {
      reject(err);
    });

    if (data) {
      req.write(JSON.stringify(data));
    }
    req.end();
  });
}

async function testGnnApi() {
  console.log('🧠 Testing GNN API Endpoints...\n');
  
  try {
    // Test 1: GNN Info endpoint
    console.log('1️⃣ Testing GNN Info endpoint...');
    const infoResponse = await makeRequest('GET', '/api/ai-gnn-info');
    if (infoResponse.status === 200) {
      console.log('✅ GNN Info endpoint working');
      console.log(`   Available models: ${infoResponse.data.gnnInfo?.availableModels?.length || 0}`);
    } else {
      console.log(`❌ GNN Info failed: ${infoResponse.status}`);
    }
    console.log();
    
    // Test 2: GNN Links endpoint
    console.log('2️⃣ Testing GNN Links endpoint...');
    const linksResponse = await makeRequest('POST', '/api/ai-gnn-links', {
      participants: testCompanies,
      modelType: 'gcn',
      topN: 3
    });
    if (linksResponse.status === 200) {
      console.log('✅ GNN Links endpoint working');
      console.log(`   Found ${linksResponse.data.gnnResults?.links?.length || 0} links`);
    } else {
      console.log(`❌ GNN Links failed: ${linksResponse.status}`);
      console.log(`   Error: ${JSON.stringify(linksResponse.data, null, 2)}`);
    }
    console.log();
    
    // Test 3: GNN Compare endpoint
    console.log('3️⃣ Testing GNN Compare endpoint...');
    const compareResponse = await makeRequest('POST', '/api/ai-gnn-compare', {
      participants: testCompanies,
      models: ['gcn', 'sage'],
      topN: 3
    });
    if (compareResponse.status === 200) {
      console.log('✅ GNN Compare endpoint working');
      console.log(`   Compared ${Object.keys(compareResponse.data.comparisonResults || {}).length} models`);
    } else {
      console.log(`❌ GNN Compare failed: ${compareResponse.status}`);
      console.log(`   Error: ${JSON.stringify(compareResponse.data, null, 2)}`);
    }
    console.log();
    
    console.log('🎉 GNN API test completed!');
    
  } catch (error) {
    console.error('❌ GNN API test failed:');
    console.error(`   Error: ${error.message}`);
    console.error('\n💡 Make sure the backend server is running on port 3001');
  }
}

testGnnApi(); 