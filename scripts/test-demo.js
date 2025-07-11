// SymbioFlows Demo Test Script
// Tests all critical functionality for the millionaire demo

const axios = require('axios');

const BASE_URL = 'http://localhost:5001';
const FRONTEND_URL = 'http://localhost:5173';

const testResults = {
  passed: 0,
  failed: 0,
  tests: []
};

function logTest(name, passed, details = '') {
  const status = passed ? '✅ PASS' : '❌ FAIL';
  const color = passed ? '\x1b[32m' : '\x1b[31m';
  const reset = '\x1b[0m';
  
  console.log(`${color}${status}${reset} ${name}`);
  if (details) {
    console.log(`   ${details}`);
  }
  
  testResults.tests.push({ name, passed, details });
  if (passed) {
    testResults.passed++;
  } else {
    testResults.failed++;
  }
}

async function testBackendHealth() {
  try {
    const response = await axios.get(`${BASE_URL}/api/health`);
    const passed = response.status === 200 && response.data.status === 'healthy';
    logTest('Backend Health Check', passed, `Status: ${response.data.status}`);
    return passed;
  } catch (error) {
    logTest('Backend Health Check', false, `Error: ${error.message}`);
    return false;
  }
}

async function testSupabaseConnection() {
  try {
    const response = await axios.get(`${BASE_URL}/api/admin/stats`);
    const passed = response.status === 200;
    logTest('Supabase Connection', passed, `Companies: ${response.data.totalCompanies || 'N/A'}`);
    return passed;
  } catch (error) {
    logTest('Supabase Connection', false, `Error: ${error.message}`);
    return false;
  }
}

async function testFreightosAPI() {
  try {
    const response = await axios.get(`${BASE_URL}/api/logistics/preview`, {
      params: {
        origin: 'Dubai, UAE',
        destination: 'Riyadh, Saudi Arabia',
        material: 'Steel Scrap',
        weight: 50000
      }
    });
    const passed = response.status === 200 && response.data.transport_modes;
    logTest('Freightos API Integration', passed, `Modes available: ${response.data.transport_modes?.length || 0}`);
    return passed;
  } catch (error) {
    logTest('Freightos API Integration', false, `Error: ${error.message}`);
    return false;
  }
}

async function testPythonAIServices() {
  const services = [
    { name: 'GNN Reasoning Engine', endpoint: '/api/ai/gnn-reasoning' },
    { name: 'Revolutionary AI Matching', endpoint: '/api/ai/revolutionary-matching' },
    { name: 'Knowledge Graph', endpoint: '/api/ai/knowledge-graph' },
    { name: 'Multi-hop Symbiosis', endpoint: '/api/ai/multi-hop-symbiosis' },
    { name: 'Advanced AI Integration', endpoint: '/api/ai/advanced-integration' }
  ];

  let allPassed = true;
  
  for (const service of services) {
    try {
      const response = await axios.post(`${BASE_URL}${service.endpoint}`, {
        action: 'test',
        data: { test: true }
      });
      const passed = response.status === 200;
      logTest(service.name, passed, `Response: ${response.data.message || 'OK'}`);
      if (!passed) allPassed = false;
    } catch (error) {
      logTest(service.name, false, `Error: ${error.message}`);
      allPassed = false;
    }
  }
  
  return allPassed;
}

async function testAdaptiveOnboarding() {
  try {
    const response = await axios.get(`${BASE_URL}/api/adaptive-onboarding/questions`);
    const passed = response.status === 200 && response.data.questions;
    logTest('Adaptive Onboarding', passed, `Questions loaded: ${response.data.questions?.length || 0}`);
    return passed;
  } catch (error) {
    logTest('Adaptive Onboarding', false, `Error: ${error.message}`);
    return false;
  }
}

async function testAIMatchingEndpoints() {
  const endpoints = [
    { name: 'AI Listings Generation', endpoint: '/api/ai/generate-listings' },
    { name: 'AI Matching Engine', endpoint: '/api/ai/matching' },
    { name: 'Symbiosis Analysis', endpoint: '/api/ai/symbiosis-analysis' }
  ];

  let allPassed = true;
  
  for (const endpoint of endpoints) {
    try {
      const response = await axios.post(`${BASE_URL}${endpoint.endpoint}`, {
        company_id: 'demo-company',
        action: 'test'
      });
      const passed = response.status === 200;
      logTest(endpoint.name, passed, `Status: ${response.status}`);
      if (!passed) allPassed = false;
    } catch (error) {
      logTest(endpoint.name, false, `Error: ${error.message}`);
      allPassed = false;
    }
  }
  
  return allPassed;
}

async function testFrontendAccess() {
  try {
    const response = await axios.get(FRONTEND_URL, { timeout: 5000 });
    const passed = response.status === 200;
    logTest('Frontend Access', passed, `Status: ${response.status}`);
    return passed;
  } catch (error) {
    logTest('Frontend Access', false, `Error: ${error.message}`);
    return false;
  }
}

async function testDemoDataGeneration() {
  try {
    const response = await axios.post(`${BASE_URL}/api/demo/generate-data`, {
      companies_count: 50,
      generate_listings: true,
      run_matching: true
    });
    const passed = response.status === 200;
    logTest('Demo Data Generation', passed, `Companies: ${response.data.companies || 0}, Matches: ${response.data.matches || 0}`);
    return passed;
  } catch (error) {
    logTest('Demo Data Generation', false, `Error: ${error.message}`);
    return false;
  }
}

async function testLogisticsPreview() {
  try {
    const response = await axios.post(`${BASE_URL}/api/logistics/preview`, {
      origin: 'Dubai, UAE',
      destination: 'Riyadh, Saudi Arabia',
      material: 'Steel Scrap',
      weight_kg: 50000,
      include_carbon: true
    });
    
    const passed = response.status === 200 && 
                   response.data.transport_modes && 
                   response.data.total_cost &&
                   response.data.total_carbon;
    
    logTest('Logistics Preview', passed, `Cost: $${response.data.total_cost}, Carbon: ${response.data.total_carbon}kg`);
    return passed;
  } catch (error) {
    logTest('Logistics Preview', false, `Error: ${error.message}`);
    return false;
  }
}

async function testCarbonTracking() {
  try {
    const response = await axios.post(`${BASE_URL}/api/carbon/calculate`, {
      transport_mode: 'truck',
      distance_km: 1200,
      weight_kg: 50000,
      material_type: 'steel'
    });
    
    const passed = response.status === 200 && response.data.carbon_emissions;
    logTest('Carbon Tracking', passed, `Emissions: ${response.data.carbon_emissions}kg CO2`);
    return passed;
  } catch (error) {
    logTest('Carbon Tracking', false, `Error: ${error.message}`);
    return false;
  }
}

async function runAllTests() {
  console.log('🚀 Starting SymbioFlows Demo Tests...\n');
  console.log('=====================================\n');

  // Core Infrastructure Tests
  console.log('📋 Core Infrastructure Tests:');
  console.log('=============================');
  await testBackendHealth();
  await testSupabaseConnection();
  await testFrontendAccess();
  console.log('');

  // AI Services Tests
  console.log('🤖 AI Services Tests:');
  console.log('=====================');
  await testPythonAIServices();
  await testAIMatchingEndpoints();
  await testAdaptiveOnboarding();
  console.log('');

  // Business Logic Tests
  console.log('💼 Business Logic Tests:');
  console.log('=========================');
  await testFreightosAPI();
  await testLogisticsPreview();
  await testCarbonTracking();
  await testDemoDataGeneration();
  console.log('');

  // Summary
  console.log('📊 Test Summary:');
  console.log('================');
  console.log(`✅ Passed: ${testResults.passed}`);
  console.log(`❌ Failed: ${testResults.failed}`);
  console.log(`📈 Success Rate: ${((testResults.passed / (testResults.passed + testResults.failed)) * 100).toFixed(1)}%`);
  console.log('');

  if (testResults.failed === 0) {
    console.log('🎉 All tests passed! Demo is ready for the millionaire meeting! 🚀');
  } else {
    console.log('⚠️  Some tests failed. Please check the issues above before the demo.');
  }

  // Detailed results
  console.log('\n📋 Detailed Results:');
  console.log('===================');
  testResults.tests.forEach(test => {
    const status = test.passed ? '✅' : '❌';
    console.log(`${status} ${test.name}`);
    if (test.details) {
      console.log(`   ${test.details}`);
    }
  });

  return testResults.failed === 0;
}

// Run tests if this script is executed directly
if (require.main === module) {
  runAllTests().then(success => {
    process.exit(success ? 0 : 1);
  }).catch(error => {
    console.error('Test execution failed:', error);
    process.exit(1);
  });
}

module.exports = {
  runAllTests,
  testResults
}; 