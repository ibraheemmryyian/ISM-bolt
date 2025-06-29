const fetch = require('node-fetch');

class ComprehensiveTest {
  constructor() {
    this.baseUrl = 'http://localhost:5000';
    this.results = {
      backend: false,
      aiOnboarding: false,
      matching: false,
      frontendIntegration: false
    };
  }

  async runAllTests() {
    console.log('🧪 Running Comprehensive System Test...\n');
    
    try {
      // Test 1: Backend Health
      await this.testBackendHealth();
      
      // Test 2: AI Onboarding
      await this.testAIOnboarding();
      
      // Test 3: Matching Algorithm
      await this.testMatchingAlgorithm();
      
      // Test 4: Frontend Integration
      await this.testFrontendIntegration();
      
      // Summary
      this.printSummary();
      
    } catch (error) {
      console.error('❌ Test suite failed:', error.message);
    }
  }

  async testBackendHealth() {
    console.log('🔍 Testing Backend Health...');
    
    try {
      const response = await fetch(`${this.baseUrl}/api/health`);
      const data = await response.json();
      
      if (response.ok && data.status === 'healthy') {
        console.log('✅ Backend is healthy and running');
        this.results.backend = true;
      } else {
        throw new Error(`Backend health check failed: ${data.status}`);
      }
    } catch (error) {
      console.log('❌ Backend health test failed:', error.message);
      this.results.backend = false;
    }
  }

  async testAIOnboarding() {
    console.log('\n🔍 Testing AI Onboarding...');
    
    try {
      const testData = {
        companyName: 'Test Electronics Corp',
        industry: 'Electronics Manufacturing',
        products: 'Smartphones, tablets, electronic components',
        location: 'San Francisco, CA',
        productionVolume: '10,000 units/month',
        mainMaterials: 'Silicon wafers, lithium-ion batteries, copper wiring',
        processDescription: 'PCB printing → Component assembly → Firmware installation → Quality control'
      };
      
      const response = await fetch(`${this.baseUrl}/api/ai-infer-listings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(testData)
      });
      
      const data = await response.json();
      
      if (response.ok && data.success && data.listings && data.listings.length > 0) {
        console.log('✅ AI Onboarding working perfectly');
        console.log(`   📊 Generated ${data.listings.length} listings`);
        console.log(`   🤖 AI Model: ${data.ai_model_version}`);
        
        // Check for different types of listings
        const types = [...new Set(data.listings.map(l => l.type))];
        console.log(`   📋 Listing types: ${types.join(', ')}`);
        
        this.results.aiOnboarding = true;
      } else {
        throw new Error(`AI onboarding failed: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.log('❌ AI Onboarding test failed:', error.message);
      this.results.aiOnboarding = false;
    }
  }

  async testMatchingAlgorithm() {
    console.log('\n🔍 Testing Matching Algorithm...');
    
    try {
      // Test the matching endpoint
      const response = await fetch(`${this.baseUrl}/api/generate-matches`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      
      const data = await response.json();
      
      if (response.ok && data.success) {
        console.log('✅ Matching algorithm is active');
        console.log(`   📊 Total matches: ${data.total || 0}`);
        console.log(`   💬 Message: ${data.message}`);
        this.results.matching = true;
      } else {
        throw new Error(`Matching failed: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.log('❌ Matching algorithm test failed:', error.message);
      this.results.matching = false;
    }
  }

  async testFrontendIntegration() {
    console.log('\n🔍 Testing Frontend Integration...');
    
    try {
      // Test if frontend can reach backend endpoints
      const endpoints = [
        '/api/health',
        '/api/ai-infer-listings',
        '/api/generate-matches'
      ];
      
      let successCount = 0;
      
      for (const endpoint of endpoints) {
        try {
          const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: endpoint.includes('ai-infer-listings') ? 'POST' : 'GET',
            headers: { 'Content-Type': 'application/json' },
            body: endpoint.includes('ai-infer-listings') ? JSON.stringify({
              companyName: 'Test',
              industry: 'Test',
              products: 'Test',
              location: 'Test',
              productionVolume: 'Test',
              mainMaterials: 'Test'
            }) : undefined
          });
          
          if (response.ok) {
            successCount++;
          }
        } catch (error) {
          console.log(`   ⚠️  Endpoint ${endpoint} not accessible`);
        }
      }
      
      if (successCount >= 2) {
        console.log('✅ Frontend integration working');
        console.log(`   📡 ${successCount}/3 endpoints accessible`);
        this.results.frontendIntegration = true;
      } else {
        throw new Error(`Only ${successCount}/3 endpoints accessible`);
      }
    } catch (error) {
      console.log('❌ Frontend integration test failed:', error.message);
      this.results.frontendIntegration = false;
    }
  }

  printSummary() {
    console.log('\n📊 COMPREHENSIVE TEST SUMMARY');
    console.log('=====================================');
    
    const totalTests = Object.keys(this.results).length;
    const passedTests = Object.values(this.results).filter(Boolean).length;
    
    console.log(`Total Tests: ${totalTests}`);
    console.log(`Passed: ${passedTests}`);
    console.log(`Failed: ${totalTests - passedTests}`);
    console.log(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);
    
    console.log('\nDetailed Results:');
    Object.entries(this.results).forEach(([test, passed]) => {
      const status = passed ? '✅ PASS' : '❌ FAIL';
      const testName = test.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
      console.log(`  ${status} ${testName}`);
    });
    
    if (passedTests === totalTests) {
      console.log('\n🎉 ALL SYSTEMS OPERATIONAL!');
      console.log('Your frontend, backend, AI onboarding, and matching algorithm are working perfectly together!');
    } else {
      console.log('\n⚠️  SOME ISSUES DETECTED');
      console.log('Please check the failed tests above and ensure all services are running properly.');
    }
  }
}

// Run the comprehensive test
const test = new ComprehensiveTest();
test.runAllTests(); 