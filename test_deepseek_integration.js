// Test DeepSeek R1 Integration
// Run this to verify the AI portfolio generation works with DeepSeek R1

// Load environment variables from .env file
require('dotenv').config();

const axios = require('axios');

// Test DeepSeek R1 API connectivity
async function testDeepSeekR1API() {
  console.log('🧪 Testing DeepSeek R1 API Integration...');
  
  const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;
  
  if (!DEEPSEEK_API_KEY) {
    console.error('❌ DEEPSEEK_API_KEY environment variable not set');
    console.log('💡 Please set your DeepSeek API key:');
    console.log('   export DEEPSEEK_API_KEY=your_api_key_here');
    return false;
  }

  try {
    console.log('📡 Testing DeepSeek R1 API connectivity...');
    
    const response = await axios.post('https://api.deepseek.com/v1/chat/completions', {
      model: 'deepseek-reasoner', // Using DeepSeek R1 reasoning model
      messages: [
        {
          role: 'user',
          content: 'You are an expert industrial symbiosis analyst. Analyze this company for industrial symbiosis opportunities: Manufacturing company in Dubai with plastic waste. Provide analysis in JSON format with waste_potential and estimated_savings, including your reasoning.'
        }
      ],
      temperature: 0.3,
      max_tokens: 300,
      stream: false
    }, {
      headers: {
        'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
        'Content-Type': 'application/json'
      }
    });

    console.log('✅ DeepSeek R1 API connection successful');
    console.log('📝 Response preview:', response.data.choices[0].message.content.substring(0, 150) + '...');
    
    return true;
    
  } catch (error) {
    console.error('❌ DeepSeek R1 API test failed:');
    if (error.response) {
      console.error('   Status:', error.response.status);
      console.error('   Data:', error.response.data);
    } else {
      console.error('   Error:', error.message);
    }
    return false;
  }
}

// Test AI Portfolio Generator
async function testAIPortfolioGenerator() {
  console.log('\n🤖 Testing AI Portfolio Generator with DeepSeek R1...');
  
  try {
    // Import the AI Portfolio Generator
    const { AIPortfolioGenerator } = require('./backend/ai-portfolio-generator');
    const generator = new AIPortfolioGenerator();
    
    // Test company profile
    const testProfile = {
      company_name: 'Test Manufacturing Corp',
      industry: 'Manufacturing',
      location: 'Dubai, UAE',
      products_services: 'Plastic products and packaging',
      main_materials: 'Plastic resins, additives, packaging materials',
      production_processes: 'Injection molding, extrusion, packaging',
      current_waste_streams: 'Plastic scraps, packaging waste, process waste',
      waste_quantities: '500 kg/day plastic scraps, 200 kg/day packaging',
      resource_needs: 'Raw plastic materials, energy, water',
      sustainability_goals: ['Reduce waste by 50%', 'Increase recycling rates'],
      employee_count: '51-200',
      annual_revenue: '$10M-$100M',
      operating_hours: '16 hours/day',
      waste_frequencies: 'Daily',
      energy_consumption: '5000 kWh/day',
      environmental_certifications: 'ISO 14001',
      current_recycling_practices: 'Basic waste segregation',
      partnership_interests: ['Material exchange', 'Waste recycling'],
      geographic_preferences: 'UAE and GCC region',
      technology_interests: 'Advanced recycling technologies'
    };
    
    console.log('📋 Generating portfolio with DeepSeek R1 for test company...');
    const portfolio = await generator.generatePortfolio(testProfile);
    
    console.log('✅ AI Portfolio generation with DeepSeek R1 successful');
    console.log('📊 Generated content:');
    console.log(`   - Materials: ${portfolio.materials.length}`);
    console.log(`   - Opportunities: ${portfolio.opportunities.length}`);
    console.log(`   - Recommendations: ${portfolio.recommendations.length}`);
    console.log(`   - Summary: ${portfolio.summary ? 'Generated' : 'Missing'}`);
    console.log(`   - Analysis: ${portfolio.analysis ? 'Generated' : 'Missing'}`);
    
    // Show reasoning capabilities
    if (portfolio.analysis && portfolio.analysis.reasoning) {
      console.log('\n🧠 DeepSeek R1 Reasoning:');
      console.log(`   ${portfolio.analysis.reasoning.substring(0, 200)}...`);
    }
    
    // Show sample content
    if (portfolio.materials.length > 0) {
      console.log('\n📦 Sample Material:');
      console.log(`   Name: ${portfolio.materials[0].material_name}`);
      console.log(`   Type: ${portfolio.materials[0].type}`);
      console.log(`   Price: $${portfolio.materials[0].price_per_unit}/unit`);
    }
    
    if (portfolio.opportunities.length > 0) {
      console.log('\n🤝 Sample Opportunity:');
      console.log(`   Title: ${portfolio.opportunities[0].title}`);
      console.log(`   Savings: ${portfolio.opportunities[0].estimated_savings}`);
      console.log(`   Impact: ${portfolio.opportunities[0].environmental_impact}`);
    }
    
    if (portfolio.recommendations.length > 0 && portfolio.recommendations[0].reasoning) {
      console.log('\n💡 Sample Recommendation Reasoning:');
      console.log(`   ${portfolio.recommendations[0].reasoning.substring(0, 150)}...`);
    }
    
    return true;
    
  } catch (error) {
    console.error('❌ AI Portfolio Generator test failed:', error.message);
    return false;
  }
}

// Test database schema
async function testDatabaseSchema() {
  console.log('\n🗄️ Testing Database Schema...');
  
  try {
    const { supabase } = require('./backend/supabase');
    
    // Test if new columns exist
    const { data, error } = await supabase
      .from('companies')
      .select('onboarding_completed, current_waste_management, waste_quantity, waste_unit, waste_frequency, user_type')
      .limit(1);
    
    if (error) {
      console.error('❌ Database schema test failed:', error.message);
      console.log('💡 Please run the comprehensive_database_schema.sql script');
      return false;
    }
    
    console.log('✅ Database schema is correct');
    console.log('📋 Available columns:', Object.keys(data[0] || {}));
    
    return true;
    
  } catch (error) {
    console.error('❌ Database test failed:', error.message);
    return false;
  }
}

// Main test function
async function runAllTests() {
  console.log('🚀 Starting DeepSeek R1 Integration Tests\n');
  
  const results = {
    deepseekR1: await testDeepSeekR1API(),
    portfolio: await testAIPortfolioGenerator(),
    database: await testDatabaseSchema()
  };
  
  console.log('\n📊 Test Results:');
  console.log(`   DeepSeek R1 API: ${results.deepseekR1 ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`   Portfolio Generator: ${results.portfolio ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`   Database Schema: ${results.database ? '✅ PASS' : '❌ FAIL'}`);
  
  const allPassed = Object.values(results).every(result => result);
  
  if (allPassed) {
    console.log('\n🎉 All tests passed! DeepSeek R1 integration is working correctly.');
    console.log('\n🧠 DeepSeek R1 Advantages:');
    console.log('   - Advanced reasoning capabilities');
    console.log('   - Detailed analysis with explanations');
    console.log('   - Context-aware recommendations');
    console.log('   - Professional business insights');
    console.log('\n📝 Next steps:');
    console.log('   1. Run the comprehensive database schema script');
    console.log('   2. Add the AI portfolio endpoint to your backend');
    console.log('   3. Replace the onboarding form with AIComprehensiveOnboarding');
    console.log('   4. Test the complete onboarding flow');
  } else {
    console.log('\n⚠️ Some tests failed. Please check the errors above and fix them.');
  }
  
  return allPassed;
}

// Run tests if this file is executed directly
if (require.main === module) {
  runAllTests().catch(console.error);
}

module.exports = { runAllTests, testDeepSeekR1API, testAIPortfolioGenerator, testDatabaseSchema }; 