const axios = require('axios');

// Test configuration
const BASE_URL = 'http://localhost:3000';
const TEST_TIMEOUT = 10000;

// Colors for console output
const colors = {
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    reset: '\x1b[0m',
    bold: '\x1b[1m'
};

// Test results tracking
let passedTests = 0;
let failedTests = 0;
let totalTests = 0;

// Helper function to log test results
function logTest(testName, passed, message = '') {
    totalTests++;
    if (passed) {
        passedTests++;
        console.log(`${colors.green}✅ PASS${colors.reset} ${testName} ${message}`);
    } else {
        failedTests++;
        console.log(`${colors.red}❌ FAIL${colors.reset} ${testName} ${message}`);
    }
}

// Helper function to make HTTP requests with timeout
async function makeRequest(method, endpoint, data = null, headers = {}) {
    try {
        const config = {
            method,
            url: `${BASE_URL}${endpoint}`,
            timeout: TEST_TIMEOUT,
            headers: {
                'Content-Type': 'application/json',
                ...headers
            }
        };
        
        if (data) {
            config.data = data;
        }
        
        const response = await axios(config);
        return { success: true, data: response.data, status: response.status };
    } catch (error) {
        return { 
            success: false, 
            error: error.message, 
            status: error.response?.status,
            data: error.response?.data 
        };
    }
}

// Test functions
async function testHealthEndpoint() {
    console.log('\n🔍 Testing Health Endpoint...');
    
    const result = await makeRequest('GET', '/api/health');
    
    if (result.success && result.status === 200) {
        logTest('Health Check', true, `Status: ${result.data.status}`);
        
        // Check required fields
        const hasStatus = result.data.hasOwnProperty('status');
        const hasTimestamp = result.data.hasOwnProperty('timestamp');
        const hasVersion = result.data.hasOwnProperty('version');
        
        logTest('Health Response Fields', hasStatus && hasTimestamp && hasVersion, 
            `Fields: status=${hasStatus}, timestamp=${hasTimestamp}, version=${hasVersion}`);
    } else {
        logTest('Health Check', false, `Error: ${result.error}`);
    }
}

async function testCompanyEndpoint() {
    console.log('\n🏢 Testing Company Endpoint (Authentication Required)...');
    
    // Test without authentication (should fail)
    const result = await makeRequest('GET', '/api/companies/current');
    
    if (!result.success && result.status === 401) {
        logTest('Company Endpoint Auth Required', true, 'Correctly requires authentication');
    } else {
        logTest('Company Endpoint Auth Required', false, 
            `Expected 401, got ${result.status}: ${result.error}`);
    }
    
    // Test with invalid token (should fail)
    const invalidResult = await makeRequest('GET', '/api/companies/current', null, {
        'Authorization': 'Bearer invalid-token'
    });
    
    if (!invalidResult.success && invalidResult.status === 401) {
        logTest('Company Endpoint Invalid Token', true, 'Correctly rejects invalid token');
    } else {
        logTest('Company Endpoint Invalid Token', false, 
            `Expected 401, got ${invalidResult.status}: ${invalidResult.error}`);
    }
}

async function testPortfolioEndpoint() {
    console.log('\n📊 Testing Portfolio Endpoint (Authentication Required)...');
    
    // Test without authentication (should fail)
    const result = await makeRequest('GET', '/api/portfolio');
    
    if (!result.success && result.status === 401) {
        logTest('Portfolio Endpoint Auth Required', true, 'Correctly requires authentication');
    } else {
        logTest('Portfolio Endpoint Auth Required', false, 
            `Expected 401, got ${result.status}: ${result.error}`);
    }
}

async function testAIInsightsEndpoint() {
    console.log('\n🤖 Testing AI Insights Endpoint...');
    
    // Test without company_id (should fail)
    const result = await makeRequest('GET', '/api/ai-insights');
    
    if (!result.success && result.status === 400) {
        logTest('AI Insights Company ID Required', true, 'Correctly requires company_id');
    } else {
        logTest('AI Insights Company ID Required', false, 
            `Expected 400, got ${result.status}: ${result.error}`);
    }
    
    // Test with company_id (should work but return empty if no data)
    const withCompanyResult = await makeRequest('GET', '/api/ai-insights?company_id=test-company');
    
    if (withCompanyResult.success && result.status === 200) {
        logTest('AI Insights With Company ID', true, 'Endpoint responds correctly');
        
        // Check response structure
        const hasInsights = withCompanyResult.data.hasOwnProperty('insights');
        const isArray = Array.isArray(withCompanyResult.data.insights);
        
        logTest('AI Insights Response Structure', hasInsights && isArray, 
            `Structure: insights=${hasInsights}, array=${isArray}`);
    } else {
        logTest('AI Insights With Company ID', false, 
            `Error: ${withCompanyResult.error}`);
    }
}

async function testLoggingEndpoints() {
    console.log('\n📝 Testing Logging Endpoints...');
    
    // Test frontend logging
    const logData = {
        timestamp: new Date().toISOString(),
        level: 'info',
        message: 'Test log message',
        data: { test: true },
        userAgent: 'Test-Agent/1.0',
        url: '/test'
    };
    
    const logResult = await makeRequest('POST', '/api/logs', logData);
    
    if (logResult.success && logResult.status === 200) {
        logTest('Frontend Logging', true, 'Logging endpoint works');
    } else {
        logTest('Frontend Logging', false, `Error: ${logResult.error}`);
    }
    
    // Test error logging
    const errorData = {
        errorId: 'test-error-123',
        message: 'Test error message',
        stack: 'Error stack trace',
        componentStack: 'React component stack',
        userAgent: 'Test-Agent/1.0',
        url: '/test',
        timestamp: new Date().toISOString()
    };
    
    const errorResult = await makeRequest('POST', '/api/logs/error', errorData);
    
    if (errorResult.success && errorResult.status === 200) {
        logTest('Error Logging', true, 'Error logging endpoint works');
    } else {
        logTest('Error Logging', false, `Error: ${errorResult.error}`);
    }
}

async function testAIMatchingEndpoint() {
    console.log('\n🔗 Testing AI Matching Endpoint...');
    
    const matchData = {
        buyer: {
            id: 'test-buyer',
            industry: 'Manufacturing',
            needs: ['steel', 'aluminum'],
            location: 'Test City, ST'
        },
        seller: {
            id: 'test-seller',
            industry: 'Metallurgy',
            products: ['steel', 'aluminum', 'copper'],
            location: 'Test City, ST'
        }
    };
    
    const result = await makeRequest('POST', '/api/match', matchData);
    
    if (result.success && result.status === 200) {
        logTest('AI Matching', true, 'Matching endpoint works');
        
        // Check response structure
        const hasMatchScore = result.data.hasOwnProperty('match_score');
        const hasCompatibility = result.data.hasOwnProperty('compatibility_factors');
        const hasRecommendations = result.data.hasOwnProperty('recommendations');
        
        logTest('AI Matching Response Structure', hasMatchScore && hasCompatibility && hasRecommendations,
            `Structure: match_score=${hasMatchScore}, compatibility=${hasCompatibility}, recommendations=${hasRecommendations}`);
    } else {
        logTest('AI Matching', false, `Error: ${result.error}`);
    }
}

async function testValidationEndpoints() {
    console.log('\n✅ Testing Validation Endpoints...');
    
    // Test AI inference with invalid data
    const invalidData = {
        companyName: '', // Empty required field
        industry: 'Manufacturing'
    };
    
    const result = await makeRequest('POST', '/api/ai-infer-listings', invalidData);
    
    if (!result.success && result.status === 400) {
        logTest('AI Inference Validation', true, 'Correctly validates required fields');
    } else {
        logTest('AI Inference Validation', false, 
            `Expected 400, got ${result.status}: ${result.error}`);
    }
    
    // Test user feedback with invalid rating
    const invalidFeedback = {
        matchId: 'test-match',
        userId: 'test-user',
        rating: 6, // Invalid: should be 1-5
        feedback: 'Test feedback'
    };
    
    const feedbackResult = await makeRequest('POST', '/api/feedback', invalidFeedback);
    
    if (!feedbackResult.success && feedbackResult.status === 400) {
        logTest('User Feedback Validation', true, 'Correctly validates rating range');
    } else {
        logTest('User Feedback Validation', false, 
            `Expected 400, got ${feedbackResult.status}: ${feedbackResult.error}`);
    }
}

// Main test runner
async function runAllTests() {
    console.log(`${colors.bold}${colors.blue}🚀 ISM AI Platform - Backend Health Check${colors.reset}`);
    console.log(`${colors.blue}Testing backend after hard-coded data removal${colors.reset}`);
    console.log(`${colors.blue}Base URL: ${BASE_URL}${colors.reset}`);
    console.log(`${colors.blue}Timeout: ${TEST_TIMEOUT}ms${colors.reset}`);
    
    try {
        await testHealthEndpoint();
        await testCompanyEndpoint();
        await testPortfolioEndpoint();
        await testAIInsightsEndpoint();
        await testLoggingEndpoints();
        await testAIMatchingEndpoint();
        await testValidationEndpoints();
        
        // Print summary
        console.log('\n' + '='.repeat(60));
        console.log(`${colors.bold}📊 TEST SUMMARY${colors.reset}`);
        console.log(`${colors.green}✅ Passed: ${passedTests}${colors.reset}`);
        console.log(`${colors.red}❌ Failed: ${failedTests}${colors.reset}`);
        console.log(`${colors.blue}📋 Total: ${totalTests}${colors.reset}`);
        
        const successRate = ((passedTests / totalTests) * 100).toFixed(1);
        console.log(`${colors.bold}📈 Success Rate: ${successRate}%${colors.reset}`);
        
        if (failedTests === 0) {
            console.log(`\n${colors.green}${colors.bold}🎉 ALL TESTS PASSED! Backend is healthy and ready!${colors.reset}`);
        } else {
            console.log(`\n${colors.yellow}${colors.bold}⚠️  Some tests failed. Check the issues above.${colors.reset}`);
        }
        
    } catch (error) {
        console.error(`${colors.red}❌ Test runner error: ${error.message}${colors.reset}`);
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    runAllTests();
}

module.exports = {
    runAllTests,
    testHealthEndpoint,
    testCompanyEndpoint,
    testPortfolioEndpoint,
    testAIInsightsEndpoint,
    testLoggingEndpoints,
    testAIMatchingEndpoint,
    testValidationEndpoints
}; 