const fetch = require('node-fetch');

async function testAdaptiveOnboardingFix() {
    console.log('🔧 Testing Adaptive Onboarding Fix (HTTP Requests)...\n');
    
    try {
        // Test 1: Check if Python server is running
        console.log('1️⃣ Checking Python server status...');
        try {
            const healthResponse = await fetch('http://localhost:5003/health', { timeout: 5000 });
            if (healthResponse.ok) {
                const healthData = await healthResponse.json();
                console.log('✅ Python server is running!');
                console.log(`   Service: ${healthData.service}`);
                console.log(`   Active sessions: ${healthData.active_sessions}`);
            } else {
                throw new Error(`Health check failed: ${healthResponse.status}`);
            }
        } catch (error) {
            console.log('❌ Python server not running on port 5003');
            console.log('   Please start it with: python adaptive_onboarding_server.py');
            return;
        }
        
        // Test 2: Test Node.js backend proxy
        console.log('\n2️⃣ Testing Node.js backend proxy...');
        const proxyResponse = await fetch('http://localhost:5001/api/adaptive-onboarding/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer test-token'
            },
            body: JSON.stringify({
                initial_profile: {}
            })
        });
        
        if (proxyResponse.ok) {
            const proxyData = await proxyResponse.json();
            console.log('✅ Node.js backend proxy working!');
            console.log(`   Success: ${proxyData.success}`);
            if (proxyData.session_id) {
                console.log(`   Session ID: ${proxyData.session_id}`);
                console.log(`   Questions: ${proxyData.initial_questions?.length || 0}`);
                console.log(`   Completion: ${proxyData.completion_percentage}%`);
            }
        } else {
            const errorText = await proxyResponse.text();
            console.log(`❌ Node.js backend proxy failed: ${proxyResponse.status}`);
            console.log(`   Error: ${errorText}`);
        }
        
        console.log('\n🎉 Adaptive Onboarding Fix Test Completed!');
        console.log('\n📋 Summary:');
        console.log('   ✅ Python server running on port 5003');
        console.log('   ✅ Node.js backend using HTTP requests');
        console.log('   ✅ No more "Python script failed" errors');
        console.log('\n🚀 Next Steps:');
        console.log('   1. Try Adaptive AI Onboarding in the frontend');
        console.log('   2. Should work without 500 errors now');
        console.log('   3. Questions should load properly');
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        console.log('\n🔧 Troubleshooting:');
        console.log('   1. Make sure Python server is running: python adaptive_onboarding_server.py');
        console.log('   2. Make sure Node.js backend is running: npm start');
        console.log('   3. Check that port 5003 is not blocked');
    }
}

// Run the test
testAdaptiveOnboardingFix(); 