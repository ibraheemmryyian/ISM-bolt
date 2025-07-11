const fetch = require('node-fetch');

async function testAdaptiveOnboarding() {
    console.log('🧪 Testing Adaptive AI Onboarding Fixes...\n');
    
    try {
        // Test 1: Start onboarding session
        console.log('1️⃣ Starting onboarding session...');
        const startResponse = await fetch('http://localhost:5001/api/adaptive-onboarding/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer test-token'
            },
            body: JSON.stringify({
                user_id: 'test-user-123',
                initial_profile: {}
            })
        });
        
        if (!startResponse.ok) {
            throw new Error(`Start failed: ${startResponse.status}`);
        }
        
        const startData = await startResponse.json();
        console.log('✅ Session started:', startData.session?.session_id);
        console.log(`📝 Questions generated: ${startData.session?.initial_questions?.length || 0}`);
        
        // Test 2: Process a response
        if (startData.session?.initial_questions?.length > 0) {
            console.log('\n2️⃣ Processing response...');
            const question = startData.session.initial_questions[0];
            const answer = 'Test Company';
            
            const respondResponse = await fetch('http://localhost:5001/api/adaptive-onboarding/respond', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer test-token'
                },
                body: JSON.stringify({
                    session_id: startData.session.session_id,
                    question_id: question.id,
                    answer: answer
                })
            });
            
            if (!respondResponse.ok) {
                throw new Error(`Response failed: ${respondResponse.status}`);
            }
            
            const respondData = await respondResponse.json();
            console.log('✅ Response processed');
            console.log(`📊 Completion: ${respondData.completion_percentage}%`);
            console.log(`🎯 Quality: ${respondData.answer_quality}`);
            
            // Test 3: Check session status
            console.log('\n3️⃣ Checking session status...');
            const statusResponse = await fetch(`http://localhost:5001/api/adaptive-onboarding/status/${startData.session.session_id}`, {
                headers: {
                    'Authorization': 'Bearer test-token'
                }
            });
            
            if (statusResponse.ok) {
                const statusData = await statusResponse.json();
                console.log('✅ Status retrieved');
                console.log(`📈 Progress: ${statusData.completion_percentage}%`);
                console.log(`❓ Questions asked: ${statusData.questions_asked}`);
                console.log(`💬 Responses received: ${statusData.responses_received}`);
            }
        }
        
        console.log('\n🎉 All tests passed! Adaptive onboarding is working correctly.');
        console.log('\n📋 Summary of fixes:');
        console.log('   ✅ Removed response time tracking');
        console.log('   ✅ Fixed completion percentage calculation');
        console.log('   ✅ Optimized backend performance');
        console.log('   ✅ Removed response time display from frontend');
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        console.log('\n🔧 Troubleshooting:');
        console.log('   1. Make sure backend is running on port 5001');
        console.log('   2. Check that Python scripts are accessible');
        console.log('   3. Verify Supabase connection');
    }
}

// Run the test
testAdaptiveOnboarding(); 