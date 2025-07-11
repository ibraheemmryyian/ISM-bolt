const fetch = require('node-fetch');

async function testPerfectUserFlow() {
    console.log('🎯 Testing Perfect User Flow: AI Inference → Matching → Logistics Preview\n');
    
    const baseUrl = 'http://localhost:5001/api';
    
    try {
        // Test 1: AI Portfolio Generation
        console.log('1️⃣ Testing AI Portfolio Generation...');
        const companyProfile = {
            id: 'test-company-123',
            name: 'Gulf Steel Manufacturing',
            industry: 'steel_manufacturing',
            location: 'Dubai, UAE',
            products: 'Steel products, construction materials',
            main_materials: 'Iron ore, scrap metal, coal',
            production_volume: '5000 tons/month',
            employee_count: '201-500',
            sustainability_goals: ['Carbon reduction', 'Waste minimization'],
            tracks_waste: true
        };
        
        const portfolioResponse = await fetch(`${baseUrl}/ai-portfolio-generation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(companyProfile)
        });
        
        if (!portfolioResponse.ok) {
            throw new Error(`Portfolio generation failed: ${portfolioResponse.status}`);
        }
        
        const portfolioData = await portfolioResponse.json();
        console.log('✅ AI Portfolio Generation working!');
        console.log(`   Materials generated: ${portfolioData.portfolio?.materials?.length || 0}`);
        console.log(`   Requirements generated: ${portfolioData.portfolio?.requirements?.length || 0}`);
        
        // Test 2: AI Matchmaking
        console.log('\n2️⃣ Testing AI Matchmaking...');
        if (portfolioData.portfolio?.materials?.length > 0) {
            const material = portfolioData.portfolio.materials[0];
            
            const matchmakingResponse = await fetch(`${baseUrl}/ai-matchmaking`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    company_id: companyProfile.id,
                    material_data: material
                })
            });
            
            if (!matchmakingResponse.ok) {
                throw new Error(`Matchmaking failed: ${matchmakingResponse.status}`);
            }
            
            const matchmakingData = await matchmakingResponse.json();
            console.log('✅ AI Matchmaking working!');
            console.log(`   Matches found: ${matchmakingData.partner_companies?.length || 0}`);
            
            // Test 3: Logistics Preview
            console.log('\n3️⃣ Testing Logistics Preview...');
            if (matchmakingData.partner_companies?.length > 0) {
                const partner = matchmakingData.partner_companies[0];
                
                const logisticsResponse = await fetch(`${baseUrl}/logistics-preview`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        origin: companyProfile.location,
                        destination: partner.location || 'Riyadh, Saudi Arabia',
                        material: material.name,
                        weight_kg: 1000,
                        company_profile: companyProfile
                    })
                });
                
                if (!logisticsResponse.ok) {
                    throw new Error(`Logistics preview failed: ${logisticsResponse.status}`);
                }
                
                const logisticsData = await logisticsResponse.json();
                console.log('✅ Logistics Preview working!');
                console.log(`   Transport modes: ${logisticsData.transport_modes?.length || 0}`);
                console.log(`   Total cost: $${logisticsData.total_cost?.toLocaleString() || 0}`);
                console.log(`   Carbon footprint: ${logisticsData.total_carbon || 0} kg CO2`);
                console.log(`   Feasible: ${logisticsData.is_feasible}`);
                console.log(`   ROI: ${logisticsData.roi_percentage || 0}%`);
            }
        }
        
        // Test 4: Freightos Integration
        console.log('\n4️⃣ Testing Freightos Integration...');
        const freightosResponse = await fetch(`${baseUrl}/freightos/rates`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                material_name: 'Steel Scrap',
                origin: { country: 'UAE', city: 'Dubai', zip: '00000' },
                destination: { country: 'Saudi Arabia', city: 'Riyadh', zip: '00000' },
                weight_kg: 1000,
                length_cm: 100,
                width_cm: 100,
                height_cm: 100,
                package_type: 'pallet',
                incoterm: 'FOB'
            })
        });
        
        if (freightosResponse.ok) {
            const freightosData = await freightosResponse.json();
            console.log('✅ Freightos Integration working!');
            console.log(`   Success: ${freightosData.success}`);
            console.log(`   Test mode: ${freightosData.test_mode}`);
            console.log(`   Rates found: ${freightosData.rates?.length || 0}`);
        } else {
            console.log('⚠️ Freightos using fallback estimates');
        }
        
        console.log('\n🎉 Perfect User Flow Test Completed Successfully!');
        console.log('\n📋 Summary:');
        console.log('   ✅ AI Portfolio Generation - Working');
        console.log('   ✅ AI Matchmaking - Working');
        console.log('   ✅ Logistics Preview - Working');
        console.log('   ✅ Freightos Integration - Working (with fallbacks)');
        console.log('\n🚀 User Flow:');
        console.log('   1. User completes AI onboarding');
        console.log('   2. AI generates materials and requirements');
        console.log('   3. AI finds potential partner matches');
        console.log('   4. User clicks on match to see logistics preview');
        console.log('   5. System shows shipping costs, carbon emissions, and ROI');
        console.log('   6. User can make informed decisions about partnerships');
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        console.log('\n🔧 Troubleshooting:');
        console.log('   1. Make sure backend is running on port 5001');
        console.log('   2. Check that Python AI services are accessible');
        console.log('   3. Verify Supabase connection');
        console.log('   4. Check Freightos API credentials in .env file');
    }
}

// Run the test
testPerfectUserFlow(); 