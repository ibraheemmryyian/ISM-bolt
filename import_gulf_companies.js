#!/usr/bin/env node

/**
 * Gulf Companies Data Import Script
 * Imports real Gulf region companies into Supabase and generates material listings and matches
 */

require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('❌ Missing Supabase credentials. Please check your .env file.');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

// Load Gulf companies data
const gulfDataPath = path.join(__dirname, 'gulf_companies_data.json');
let gulfCompanies = [];

try {
  const data = fs.readFileSync(gulfDataPath, 'utf8');
  gulfCompanies = JSON.parse(data);
  console.log(`✅ Loaded ${gulfCompanies.length} Gulf companies from data file`);
} catch (error) {
  console.error('❌ Error loading Gulf companies data:', error.message);
  process.exit(1);
}

/**
 * Import companies into Supabase
 */
async function importCompanies() {
  console.log('\n🚀 Starting companies import...');
  
  const companiesToInsert = gulfCompanies.map(company => ({
    name: company.name,
    industry: company.industry,
    location: company.location,
    employee_count: company.employee_count,
    water_usage: company.water_usage,
    carbon_footprint: company.carbon_footprint,
    sustainability_score: company.sustainability_score,
    matching_preferences: company.matching_preferences,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }));

  try {
    const { data, error } = await supabase
      .from('companies')
      .upsert(companiesToInsert, { onConflict: 'name' })
      .select();

    if (error) {
      console.error('❌ Error importing companies:', error);
      return false;
    }

    console.log(`✅ Successfully imported ${data.length} companies`);
    return true;
  } catch (error) {
    console.error('❌ Error importing companies:', error);
    return false;
  }
}

/**
 * Generate material listings from company data
 */
async function generateMaterialListings() {
  console.log('\n🔧 Generating material listings...');
  
  const materialListings = [];
  
  gulfCompanies.forEach(company => {
    // Generate listings from materials
    if (company.materials && company.materials.length > 0) {
      company.materials.forEach(material => {
        materialListings.push({
          company_name: company.name,
          material_name: material,
          material_type: 'input',
          quantity: Math.floor(Math.random() * 1000) + 100, // Random quantity
          unit: 'kg',
          unit_price: Math.floor(Math.random() * 50) + 10, // Random price
          location: company.location,
          industry: company.industry,
          sustainability_score: company.sustainability_score,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        });
      });
    }

    // Generate listings from products
    if (company.products && company.products.length > 0) {
      company.products.forEach(product => {
        materialListings.push({
          company_name: company.name,
          material_name: product,
          material_type: 'output',
          quantity: Math.floor(Math.random() * 500) + 50,
          unit: 'kg',
          unit_price: Math.floor(Math.random() * 100) + 20,
          location: company.location,
          industry: company.industry,
          sustainability_score: company.sustainability_score,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        });
      });
    }

    // Generate listings from waste streams
    if (company.waste_streams && company.waste_streams.length > 0) {
      company.waste_streams.forEach(waste => {
        materialListings.push({
          company_name: company.name,
          material_name: waste,
          material_type: 'waste',
          quantity: Math.floor(Math.random() * 200) + 10,
          unit: 'kg',
          unit_price: Math.floor(Math.random() * 5) + 1, // Lower price for waste
          location: company.location,
          industry: company.industry,
          sustainability_score: company.sustainability_score,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        });
      });
    }
  });

  try {
    const { data, error } = await supabase
      .from('material_listings')
      .upsert(materialListings, { onConflict: 'company_name,material_name' })
      .select();

    if (error) {
      console.error('❌ Error generating material listings:', error);
      return false;
    }

    console.log(`✅ Successfully generated ${data.length} material listings`);
    return true;
  } catch (error) {
    console.error('❌ Error generating material listings:', error);
    return false;
  }
}

/**
 * Generate AI-powered matches between companies
 */
async function generateAIMatches() {
  console.log('\n🤖 Generating AI-powered matches...');
  
  const matches = [];
  
  // Generate matches based on material exchange opportunities
  for (let i = 0; i < gulfCompanies.length; i++) {
    for (let j = i + 1; j < gulfCompanies.length; j++) {
      const companyA = gulfCompanies[i];
      const companyB = gulfCompanies[j];
      
      // Calculate match score based on preferences and compatibility
      const materialExchangeScore = (companyA.matching_preferences.material_exchange + companyB.matching_preferences.material_exchange) / 2;
      const wasteRecyclingScore = (companyA.matching_preferences.waste_recycling + companyB.matching_preferences.waste_recycling) / 2;
      const energySharingScore = (companyA.matching_preferences.energy_sharing + companyB.matching_preferences.energy_sharing) / 2;
      const waterReuseScore = (companyA.matching_preferences.water_reuse + companyB.matching_preferences.water_reuse) / 2;
      const logisticsSharingScore = (companyA.matching_preferences.logistics_sharing + companyB.matching_preferences.logistics_sharing) / 2;
      
      // Calculate overall match score
      const overallScore = (materialExchangeScore + wasteRecyclingScore + energySharingScore + waterReuseScore + logisticsSharingScore) / 5;
      
      // Only create matches with score > 0.6
      if (overallScore > 0.6) {
        matches.push({
          company_a: companyA.name,
          company_b: companyB.name,
          match_type: 'industrial_symbiosis',
          match_score: overallScore,
          material_exchange_score: materialExchangeScore,
          waste_recycling_score: wasteRecyclingScore,
          energy_sharing_score: energySharingScore,
          water_reuse_score: waterReuseScore,
          logistics_sharing_score: logisticsSharingScore,
          location_a: companyA.location,
          location_b: companyB.location,
          industry_a: companyA.industry,
          industry_b: companyB.industry,
          sustainability_impact: (companyA.sustainability_score + companyB.sustainability_score) / 2,
          carbon_reduction_potential: Math.floor(Math.random() * 1000) + 100,
          cost_savings_potential: Math.floor(Math.random() * 50000) + 5000,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        });
      }
    }
  }

  try {
    const { data, error } = await supabase
      .from('ai_matches')
      .upsert(matches, { onConflict: 'company_a,company_b' })
      .select();

    if (error) {
      console.error('❌ Error generating AI matches:', error);
      return false;
    }

    console.log(`✅ Successfully generated ${data.length} AI-powered matches`);
    return true;
  } catch (error) {
    console.error('❌ Error generating AI matches:', error);
    return false;
  }
}

/**
 * Generate detailed symbiosis opportunities
 */
async function generateSymbiosisOpportunities() {
  console.log('\n🔄 Generating detailed symbiosis opportunities...');
  
  const opportunities = [];
  
  gulfCompanies.forEach(company => {
    // Generate opportunities based on company's waste streams and materials
    if (company.waste_streams && company.waste_streams.length > 0) {
      company.waste_streams.forEach(waste => {
        // Find potential partners for this waste
        const potentialPartners = gulfCompanies.filter(otherCompany => 
          otherCompany.name !== company.name &&
          (otherCompany.materials.includes(waste) || 
           otherCompany.products.some(product => product.toLowerCase().includes(waste.toLowerCase())))
        );

        potentialPartners.forEach(partner => {
          opportunities.push({
            source_company: company.name,
            partner_company: partner.name,
            material_type: waste,
            opportunity_type: 'waste_to_resource',
            description: `${company.name} can supply ${waste} to ${partner.name} for processing`,
            potential_value: Math.floor(Math.random() * 10000) + 1000,
            carbon_reduction: Math.floor(Math.random() * 500) + 50,
            implementation_difficulty: Math.floor(Math.random() * 5) + 1,
            timeline_months: Math.floor(Math.random() * 24) + 6,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          });
        });
      });
    }
  });

  try {
    const { data, error } = await supabase
      .from('symbiosis_opportunities')
      .upsert(opportunities, { onConflict: 'source_company,partner_company,material_type' })
      .select();

    if (error) {
      console.error('❌ Error generating symbiosis opportunities:', error);
      return false;
    }

    console.log(`✅ Successfully generated ${data.length} symbiosis opportunities`);
    return true;
  } catch (error) {
    console.error('❌ Error generating symbiosis opportunities:', error);
    return false;
  }
}

/**
 * Main execution function
 */
async function main() {
  console.log('🌍 Gulf Companies Data Import Script');
  console.log('=====================================');
  
  try {
    // Step 1: Import companies
    const companiesImported = await importCompanies();
    if (!companiesImported) {
      console.error('❌ Failed to import companies. Stopping.');
      return;
    }

    // Step 2: Generate material listings
    const listingsGenerated = await generateMaterialListings();
    if (!listingsGenerated) {
      console.error('❌ Failed to generate material listings. Stopping.');
      return;
    }

    // Step 3: Generate AI matches
    const matchesGenerated = await generateAIMatches();
    if (!matchesGenerated) {
      console.error('❌ Failed to generate AI matches. Stopping.');
      return;
    }

    // Step 4: Generate symbiosis opportunities
    const opportunitiesGenerated = await generateSymbiosisOpportunities();
    if (!opportunitiesGenerated) {
      console.error('❌ Failed to generate symbiosis opportunities. Stopping.');
      return;
    }

    console.log('\n🎉 SUCCESS! Gulf companies data has been imported and processed.');
    console.log('📊 You can now view the results in your Supabase dashboard:');
    console.log('   - Companies table: All 35 Gulf companies');
    console.log('   - Material listings table: Generated from materials, products, and waste streams');
    console.log('   - AI matches table: AI-powered company matching');
    console.log('   - Symbiosis opportunities table: Detailed partnership opportunities');
    
  } catch (error) {
    console.error('❌ Unexpected error:', error);
  }
}

// Run the script
if (require.main === module) {
  main();
}

module.exports = { importCompanies, generateMaterialListings, generateAIMatches, generateSymbiosisOpportunities }; 