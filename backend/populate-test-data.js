const { createClient } = require('@supabase/supabase-js');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

async function populateTestData() {
  console.log('Populating database with test data...');

  try {
    // Create test companies with proper UUIDs and existing schema
    const companies = [
      {
        id: uuidv4(),
        name: 'EcoSteel Manufacturing',
        email: 'contact@ecosteel.com',
        role: 'user'
      },
      {
        id: uuidv4(),
        name: 'GreenChem Solutions',
        email: 'info@greenchem.com',
        role: 'user'
      },
      {
        id: uuidv4(),
        name: 'BioEnergy Corp',
        email: 'hello@bioenergy.com',
        role: 'user'
      },
      {
        id: uuidv4(),
        name: 'RecycleTech Industries',
        email: 'contact@recycletech.com',
        role: 'user'
      },
      {
        id: uuidv4(),
        name: 'Sustainable Materials Ltd',
        email: 'info@sustainablematerials.com',
        role: 'user'
      }
    ];

    // Insert companies
    for (const company of companies) {
      const { error } = await supabase
        .from('companies')
        .upsert(company, { onConflict: 'id' });
      
      if (error) {
        console.error(`Error inserting company ${company.name}:`, error);
      } else {
        console.log(`✓ Inserted company: ${company.name}`);
      }
    }

    // Create test materials
    const materials = [
      // Waste materials
      {
        material_name: 'Steel Scrap',
        quantity: 500,
        unit: 'tons',
        description: 'High-quality steel scrap from manufacturing processes',
        type: 'waste',
        company_id: companies[0].id
      },
      {
        material_name: 'Chemical Waste',
        quantity: 200,
        unit: 'tons',
        description: 'Non-hazardous chemical byproducts from production',
        type: 'waste',
        company_id: companies[1].id
      },
      {
        material_name: 'Biomass Waste',
        quantity: 1000,
        unit: 'tons',
        description: 'Organic waste from biofuel production',
        type: 'waste',
        company_id: companies[2].id
      },
      {
        material_name: 'Plastic Waste',
        quantity: 300,
        unit: 'tons',
        description: 'Recyclable plastic materials from various sources',
        type: 'waste',
        company_id: companies[3].id
      },
      {
        material_name: 'Construction Debris',
        quantity: 800,
        unit: 'tons',
        description: 'Mixed construction and demolition waste',
        type: 'waste',
        company_id: companies[4].id
      },
      
      // Requirement materials
      {
        material_name: 'Steel Scrap',
        quantity: 400,
        unit: 'tons',
        description: 'Need steel scrap for recycling and new production',
        type: 'requirement',
        company_id: companies[1].id
      },
      {
        material_name: 'Chemical Waste',
        quantity: 150,
        unit: 'tons',
        description: 'Looking for chemical waste for processing',
        type: 'requirement',
        company_id: companies[2].id
      },
      {
        material_name: 'Biomass Waste',
        quantity: 800,
        unit: 'tons',
        description: 'Need biomass for energy production',
        type: 'requirement',
        company_id: companies[3].id
      },
      {
        material_name: 'Plastic Waste',
        quantity: 250,
        unit: 'tons',
        description: 'Seeking plastic waste for recycling operations',
        type: 'requirement',
        company_id: companies[4].id
      },
      {
        material_name: 'Construction Debris',
        quantity: 600,
        unit: 'tons',
        description: 'Need construction waste for processing',
        type: 'requirement',
        company_id: companies[0].id
      }
    ];

    // Insert materials
    for (const material of materials) {
      const { error } = await supabase
        .from('materials')
        .insert(material);
      
      if (error) {
        console.error(`Error inserting material ${material.material_name}:`, error);
      } else {
        console.log(`✓ Inserted material: ${material.material_name}`);
      }
    }

    // Create some AI recommendations
    const recommendations = [
      {
        id: 'rec-1',
        company_id: companies[0].id,
        type: 'connection',
        title: 'Connect with GreenChem Solutions',
        description: 'High compatibility for steel scrap exchange',
        confidence: 87,
        status: 'pending'
      },
      {
        id: 'rec-2',
        company_id: companies[1].id,
        type: 'material',
        title: 'Expand Chemical Waste Listings',
        description: 'Add more chemical waste types to increase matches',
        confidence: 92,
        status: 'pending'
      },
      {
        id: 'rec-3',
        company_id: companies[2].id,
        type: 'opportunity',
        title: 'Sustainability Opportunity',
        description: 'Your biomass waste could save €25,000 annually',
        confidence: 78,
        status: 'pending'
      }
    ];

    for (const rec of recommendations) {
      const { error } = await supabase
        .from('ai_recommendations')
        .upsert(rec, { onConflict: 'id' });
      
      if (error) {
        console.error(`Error inserting recommendation:`, error);
      } else {
        console.log(`✓ Inserted AI recommendation: ${rec.title}`);
      }
    }

    console.log('✓ Test data population completed successfully!');
    console.log('You can now test the AI matching and global impact features.');

  } catch (error) {
    console.error('Error populating test data:', error);
  }
}

// Run the population script
populateTestData(); 