// SymbioFlows Demo Data Generator
// Generates impressive demo data for the millionaire meeting

const axios = require('axios');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:5001';

// Real Gulf company data for demo
const gulfCompanies = [
  {
    name: 'Saudi Aramco',
    industry: 'oil_gas',
    location: 'Dhahran, Saudi Arabia',
    employee_count: '10000+',
    description: 'World\'s largest oil company, leading energy producer',
    sustainability_score: 85,
    materials: ['Crude oil byproducts', 'Natural gas', 'Petrochemical waste', 'Heat recovery']
  },
  {
    name: 'Emirates Global Aluminium',
    industry: 'aluminum_manufacturing',
    location: 'Dubai, UAE',
    employee_count: '5001-10000',
    description: 'Leading aluminum producer in the Middle East',
    sustainability_score: 88,
    materials: ['Aluminum scrap', 'Red mud waste', 'Heat recovery', 'Bauxite residue']
  },
  {
    name: 'Qatar Steel',
    industry: 'steel_manufacturing',
    location: 'Doha, Qatar',
    employee_count: '1001-5000',
    description: 'Major steel manufacturer in Qatar',
    sustainability_score: 82,
    materials: ['Steel scrap', 'Slag waste', 'Heat recovery', 'Iron ore dust']
  },
  {
    name: 'Kuwait Petroleum Corporation',
    industry: 'oil_gas',
    location: 'Kuwait City, Kuwait',
    employee_count: '5001-10000',
    description: 'State-owned oil company of Kuwait',
    sustainability_score: 79,
    materials: ['Petroleum byproducts', 'Natural gas', 'Refinery waste', 'Heat recovery']
  },
  {
    name: 'Oman Oil Refineries',
    industry: 'oil_gas',
    location: 'Muscat, Oman',
    employee_count: '1001-5000',
    description: 'Leading oil refining company in Oman',
    sustainability_score: 81,
    materials: ['Refinery byproducts', 'Petroleum waste', 'Heat recovery', 'Chemical waste']
  },
  {
    name: 'Bahrain Steel',
    industry: 'steel_manufacturing',
    location: 'Manama, Bahrain',
    employee_count: '501-1000',
    description: 'Steel manufacturing and processing',
    sustainability_score: 84,
    materials: ['Steel scrap', 'Slag waste', 'Heat recovery', 'Iron dust']
  },
  {
    name: 'Abu Dhabi Polymers',
    industry: 'chemical_manufacturing',
    location: 'Abu Dhabi, UAE',
    employee_count: '1001-5000',
    description: 'Advanced polymer and chemical manufacturing',
    sustainability_score: 87,
    materials: ['Polymer waste', 'Chemical byproducts', 'Heat recovery', 'Plastic waste']
  },
  {
    name: 'Riyadh Cement Company',
    industry: 'cement_manufacturing',
    location: 'Riyadh, Saudi Arabia',
    employee_count: '501-1000',
    description: 'Leading cement manufacturer in Saudi Arabia',
    sustainability_score: 83,
    materials: ['Cement waste', 'Limestone dust', 'Heat recovery', 'Fly ash']
  },
  {
    name: 'Dubai Aluminium',
    industry: 'aluminum_manufacturing',
    location: 'Dubai, UAE',
    employee_count: '1001-5000',
    description: 'Major aluminum smelting and processing',
    sustainability_score: 86,
    materials: ['Aluminum scrap', 'Red mud', 'Heat recovery', 'Bauxite waste']
  },
  {
    name: 'Kuwait Cement Company',
    industry: 'cement_manufacturing',
    location: 'Kuwait City, Kuwait',
    employee_count: '501-1000',
    description: 'Cement and building materials manufacturer',
    sustainability_score: 80,
    materials: ['Cement waste', 'Limestone dust', 'Heat recovery', 'Gypsum waste']
  }
];

// Generate AI materials for each company
function generateAIMaterials(company) {
  const materials = [];
  
  company.materials.forEach((material, index) => {
    const baseValue = Math.floor(Math.random() * 100000) + 25000;
    const quantity = Math.floor(Math.random() * 100) + 10;
    const frequency = ['Monthly', 'Weekly', 'Daily'][Math.floor(Math.random() * 3)];
    const qualityGrade = ['high', 'medium', 'low'][Math.floor(Math.random() * 3)];
    const sustainabilityScore = Math.floor(Math.random() * 20) + 75;
    
    materials.push({
      id: `${company.name.toLowerCase().replace(/\s+/g, '_')}_${index}`,
      name: material,
      category: getCategoryFromMaterial(material),
      description: `AI-identified ${material.toLowerCase()} from ${company.name} production processes`,
      quantity: `${quantity} tons/${frequency.toLowerCase()}`,
      frequency: frequency,
      potential_value: `$${baseValue.toLocaleString()}`,
      quality_grade: qualityGrade,
      potential_uses: generatePotentialUses(material),
      symbiosis_opportunities: generateSymbiosisOpportunities(material),
      ai_generated: true,
      sustainability_score: sustainabilityScore,
      company_id: company.name.toLowerCase().replace(/\s+/g, '_')
    });
  });
  
  return materials;
}

function getCategoryFromMaterial(material) {
  if (material.includes('oil') || material.includes('gas') || material.includes('petroleum')) {
    return 'Oil & Gas Waste';
  } else if (material.includes('steel') || material.includes('iron')) {
    return 'Metal Waste';
  } else if (material.includes('aluminum') || material.includes('bauxite')) {
    return 'Aluminum Waste';
  } else if (material.includes('cement') || material.includes('limestone')) {
    return 'Construction Waste';
  } else if (material.includes('chemical') || material.includes('polymer')) {
    return 'Chemical Waste';
  } else if (material.includes('heat')) {
    return 'Energy Recovery';
  } else {
    return 'Industrial Waste';
  }
}

function generatePotentialUses(material) {
  const uses = {
    'Crude oil byproducts': ['Fuel production', 'Lubricant manufacturing', 'Chemical feedstock'],
    'Steel scrap': ['Steel manufacturing', 'Construction materials', 'Automotive parts'],
    'Aluminum scrap': ['Aluminum production', 'Packaging materials', 'Automotive parts'],
    'Cement waste': ['Road construction', 'Building materials', 'Soil stabilization'],
    'Chemical byproducts': ['Water treatment', 'Fertilizer production', 'Textile processing'],
    'Heat recovery': ['District heating', 'Greenhouse operations', 'Industrial processes']
  };
  
  return uses[material] || ['Recycling', 'Reprocessing', 'Alternative uses'];
}

function generateSymbiosisOpportunities(material) {
  const opportunities = {
    'Crude oil byproducts': ['Refineries', 'Chemical plants', 'Fuel producers'],
    'Steel scrap': ['Steel mills', 'Construction companies', 'Automotive manufacturers'],
    'Aluminum scrap': ['Aluminum smelters', 'Packaging companies', 'Automotive manufacturers'],
    'Cement waste': ['Construction companies', 'Road builders', 'Building material producers'],
    'Chemical byproducts': ['Water treatment plants', 'Agricultural companies', 'Textile manufacturers'],
    'Heat recovery': ['District heating networks', 'Greenhouse farms', 'Nearby industries']
  };
  
  return opportunities[material] || ['Local industries', 'Recycling facilities', 'Processing plants'];
}

// Generate AI matches
function generateAIMatches(materials) {
  const matches = [];
  const partnerCompanies = [
    'Arabian Steel Solutions', 'Gulf Chemical Industries', 'Dubai District Heating',
    'Saudi Recycling Corp', 'Emirates Waste Management', 'Qatar Industrial Partners',
    'Kuwait Green Solutions', 'Oman Circular Economy', 'Bahrain Sustainable Industries',
    'Abu Dhabi Industrial Exchange'
  ];
  
  materials.forEach((material, index) => {
    const partner = partnerCompanies[index % partnerCompanies.length];
    const matchScore = Math.floor(Math.random() * 20) + 80; // 80-100%
    const potentialSavings = Math.floor(Math.random() * 50000) + 25000;
    const carbonReduction = Math.floor(Math.random() * 50) + 20;
    const implementationTime = `${Math.floor(Math.random() * 4) + 2}-${Math.floor(Math.random() * 4) + 4} months`;
    const riskLevel = ['low', 'medium', 'high'][Math.floor(Math.random() * 3)];
    const partnershipType = ['Waste Exchange', 'Byproduct Exchange', 'Energy Exchange', 'Resource Sharing'][Math.floor(Math.random() * 4)];
    
    matches.push({
      id: `match_${index}`,
      company_name: partner,
      company_type: getCompanyTypeFromPartner(partner),
      location: getLocationFromPartner(partner),
      match_score: matchScore,
      material_match: material.name,
      potential_savings: `$${potentialSavings.toLocaleString()}`,
      carbon_reduction: `${carbonReduction} tons CO2`,
      implementation_time: implementationTime,
      contact_info: `partnerships@${partner.toLowerCase().replace(/\s+/g, '')}.com`,
      ai_generated: true,
      sustainability_impact: getSustainabilityImpact(partnershipType),
      risk_level: riskLevel,
      partnership_type: partnershipType,
      material_id: material.id
    });
  });
  
  return matches;
}

function getCompanyTypeFromPartner(partner) {
  if (partner.includes('Steel')) return 'Steel Manufacturing';
  if (partner.includes('Chemical')) return 'Chemical Processing';
  if (partner.includes('Heating')) return 'Energy Services';
  if (partner.includes('Recycling')) return 'Waste Management';
  if (partner.includes('Waste')) return 'Waste Management';
  if (partner.includes('Industrial')) return 'Industrial Services';
  if (partner.includes('Circular')) return 'Circular Economy';
  if (partner.includes('Sustainable')) return 'Sustainable Industries';
  if (partner.includes('Green')) return 'Environmental Services';
  return 'Industrial Services';
}

function getLocationFromPartner(partner) {
  const locations = {
    'Arabian': 'Riyadh, Saudi Arabia',
    'Gulf': 'Jeddah, Saudi Arabia',
    'Dubai': 'Dubai, UAE',
    'Saudi': 'Riyadh, Saudi Arabia',
    'Emirates': 'Abu Dhabi, UAE',
    'Qatar': 'Doha, Qatar',
    'Kuwait': 'Kuwait City, Kuwait',
    'Oman': 'Muscat, Oman',
    'Bahrain': 'Manama, Bahrain'
  };
  
  for (const [key, location] of Object.entries(locations)) {
    if (partner.includes(key)) return location;
  }
  
  return 'Dubai, UAE';
}

function getSustainabilityImpact(partnershipType) {
  const impacts = {
    'Waste Exchange': 'High - Circular economy model',
    'Byproduct Exchange': 'Medium - Resource optimization',
    'Energy Exchange': 'Very High - Renewable energy',
    'Resource Sharing': 'High - Collaborative consumption'
  };
  
  return impacts[partnershipType] || 'Medium - Resource optimization';
}

// Generate logistics preview data
function generateLogisticsPreview(match) {
  const transportModes = [
    {
      mode: 'Truck',
      cost: Math.floor(Math.random() * 5000) + 5000,
      transit_time: Math.floor(Math.random() * 3) + 1,
      carbon_emissions: Math.floor(Math.random() * 300) + 500,
      reliability: 0.95,
      sustainability_score: 65
    },
    {
      mode: 'Sea',
      cost: Math.floor(Math.random() * 3000) + 4000,
      transit_time: Math.floor(Math.random() * 5) + 3,
      carbon_emissions: Math.floor(Math.random() * 200) + 300,
      reliability: 0.90,
      sustainability_score: 85
    },
    {
      mode: 'Rail',
      cost: Math.floor(Math.random() * 2000) + 5000,
      transit_time: Math.floor(Math.random() * 4) + 2,
      carbon_emissions: Math.floor(Math.random() * 150) + 200,
      reliability: 0.92,
      sustainability_score: 90
    }
  ];
  
  const totalCost = transportModes[0].cost; // Use truck as default
  const totalCarbon = transportModes[0].carbon_emissions;
  
  return {
    origin: 'Dubai, UAE',
    destination: match.location,
    material: match.material_match,
    weight_kg: Math.floor(Math.random() * 50000) + 25000,
    transport_modes: transportModes,
    total_cost: totalCost,
    total_carbon: totalCarbon,
    cost_breakdown: {
      transport: Math.floor(totalCost * 0.8),
      handling: Math.floor(totalCost * 0.12),
      customs: Math.floor(totalCost * 0.06),
      insurance: Math.floor(totalCost * 0.02)
    },
    recommendations: [
      'Truck transport offers best cost-benefit ratio for this distance',
      'Consider bulk shipping for quantities over 100 tons',
      'Negotiate long-term contracts for better rates',
      'Explore carbon offset options to improve sustainability score'
    ],
    is_feasible: true,
    roi_percentage: Math.floor(Math.random() * 100) + 100,
    payback_period: `${Math.floor(Math.random() * 6) + 1}.${Math.floor(Math.random() * 9) + 1} months`
  };
}

// Main function to generate all demo data
async function generateDemoData() {
  console.log('🚀 Generating SymbioFlows Demo Data...\n');
  
  try {
    // Generate materials for all companies
    const allMaterials = [];
    gulfCompanies.forEach(company => {
      const materials = generateAIMaterials(company);
      allMaterials.push(...materials);
    });
    
    console.log(`✅ Generated ${allMaterials.length} AI materials for ${gulfCompanies.length} companies`);
    
    // Generate AI matches
    const aiMatches = generateAIMatches(allMaterials);
    console.log(`✅ Generated ${aiMatches.length} AI matches`);
    
    // Generate logistics previews
    const logisticsPreviews = aiMatches.map(match => generateLogisticsPreview(match));
    console.log(`✅ Generated ${logisticsPreviews.length} logistics previews`);
    
    // Calculate demo metrics
    const totalSavings = aiMatches.reduce((sum, match) => {
      return sum + parseInt(match.potential_savings.replace(/[$,]/g, ''));
    }, 0);
    
    const totalCarbonReduction = aiMatches.reduce((sum, match) => {
      return sum + parseInt(match.carbon_reduction.split(' ')[0]);
    }, 0);
    
    const averageMatchScore = aiMatches.reduce((sum, match) => sum + match.match_score, 0) / aiMatches.length;
    
    // Create demo data object
    const demoData = {
      companies: gulfCompanies,
      materials: allMaterials,
      matches: aiMatches,
      logistics: logisticsPreviews,
      metrics: {
        totalCompanies: gulfCompanies.length,
        totalMaterials: allMaterials.length,
        totalMatches: aiMatches.length,
        totalSavings: totalSavings,
        totalCarbonReduction: totalCarbonReduction,
        averageMatchScore: averageMatchScore.toFixed(1),
        sustainabilityScore: 87,
        roiPercentage: 156
      },
      generated_at: new Date().toISOString()
    };
    
    // Save demo data to file
    const demoDataPath = path.join(__dirname, '..', 'data', 'demo-data.json');
    fs.writeFileSync(demoDataPath, JSON.stringify(demoData, null, 2));
    
    console.log('\n📊 Demo Data Summary:');
    console.log('=====================');
    console.log(`🏢 Companies: ${demoData.metrics.totalCompanies}`);
    console.log(`📦 Materials: ${demoData.metrics.totalMaterials}`);
    console.log(`🤝 Matches: ${demoData.metrics.totalMatches}`);
    console.log(`💰 Total Savings: $${totalSavings.toLocaleString()}`);
    console.log(`🌱 Carbon Reduction: ${totalCarbonReduction} tons CO2`);
    console.log(`🎯 Average Match Score: ${demoData.metrics.averageMatchScore}%`);
    console.log(`📈 ROI: ${demoData.metrics.roiPercentage}%`);
    
    console.log('\n✅ Demo data generated successfully!');
    console.log(`📁 Saved to: ${demoDataPath}`);
    
    return demoData;
    
  } catch (error) {
    console.error('❌ Error generating demo data:', error);
    throw error;
  }
}

// Export for use in other scripts
module.exports = {
  generateDemoData,
  gulfCompanies,
  generateAIMaterials,
  generateAIMatches,
  generateLogisticsPreview
};

// Run if executed directly
if (require.main === module) {
  generateDemoData().then(() => {
    console.log('\n🎉 Demo data generation complete!');
  }).catch(error => {
    console.error('Failed to generate demo data:', error);
    process.exit(1);
  });
} 