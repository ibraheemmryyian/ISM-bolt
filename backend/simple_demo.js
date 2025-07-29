const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Mock data
const materials = [
  {
    id: 'mat_001',
    material_name: 'Wood Scraps',
    description: 'High-quality wood waste from furniture manufacturing',
    material_type: 'Organic Waste',
    quantity_available: '500 kg/day',
    company_id: 'comp_001',
    location: 'Berlin, Germany',
    sustainability_score: 85,
    created_at: new Date().toISOString()
  },
  {
    id: 'mat_002',
    material_name: 'Metal Shavings',
    description: 'Aluminum and steel shavings from automotive production',
    material_type: 'Metal Waste',
    quantity_available: '200 kg/day',
    company_id: 'comp_002',
    location: 'Lyon, France',
    sustainability_score: 92,
    created_at: new Date().toISOString()
  },
  {
    id: 'mat_003',
    material_name: 'Plastic Waste',
    description: 'Clean PET and HDPE plastic waste from packaging',
    material_type: 'Plastic Waste',
    quantity_available: '300 kg/day',
    company_id: 'comp_003',
    location: 'Amsterdam, Netherlands',
    sustainability_score: 78,
    created_at: new Date().toISOString()
  }
];

const companies = [
  {
    id: 'comp_001',
    name: 'EcoFurniture Ltd',
    industry: 'Furniture Manufacturing',
    location: 'Berlin, Germany',
    sustainability_rating: 'A+',
    needs: ['Wood Scraps', 'Paper Waste']
  },
  {
    id: 'comp_002',
    name: 'GreenMetal Recycling',
    industry: 'Metal Recycling',
    location: 'Lyon, France',
    sustainability_rating: 'A',
    needs: ['Metal Waste', 'Scrap Metal']
  },
  {
    id: 'comp_003',
    name: 'PlastiCycle BV',
    industry: 'Plastic Recycling',
    location: 'Amsterdam, Netherlands',
    sustainability_rating: 'A-',
    needs: ['Plastic Waste', 'PET Bottles']
  }
];

// Generate matches function
function generateMatches(materialId) {
  const material = materials.find(m => m.id === materialId);
  if (!material) return [];

  const matches = [];
  companies.forEach(company => {
    if (company.needs.some(need => 
      material.material_name.includes(need) || 
      material.material_type.includes(need.split(' ')[0])
    )) {
      matches.push({
        match_id: `match_${materialId}_${company.id}`,
        material_id: materialId,
        material_name: material.material_name,
        company_id: company.id,
        company_name: company.name,
        match_score: (Math.random() * 0.3 + 0.7).toFixed(3),
        compatibility_score: (Math.random() * 0.25 + 0.75).toFixed(3),
        environmental_impact: `${Math.floor(Math.random() * 100 + 10)} tons CO2 saved/year`,
        potential_revenue: `€${Math.floor(Math.random() * 50000 + 10000)}/year`,
        match_reason: `${company.name} specializes in processing ${material.material_type}`,
        logistics_cost: `€${Math.floor(Math.random() * 500 + 100)}/shipment`,
        created_at: new Date().toISOString()
      });
    }
  });

  return matches.sort((a, b) => parseFloat(b.match_score) - parseFloat(a.match_score));
}

// Routes
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    message: 'SymbioFlows Demo API is running',
    timestamp: new Date().toISOString(),
    demo_mode: true
  });
});

app.get('/api/materials', (req, res) => {
  res.json({
    success: true,
    data: materials,
    count: materials.length
  });
});

app.get('/api/companies', (req, res) => {
  res.json({
    success: true,
    data: companies,
    count: companies.length
  });
});

app.get('/api/materials/:materialId', (req, res) => {
  const material = materials.find(m => m.id === req.params.materialId);
  if (!material) {
    return res.status(404).json({
      success: false,
      error: 'Material not found'
    });
  }
  
  res.json({
    success: true,
    data: material
  });
});

app.get('/api/materials/:materialId/matches', (req, res) => {
  const matches = generateMatches(req.params.materialId);
  res.json({
    success: true,
    data: matches,
    material_id: req.params.materialId,
    count: matches.length
  });
});

app.get('/api/statistics', (req, res) => {
  let totalMatches = 0;
  let totalRevenue = 0;
  let totalCarbonSaved = 0;

  materials.forEach(material => {
    const matches = generateMatches(material.id);
    totalMatches += matches.length;
    matches.forEach(match => {
      const revenueMatch = match.potential_revenue.match(/€(\d+)/);
      if (revenueMatch) totalRevenue += parseInt(revenueMatch[1]);
      
      const carbonMatch = match.environmental_impact.match(/(\d+) tons/);
      if (carbonMatch) totalCarbonSaved += parseInt(carbonMatch[1]);
    });
  });

  res.json({
    success: true,
    data: {
      total_materials: materials.length,
      total_companies: companies.length,
      total_matches: totalMatches,
      potential_revenue: `€${totalRevenue.toLocaleString()}/year`,
      carbon_impact: `${totalCarbonSaved} tons CO2 saved/year`,
      average_match_score: 0.85,
      active_partnerships: Math.floor(totalMatches * 0.3),
      sustainability_improvement: '23%'
    }
  });
});

app.post('/api/materials', (req, res) => {
  const newMaterial = {
    id: `mat_${Date.now()}`,
    ...req.body,
    created_at: new Date().toISOString()
  };
  
  materials.push(newMaterial);
  
  res.status(201).json({
    success: true,
    data: newMaterial,
    message: 'Material created successfully'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    available_endpoints: [
      'GET /api/health',
      'GET /api/materials',
      'GET /api/companies',
      'GET /api/materials/:id',
      'GET /api/materials/:id/matches',
      'GET /api/statistics',
      'POST /api/materials'
    ]
  });
});

// Start server
app.listen(PORT, () => {
  console.log('🚀 SymbioFlows Simple Demo Backend Started!');
  console.log(`📡 API running on: http://localhost:${PORT}`);
  console.log(`🏥 Health check: http://localhost:${PORT}/api/health`);
  console.log(`📊 Materials: http://localhost:${PORT}/api/materials`);
  console.log(`🏢 Companies: http://localhost:${PORT}/api/companies`);
  console.log(`📈 Statistics: http://localhost:${PORT}/api/statistics`);
  console.log('\n✅ Simple Backend is ready for demo!');
  console.log('\n🎯 Demo endpoints:');
  console.log('   - Materials with matches: GET /api/materials/mat_001/matches');
  console.log('   - Real-time statistics: GET /api/statistics');
  console.log('   - Add new material: POST /api/materials');
});

module.exports = app;