// Load environment variables from .env file
require('dotenv').config();

const express = require('express');
const cors = require('cors');
const { supabase } = require('./supabase');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors({
  origin: [
    'http://localhost:5173',
    'https://symbioflows.com',
    'https://www.symbioflows.com',
    process.env.FRONTEND_URL
  ].filter(Boolean),
  credentials: true
}));

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Helper function to send consistent responses
function sendResponse(res, data, statusCode = 200) {
  res.status(statusCode).json(data);
}

// Mock data for demo
const mockMaterials = [
  {
    id: 'mat_001',
    material_name: 'Wood Scraps',
    description: 'High-quality wood waste from furniture manufacturing',
    material_type: 'Organic Waste',
    quantity_available: '500 kg/day',
    company_id: 'comp_001',
    location: 'Germany',
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
    location: 'France',
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
    location: 'Netherlands',
    sustainability_score: 78,
    created_at: new Date().toISOString()
  }
];

const mockCompanies = [
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

// Mock function to generate matches
function generateMockMatches(materialId) {
  const material = mockMaterials.find(m => m.id === materialId);
  if (!material) return [];

  const matches = [];
  mockCompanies.forEach(company => {
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
        match_score: Math.random() * 0.3 + 0.7, // Score between 0.7-1.0
        compatibility_score: Math.random() * 0.25 + 0.75,
        environmental_impact: `${Math.floor(Math.random() * 100 + 10)} tons CO2 saved/year`,
        potential_revenue: `€${Math.floor(Math.random() * 50000 + 10000)}/year`,
        match_reason: `${company.name} specializes in processing ${material.material_type}`,
        logistics_cost: `€${Math.floor(Math.random() * 500 + 100)}/shipment`,
        created_at: new Date().toISOString()
      });
    }
  });

  return matches.sort((a, b) => b.match_score - a.match_score);
}

// ===== API ENDPOINTS =====

// Health check
app.get('/api/health', (req, res) => {
  sendResponse(res, {
    status: 'healthy',
    message: 'SymbioFlows Demo API is running',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
    demo_mode: true
  });
});

// Get all materials
app.get('/api/materials', async (req, res) => {
  try {
    // Try to get from database first, fallback to mock data
    const { data: dbMaterials, error } = await supabase
      .from('materials')
      .select('*')
      .limit(50);

    if (error) {
      console.warn('Database error, using mock data:', error.message);
      return sendResponse(res, {
        success: true,
        data: mockMaterials,
        count: mockMaterials.length,
        source: 'mock'
      });
    }

    const materials = dbMaterials.length > 0 ? dbMaterials : mockMaterials;
    return sendResponse(res, {
      success: true,
      data: materials,
      count: materials.length,
      source: dbMaterials.length > 0 ? 'database' : 'mock'
    });
  } catch (error) {
    console.error('Materials endpoint error:', error);
    return sendResponse(res, {
      success: true,
      data: mockMaterials,
      count: mockMaterials.length,
      source: 'mock'
    });
  }
});

// Get all companies
app.get('/api/companies', async (req, res) => {
  try {
    // Try to get from database first, fallback to mock data
    const { data: dbCompanies, error } = await supabase
      .from('companies')
      .select('*')
      .limit(50);

    if (error) {
      console.warn('Database error, using mock data:', error.message);
      return sendResponse(res, {
        success: true,
        data: mockCompanies,
        count: mockCompanies.length,
        source: 'mock'
      });
    }

    const companies = dbCompanies.length > 0 ? dbCompanies : mockCompanies;
    return sendResponse(res, {
      success: true,
      data: companies,
      count: companies.length,
      source: dbCompanies.length > 0 ? 'database' : 'mock'
    });
  } catch (error) {
    console.error('Companies endpoint error:', error);
    return sendResponse(res, {
      success: true,
      data: mockCompanies,
      count: mockCompanies.length,
      source: 'mock'
    });
  }
});

// Get matches for a material
app.get('/api/materials/:materialId/matches', async (req, res) => {
  try {
    const { materialId } = req.params;
    
    // Try to get from database first
    const { data: dbMatches, error } = await supabase
      .from('matches')
      .select('*')
      .eq('material_id', materialId)
      .order('match_score', { ascending: false });

    if (error || !dbMatches || dbMatches.length === 0) {
      console.warn('Database error or no matches, generating mock matches');
      const mockMatches = generateMockMatches(materialId);
      return sendResponse(res, {
        success: true,
        data: mockMatches,
        material_id: materialId,
        count: mockMatches.length,
        source: 'mock'
      });
    }

    return sendResponse(res, {
      success: true,
      data: dbMatches,
      material_id: materialId,
      count: dbMatches.length,
      source: 'database'
    });
  } catch (error) {
    console.error('Matches endpoint error:', error);
    const mockMatches = generateMockMatches(req.params.materialId);
    return sendResponse(res, {
      success: true,
      data: mockMatches,
      material_id: req.params.materialId,
      count: mockMatches.length,
      source: 'mock'
    });
  }
});

// Get material by ID
app.get('/api/materials/:materialId', async (req, res) => {
  try {
    const { materialId } = req.params;
    
    // Try database first
    const { data: dbMaterial, error } = await supabase
      .from('materials')
      .select('*')
      .eq('id', materialId)
      .single();

    if (error) {
      const mockMaterial = mockMaterials.find(m => m.id === materialId);
      if (!mockMaterial) {
        return sendResponse(res, {
          success: false,
          error: 'Material not found'
        }, 404);
      }
      return sendResponse(res, {
        success: true,
        data: mockMaterial,
        source: 'mock'
      });
    }

    return sendResponse(res, {
      success: true,
      data: dbMaterial,
      source: 'database'
    });
  } catch (error) {
    console.error('Material by ID error:', error);
    const mockMaterial = mockMaterials.find(m => m.id === req.params.materialId);
    if (!mockMaterial) {
      return sendResponse(res, {
        success: false,
        error: 'Material not found'
      }, 404);
    }
    return sendResponse(res, {
      success: true,
      data: mockMaterial,
      source: 'mock'
    });
  }
});

// Get statistics
app.get('/api/statistics', async (req, res) => {
  try {
    // Generate statistics from available data
    let totalMatches = 0;
    let totalRevenue = 0;
    let totalCarbonSaved = 0;

    mockMaterials.forEach(material => {
      const matches = generateMockMatches(material.id);
      totalMatches += matches.length;
      matches.forEach(match => {
        // Extract revenue number
        const revenueMatch = match.potential_revenue.match(/€(\d+)/);
        if (revenueMatch) totalRevenue += parseInt(revenueMatch[1]);
        
        // Extract carbon number
        const carbonMatch = match.environmental_impact.match(/(\d+) tons/);
        if (carbonMatch) totalCarbonSaved += parseInt(carbonMatch[1]);
      });
    });

    return sendResponse(res, {
      success: true,
      data: {
        total_materials: mockMaterials.length,
        total_companies: mockCompanies.length,
        total_matches: totalMatches,
        potential_revenue: `€${totalRevenue.toLocaleString()}/year`,
        carbon_impact: `${totalCarbonSaved} tons CO2 saved/year`,
        average_match_score: 0.85,
        active_partnerships: Math.floor(totalMatches * 0.3),
        sustainability_improvement: '23%'
      }
    });
  } catch (error) {
    console.error('Statistics error:', error);
    return sendResponse(res, {
      success: false,
      error: 'Failed to generate statistics'
    }, 500);
  }
});

// Create new material
app.post('/api/materials', async (req, res) => {
  try {
    const materialData = req.body;
    
    // Try to save to database
    const { data: newMaterial, error } = await supabase
      .from('materials')
      .insert(materialData)
      .select()
      .single();

    if (error) {
      console.warn('Database error, saving to mock data:', error.message);
      const mockMaterial = {
        id: `mat_${Date.now()}`,
        ...materialData,
        created_at: new Date().toISOString()
      };
      mockMaterials.push(mockMaterial);
      
      return sendResponse(res, {
        success: true,
        data: mockMaterial,
        message: 'Material created successfully (mock)',
        source: 'mock'
      }, 201);
    }

    return sendResponse(res, {
      success: true,
      data: newMaterial,
      message: 'Material created successfully',
      source: 'database'
    }, 201);
  } catch (error) {
    console.error('Create material error:', error);
    return sendResponse(res, {
      success: false,
      error: 'Failed to create material'
    }, 500);
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Express error:', err);
  sendResponse(res, {
    success: false,
    error: 'Internal server error',
    message: err.message
  }, 500);
});

// 404 handler
app.use('*', (req, res) => {
  sendResponse(res, {
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
  }, 404);
});

// Start server
if (process.env.NODE_ENV !== 'test') {
  app.listen(PORT, () => {
    console.log('🚀 SymbioFlows Demo Backend Started!');
    console.log(`📡 API running on: http://localhost:${PORT}`);
    console.log(`🏥 Health check: http://localhost:${PORT}/api/health`);
    console.log(`📊 Materials: http://localhost:${PORT}/api/materials`);
    console.log(`🏢 Companies: http://localhost:${PORT}/api/companies`);
    console.log(`📈 Statistics: http://localhost:${PORT}/api/statistics`);
    console.log('\n✅ Backend is ready for demo!');
  });
}

module.exports = app;