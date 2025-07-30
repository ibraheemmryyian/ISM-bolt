// Super simple backend for immediate material generation demo
require('dotenv').config();

const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

console.log('🚀 Starting Simple Backend...');

// Health check endpoints
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    database: 'connected',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    database: 'connected',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Test endpoint
app.get('/api/test', (req, res) => {
  res.json({
    status: 'OK',
    message: 'Simple backend is working!',
    timestamp: new Date().toISOString()
  });
});

// Mock material generation endpoint
app.post('/api/ai-portfolio-generation', async (req, res) => {
  try {
    console.log('🚀 Generating materials for company...');
    
    const companyProfile = req.body.companyProfile || {};
    console.log('Company:', companyProfile.name || 'Unknown');
    
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Generate mock materials based on company profile
    const mockMaterials = {
      predicted_outputs: [
        {
          id: '1',
          material: 'Steel Scrap',
          quantity: '500 tons/month',
          quality: 'High grade',
          location: companyProfile.location || 'Manufacturing plant',
          availability: 'Immediate',
          description: 'High-quality steel scrap from manufacturing processes',
          symbiosis_opportunities: ['Construction companies', 'Automotive recycling', 'Metal foundries'],
          estimated_value: '$25,000/month',
          carbon_impact: 'Reduces 150 tons CO2/month'
        },
        {
          id: '2',
          material: 'Heat Waste',
          quantity: '1000 kWh/day',
          quality: 'Medium temperature (200°C)',
          location: companyProfile.location || 'Production facility',
          availability: 'Continuous',
          description: 'Excess heat from manufacturing processes',
          symbiosis_opportunities: ['Greenhouse operations', 'District heating', 'Food processing'],
          estimated_value: '$15,000/month',
          carbon_impact: 'Reduces 80 tons CO2/month'
        },
        {
          id: '3',
          material: 'Wood Waste',
          quantity: '200 tons/month',
          quality: 'Clean, untreated',
          location: companyProfile.location || 'Workshop',
          availability: 'Weekly',
          description: 'Clean wood waste from manufacturing',
          symbiosis_opportunities: ['Biomass energy', 'Particle board', 'Landscaping'],
          estimated_value: '$8,000/month',
          carbon_impact: 'Reduces 40 tons CO2/month'
        }
      ],
      predicted_inputs: [
        {
          id: '4',
          material: 'Recycled Plastic',
          quantity: '200 tons/month',
          quality: 'Food grade PET',
          location: companyProfile.location || 'Processing facility',
          urgency: 'High',
          description: 'Need recycled plastic for sustainable packaging',
          symbiosis_opportunities: ['Waste management companies', 'Recycling centers', 'Packaging suppliers'],
          estimated_cost: '$12,000/month',
          carbon_benefit: 'Avoids 100 tons CO2/month'
        },
        {
          id: '5',
          material: 'Organic Waste',
          quantity: '50 tons/month',
          quality: 'Compostable',
          location: companyProfile.location || 'Facility',
          urgency: 'Medium',
          description: 'Organic waste for composting operations',
          symbiosis_opportunities: ['Farms', 'Composting facilities', 'Biogas plants'],
          estimated_cost: '$5,000/month',
          carbon_benefit: 'Avoids 30 tons CO2/month'
        }
      ]
    };
    
    console.log(`✅ Generated ${mockMaterials.predicted_outputs.length} outputs and ${mockMaterials.predicted_inputs.length} inputs`);
    
    res.json({
      success: true,
      portfolio: mockMaterials,
      message: 'Material generation completed successfully',
      outputs_count: mockMaterials.predicted_outputs.length,
      inputs_count: mockMaterials.predicted_inputs.length,
      company: companyProfile.name || 'Your Company',
      processing_time: '2.1 seconds'
    });
    
  } catch (error) {
    console.error('❌ Material generation error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      message: 'Material generation failed'
    });
  }
});

// Onboarding completion endpoint
app.post('/api/complete-onboarding', async (req, res) => {
  try {
    const { companyProfile } = req.body;
    
    console.log('🎯 Completing onboarding for:', companyProfile?.name || 'Unknown company');
    
    // Simulate processing
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    res.json({
      success: true,
      message: 'Onboarding completed successfully. Your profile has been created and analyzed for symbiosis opportunities.',
      next_steps: [
        'Review your generated material listings',
        'Explore symbiosis opportunities', 
        'Connect with potential partners'
      ],
      materials_generated: true,
      ready_for_matching: true
    });
    
  } catch (error) {
    console.error('❌ Onboarding completion error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get materials listing endpoint
app.get('/api/materials', async (req, res) => {
  try {
    console.log('📋 Fetching materials...');
    
    const materials = {
      outputs: [
        {
          id: '1',
          material: 'Steel Scrap',
          quantity: '500 tons/month',
          quality: 'High grade',
          location: 'Manufacturing plant',
          availability: 'Immediate',
          symbiosis_opportunities: ['Construction', 'Automotive recycling']
        },
        {
          id: '2', 
          material: 'Heat Waste',
          quantity: '1000 kWh/day',
          quality: 'Medium temperature',
          location: 'Production facility',
          availability: 'Continuous',
          symbiosis_opportunities: ['Greenhouse heating', 'District heating']
        }
      ],
      inputs: [
        {
          id: '3',
          material: 'Recycled Plastic',
          quantity: '200 tons/month',
          quality: 'Food grade',
          location: 'Processing facility',
          urgency: 'High',
          symbiosis_opportunities: ['Packaging', 'Consumer goods']
        }
      ]
    };
    
    res.json({
      success: true,
      materials: materials,
      total_outputs: materials.outputs.length,
      total_inputs: materials.inputs.length
    });
    
  } catch (error) {
    console.error('❌ Materials fetch error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Simple Backend Server running on port ${PORT}`);
  console.log(`📡 API available at http://localhost:${PORT}/api`);
  console.log(`🏥 Health check at http://localhost:${PORT}/api/health`);
  console.log(`🧪 Test endpoint at http://localhost:${PORT}/api/test`);
  console.log(`📦 Material generation at http://localhost:${PORT}/api/ai-portfolio-generation`);
  console.log(`🎯 Onboarding completion at http://localhost:${PORT}/api/complete-onboarding`);
  console.log('✅ Ready to handle material generation requests!');
});

module.exports = app;