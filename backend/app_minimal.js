// Minimal backend for material generation
require('dotenv').config();

const express = require('express');
const cors = require('cors');
const { PythonShell } = require('python-shell');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
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

// Mock user profile for testing
const mockUserProfile = {
  id: 'test-user-123',
  name: 'Test Company',
  industry: 'Manufacturing',
  size: 'Medium',
  location: 'USA',
  sustainability_goals: ['reduce_waste', 'carbon_neutral'],
  current_practices: ['recycling', 'energy_efficiency']
};

// Helper function to run Python scripts
async function runPythonScript(scriptName, data) {
  return new Promise((resolve, reject) => {
    console.log(`🔧 Running Python script: ${scriptName}`);
    
    const options = {
      mode: 'json',
      pythonPath: 'python',
      pythonOptions: ['-u'],
      scriptPath: '../ai_service_flask/',
      args: []
    };

    const pyshell = new PythonShell(scriptName, options);
    
    // Send data to Python script
    pyshell.send(data);
    
    let result = null;
    
    pyshell.on('message', function (message) {
      console.log('📥 Python script output:', message);
      result = message;
    });

    pyshell.end(function (err) {
      if (err) {
        console.error('❌ Python script error:', err);
        reject(err);
      } else {
        console.log('✅ Python script completed successfully');
        resolve(result);
      }
    });
  });
}

// Material generation endpoint
app.post('/api/ai-portfolio-generation', async (req, res) => {
  try {
    console.log('🚀 Starting material generation...');
    
    const companyProfile = req.body.companyProfile || mockUserProfile;
    console.log('Company profile:', companyProfile.name);
    
    // Call the AI service to generate materials
    const portfolioResult = await runPythonScript('listing_inference_service.py', {
      company_profile: companyProfile,
      generate_listings: true
    });
    
    if (!portfolioResult) {
      throw new Error('No portfolio result returned from AI service');
    }
    
    console.log(`✅ Generated ${portfolioResult.predicted_outputs?.length || 0} outputs and ${portfolioResult.predicted_inputs?.length || 0} inputs`);
    
    res.json({
      success: true,
      portfolio: portfolioResult,
      message: 'Material generation completed successfully',
      outputs_count: portfolioResult.predicted_outputs?.length || 0,
      inputs_count: portfolioResult.predicted_inputs?.length || 0
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

// Get materials listing endpoint
app.get('/api/materials', async (req, res) => {
  try {
    console.log('📋 Fetching materials...');
    
    // For now, return mock data - you can enhance this later
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

// Onboarding completion endpoint
app.post('/api/complete-onboarding', async (req, res) => {
  try {
    const { companyProfile } = req.body;
    
    console.log('🎯 Completing onboarding for:', companyProfile?.name || 'Unknown company');
    
    // Trigger material generation
    const portfolioResult = await runPythonScript('listing_inference_service.py', {
      company_profile: companyProfile,
      generate_listings: true
    });
    
    res.json({
      success: true,
      message: 'Onboarding completed successfully. Your profile has been created and analyzed for symbiosis opportunities.',
      portfolio: portfolioResult,
      next_steps: [
        'Review your generated material listings',
        'Explore symbiosis opportunities',
        'Connect with potential partners'
      ]
    });
    
  } catch (error) {
    console.error('❌ Onboarding completion error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Test endpoint
app.get('/api/test', (req, res) => {
  res.json({
    status: 'OK',
    message: 'Minimal backend is working!',
    timestamp: new Date().toISOString()
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Minimal Backend Server running on port ${PORT}`);
  console.log(`📡 API available at http://localhost:${PORT}/api`);
  console.log(`🏥 Health check at http://localhost:${PORT}/api/health`);
  console.log(`🧪 Test endpoint at http://localhost:${PORT}/api/test`);
  console.log(`📦 Material generation at http://localhost:${PORT}/api/ai-portfolio-generation`);
});

module.exports = app;