// Load environment variables from .env file
require('dotenv').config();

const express = require('express');
const { PythonShell } = require('python-shell');
const crypto = require('crypto');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const cors = require('cors');
const { body, validationResult } = require('express-validator');
const fetch = require('node-fetch');
const { supabase } = require('./supabase');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');
const { spawn } = require('child_process');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const shippingService = require('./services/shippingService');
const materialsService = require('./services/materialsService');
const paymentService = require('./services/paymentService');
const heightService = require('./services/heightService');
const intelligentMatchingService = require('./services/intelligentMatchingService');
const apiFusionService = require('./services/apiFusionService');
const aiEvolutionEngine = require('./services/aiEvolutionEngine');
const FreightosLogisticsService = require('./services/freightosLogisticsService');
const freightosService = new FreightosLogisticsService();
const requestWithRetry = require('./utils/requestWithRetry');
const swaggerJsdoc = require('swagger-jsdoc');
const swaggerUi = require('swagger-ui-express');
// Adaptive onboarding is handled via proxy to Python Flask server

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true
}));

// Body parsing middleware - MUST come before routes
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// JSON parsing error handler
app.use((error, req, res, next) => {
  if (error instanceof SyntaxError && error.status === 400 && 'body' in error) {
    return res.status(400).json({
      error: 'Invalid JSON',
      message: 'Malformed JSON in request body'
    });
  }
  next(error);
});

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use('/api/', limiter);

// Input validation middleware
const validateAIInference = [
  body('companyName').isString().trim().isLength({ min: 1, max: 100 }),
  body('industry').isString().trim().isLength({ min: 1, max: 100 }),
  body('products').isString().trim().isLength({ min: 1, max: 500 }),
  body('location').isString().trim().isLength({ min: 1, max: 200 }),
  body('productionVolume').isString().trim().isLength({ min: 1, max: 100 }),
  body('mainMaterials').isString().trim().isLength({ min: 1, max: 500 }),
  body('processDescription').optional().isString().trim().isLength({ max: 1000 }),
];

const validateUserFeedback = [
  body('matchId').isString().trim().isLength({ min: 1, max: 100 }),
  body('userId').isString().trim().isLength({ min: 1, max: 100 }),
  body('rating').isInt({ min: 1, max: 5 }),
  body('feedback').optional().isString().trim().isLength({ max: 1000 }),
  body('categories').optional().isArray(),
];

const validateSymbiosisRequest = [
  body('participants').isArray({ min: 2, max: 50 }),
  body('participants.*.id').isString().trim().isLength({ min: 1, max: 100 }),
  body('participants.*.industry').isString().trim().isLength({ min: 1, max: 100 }),
  body('participants.*.annual_waste').optional().isNumeric(),
  body('participants.*.carbon_footprint').optional().isNumeric(),
  body('participants.*.waste_type').optional().isString().trim(),
  body('participants.*.location').optional().isString().trim(),
];

// Error handling middleware
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      error: 'Validation failed',
      details: errors.array()
    });
  }
  next();
};

// Global error handler
app.use((error, req, res, next) => {
  console.error('Global error:', error);
  
  // Log to Sentry if available
  if (process.env.SENTRY_DSN) {
    const Sentry = require('@sentry/node');
    Sentry.captureException(error);
  }
  
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'production' ? 'Something went wrong' : error.message
  });
});

// --- Standardized API Response Utility ---
function sendResponse(res, { success, data = null, error = null, message = null }, statusCode = 200) {
  const response = { success };
  if (data !== null) response.data = data;
  if (error !== null) response.error = error;
  if (message !== null) response.message = message;
  return res.status(statusCode).json(response);
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  sendResponse(res, {
    success: true,
    data: {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0'
    }
  });
});

// Logging endpoints for frontend error handling and monitoring
app.post('/api/logs', async (req, res) => {
  try {
    const { timestamp, level, message, data, userAgent, url } = req.body;
    
    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`[${level.toUpperCase()}] ${message}`, data || '');
    }
    
    // Store in database for production monitoring
    if (process.env.NODE_ENV === 'production') {
      const { error } = await supabase
        .from('application_logs')
        .insert({
          timestamp: timestamp || new Date().toISOString(),
          level,
          message,
          data: data ? JSON.stringify(data) : null,
          user_agent: userAgent,
          url,
          source: 'frontend'
        });
      
      if (error) {
        console.error('Failed to store log:', error);
      }
    }
    
    sendResponse(res, { success: true, message: 'Log stored' });
  } catch (error) {
    sendResponse(res, { success: false, error: 'Failed to log message' }, 500);
  }
});

app.post('/api/logs/error', async (req, res) => {
  try {
    const { errorId, message, stack, componentStack, userAgent, url, timestamp } = req.body;
    
    // Log error to console
    console.error(`[ERROR] ${errorId}: ${message}`, {
      stack,
      componentStack,
      userAgent,
      url,
      timestamp
    });
    
    // Store error in database for production monitoring
    if (process.env.NODE_ENV === 'production') {
      const { error } = await supabase
        .from('error_logs')
        .insert({
          error_id: errorId,
          message,
          stack_trace: stack,
          component_stack: componentStack,
          user_agent: userAgent,
          url,
          timestamp: timestamp || new Date().toISOString(),
          source: 'frontend'
        });
      
      if (error) {
        console.error('Failed to store error log:', error);
      }
    }
    
    // Send to Sentry if configured
    if (process.env.SENTRY_DSN) {
      const Sentry = require('@sentry/node');
      Sentry.captureException(new Error(message), {
        extra: {
          errorId,
          stack,
          componentStack,
          userAgent,
          url
        }
      });
    }
    
    res.json({ success: true });
  } catch (error) {
    console.error('Error logging error:', error);
    res.status(500).json({ error: 'Failed to log error' });
  }
});

// Test route
app.get('/api', (req, res) => {
  res.json({ message: 'Industrial AI Marketplace API' });
});

// Supabase configuration
const supabaseUrl = process.env.SUPABASE_URL || 'https://your-project.supabase.co';
const supabaseKey = process.env.SUPABASE_ANON_KEY || 'your-anon-key';
const supabaseClient = createClient(supabaseUrl, supabaseKey);
const supabaseServiceClient = createClient(supabaseUrl, process.env.SUPABASE_SERVICE_ROLE_KEY);

// Helper function to run Python scripts
function runPythonScript(scriptPath, data) {
    return new Promise((resolve, reject) => {
        // Handle both object and array parameters
        const args = Array.isArray(data) ? [scriptPath, ...data] : [scriptPath, JSON.stringify(data)];
        const pythonProcess = spawn('python', args, {
            timeout: 10000, // 10 second timeout
            stdio: ['pipe', 'pipe', 'pipe'],
            cwd: __dirname // Set working directory to backend folder
        });
        
        let result = '';
        let error = '';
        
        pythonProcess.stdout.on('data', (data) => {
            result += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            error += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Python script failed: ${error}`));
            } else {
                try {
                    const parsedResult = JSON.parse(result);
                    resolve(parsedResult);
                } catch (e) {
                    reject(new Error(`Failed to parse Python output: ${result}`));
                }
            }
        });
        
        pythonProcess.on('error', (err) => {
            reject(new Error(`Failed to start Python process: ${err.message}`));
        });
    });
}

// Helper function to save portfolio to database
async function savePortfolioToDatabase(companyId, portfolioData) {
    try {
        const { supabase } = require('./supabase');
        
        const savedMaterials = [];
        const savedRequirements = [];
        
        // Save predicted outputs (materials)
        if (portfolioData.predicted_outputs && portfolioData.predicted_outputs.length > 0) {
            for (const material of portfolioData.predicted_outputs) {
                const { data: savedMaterial, error: materialError } = await supabase
                    .from('materials')
                    .insert({
                        company_id: companyId,
                        name: material.name,
                        description: material.description,
                        category: material.category,
                        quantity_estimate: material.quantity,
                        quantity: material.quantity,
                        frequency: material.frequency,
                        notes: material.notes,
                        potential_value: material.potential_value || 'Unknown',
                        quality_grade: material.quality_grade || 'medium',
                        potential_uses: material.potential_uses || [],
                        symbiosis_opportunities: material.symbiosis_opportunities || [],
                        ai_generated: true
                    })
                    .select()
                    .single();
                
                if (materialError) {
                    console.error('Error saving material:', materialError);
                } else {
                    savedMaterials.push(savedMaterial);
                }
            }
        }
        
        // Save predicted inputs (requirements)
        if (portfolioData.predicted_inputs && portfolioData.predicted_inputs.length > 0) {
            for (const requirement of portfolioData.predicted_inputs) {
                const { data: savedRequirement, error: requirementError } = await supabase
                    .from('requirements')
                    .insert({
                        company_id: companyId,
                        name: requirement.name,
                        description: requirement.description,
                        category: requirement.category,
                        quantity_needed: requirement.quantity,
                        quantity: requirement.quantity,
                        frequency: requirement.frequency,
                        notes: requirement.notes,
                        current_cost: requirement.current_cost || 'Unknown',
                        priority: requirement.priority || 'medium',
                        potential_sources: requirement.potential_sources || [],
                        symbiosis_opportunities: requirement.symbiosis_opportunities || [],
                        ai_generated: true
                    })
                    .select()
                    .single();
                
                if (requirementError) {
                    console.error('Error saving requirement:', requirementError);
                } else {
                    savedRequirements.push(savedRequirement);
                }
            }
        }
        
        return {
            materials: savedMaterials,
            requirements: savedRequirements,
            total_materials: savedMaterials.length,
            total_requirements: savedRequirements.length
        };
        
    } catch (error) {
        console.error('Error saving portfolio to database:', error);
        throw error;
    }
}

// REAL CARBON CALCULATION ENDPOINT
app.post('/api/carbon-calculate', async (req, res) => {
    try {
        const companyData = req.body;
        
        // Call real carbon calculation engine
        const carbonResult = await runPythonScript('carbon_calculation_engine.py', {
            action: 'calculate_carbon_footprint',
            company_data: companyData
        });
        
        res.json(carbonResult);
    } catch (error) {
        console.error('Carbon calculation error:', error);
        res.status(500).json({ error: error.message });
    }
});

// REAL WASTE TRACKING ENDPOINT
app.post('/api/waste-calculate', async (req, res) => {
    try {
        const companyData = req.body;
        
        // Call real waste tracking engine
        const wasteResult = await runPythonScript('waste_tracking_engine.py', {
            action: 'calculate_waste_profile',
            company_data: companyData
        });
        
        res.json(wasteResult);
    } catch (error) {
        console.error('Waste calculation error:', error);
        res.status(500).json({ error: error.message });
    }
});

// PHASE 1: DEEP PORTFOLIO GENERATION ENDPOINT
app.post('/api/ai-portfolio-generation', async (req, res) => {
    try {
        const companyProfile = req.body;
        
        console.log('Starting Phase 1: Deep Portfolio Generation for company:', companyProfile.name);
        
        // Call Phase 1: DeepSeek API for portfolio generation
        const portfolioResult = await runPythonScript('listing_inference_service.py', {
            action: 'generate_listings_from_profile',
            company_profile: companyProfile
        });
        
        // Save the generated portfolio to database
        const savedPortfolio = await savePortfolioToDatabase(companyProfile.id, portfolioResult);
        
        console.log(`Phase 1 completed: Generated ${portfolioResult.predicted_outputs?.length || 0} outputs and ${portfolioResult.predicted_inputs?.length || 0} inputs`);
        
        res.json({
            success: true,
            phase: 1,
            portfolio: savedPortfolio,
            message: 'Portfolio generation completed successfully'
        });
        
    } catch (error) {
        console.error('Phase 1 Portfolio Generation error:', error);
        res.status(500).json({ error: error.message });
    }
});

// PHASE 2: AI-POWERED MATCHMAKING ENDPOINT
app.post('/api/ai-matchmaking', async (req, res) => {
    try {
        const { company_id, material_data } = req.body;
        
        console.log('Starting Phase 2: AI-Powered Matchmaking for company:', company_id);
        
        // Call Phase 2: DeepSeek API for partner company recommendations
        const partnerResult = await runPythonScript('ai_matchmaking_service.py', {
            action: 'find_partner_companies',
            company_id: company_id,
            material_data: material_data
        });
        
        // Create matches in database
        const createdMatches = await runPythonScript('ai_matchmaking_service.py', {
            action: 'create_matches_in_database',
            company_id: company_id,
            partner_companies: partnerResult,
            material_name: material_data.name
        });
        
        console.log(`Phase 2 completed: Found ${partnerResult?.length || 0} partner companies and created ${createdMatches?.length || 0} matches`);
        
        res.json({
            success: true,
            phase: 2,
            partner_companies: partnerResult,
            created_matches: createdMatches,
            message: 'Matchmaking completed successfully'
        });
        
    } catch (error) {
        console.error('Phase 2 AI Matchmaking error:', error);
        res.status(500).json({ error: error.message });
    }
});

// ORCHESTRATED TWO-PHASE AI PIPELINE ENDPOINT
app.post('/api/ai-pipeline', async (req, res) => {
    try {
        const companyProfile = req.body;
        
        console.log('Starting Orchestrated Two-Phase AI Pipeline for company:', companyProfile.name);
        
        // Phase 1: Generate portfolio
        console.log('Executing Phase 1: Deep Portfolio Generation...');
        const portfolioResult = await runPythonScript('listing_inference_service.py', {
            action: 'generate_listings_from_profile',
            company_profile: companyProfile
        });
        
        // Save portfolio to database
        const savedPortfolio = await savePortfolioToDatabase(companyProfile.id, portfolioResult);
        
        // Phase 2: Find matches for each output material
        console.log('Executing Phase 2: AI-Powered Matchmaking...');
        const allMatches = [];
        
        if (portfolioResult.predicted_outputs && portfolioResult.predicted_outputs.length > 0) {
            for (const material of portfolioResult.predicted_outputs) {
                const partnerResult = await runPythonScript('ai_matchmaking_service.py', {
                    action: 'find_partner_companies',
                    company_id: companyProfile.id,
                    material_data: material
                });
                
                const createdMatches = await runPythonScript('ai_matchmaking_service.py', {
                    action: 'create_matches_in_database',
                    company_id: companyProfile.id,
                    partner_companies: partnerResult,
                    material_name: material.name
                });
                
                allMatches.push(...createdMatches);
            }
        }
        
        console.log(`Two-Phase AI Pipeline completed: Generated portfolio with ${portfolioResult.predicted_outputs?.length || 0} outputs and created ${allMatches.length} matches`);
        
        sendResponse(res, {
          success: true,
          data: {
            phase: 'complete',
            portfolio: savedPortfolio,
            matches_created: allMatches.length,
            total_matches: allMatches,
          },
          message: 'Two-phase AI pipeline completed successfully'
        });
        
    } catch (error) {
        console.error('Two-Phase AI Pipeline error:', error);
        sendResponse(res, { success: false, error: error.message }, 500);
    }
});

// LEGACY AI MATCHING ENDPOINT (for backward compatibility)
app.post('/api/ai-match', async (req, res) => {
    try {
        const companyData = req.body;
        
        // Call real AI matching engine
        const matchResult = await runPythonScript('real_ai_matching_engine.py', {
            action: 'find_symbiotic_matches',
            company_data: companyData,
            top_k: 10
        });
        
        res.json(matchResult);
    } catch (error) {
        console.error('AI matching error:', error);
        res.status(500).json({ error: error.message });
    }
});

// REAL SUSTAINABILITY INITIATIVES ENDPOINT
app.post('/api/sustainability-initiatives', async (req, res) => {
    try {
        const companyData = req.body;
        
        // Get carbon footprint first
        const carbonResult = await runPythonScript('carbon_calculation_engine.py', {
            action: 'calculate_carbon_footprint',
            company_data: companyData
        });
        
        // Get waste profile
        const wasteResult = await runPythonScript('waste_tracking_engine.py', {
            action: 'calculate_waste_profile',
            company_data: companyData
        });
        
        // Generate comprehensive sustainability initiatives
        const initiatives = [];
        
        // Carbon reduction initiatives
        if (carbonResult.total_carbon_footprint > 1000) {
            initiatives.push({
                id: 'carbon_1',
                category: 'Energy Efficiency',
                question: 'Would you like to implement energy-efficient lighting and HVAC systems?',
                description: 'Reduce energy consumption by 15-20% through smart lighting and climate control',
                potential_savings: carbonResult.total_carbon_footprint * 0.15 * 50, // $50 per ton CO2
                implementation_cost: 75000,
                payback_period: '2-3 years'
            });
        }
        
        // Waste reduction initiatives
        if (wasteResult.total_waste_generated > 500) {
            initiatives.push({
                id: 'waste_1',
                category: 'Waste Management',
                question: 'Are you interested in implementing a comprehensive recycling program?',
                description: 'Increase recycling rate and reduce disposal costs',
                potential_savings: wasteResult.waste_costs.total_net_cost * 0.25,
                implementation_cost: 50000,
                payback_period: '1-2 years'
            });
        }
        
        // Water conservation initiatives
        if (companyData.water_usage > 1000) {
            initiatives.push({
                id: 'water_1',
                category: 'Water Conservation',
                question: 'Would you like to implement water recycling and conservation systems?',
                description: 'Reduce water consumption by 20-30% through recycling and efficient usage',
                potential_savings: companyData.water_usage * 0.25 * 2, // $2 per m3
                implementation_cost: 40000,
                payback_period: '1-2 years'
            });
        }
        
        // Renewable energy initiatives
        initiatives.push({
            id: 'renewable_1',
            category: 'Renewable Energy',
            question: 'Are you interested in installing solar panels or wind turbines?',
            description: 'Generate clean energy and reduce carbon footprint',
            potential_savings: carbonResult.energy_emissions * 0.25 * 0.12, // $0.12 per kWh
            implementation_cost: 150000,
            payback_period: '5-7 years'
        });
        
        // Supply chain optimization
        initiatives.push({
            id: 'supply_1',
            category: 'Supply Chain',
            question: 'Would you like to optimize your supply chain for sustainability?',
            description: 'Reduce transportation emissions and costs through route optimization',
            potential_savings: carbonResult.transport_emissions * 0.20 * 50,
            implementation_cost: 30000,
            payback_period: '1 year'
        });
        
        res.json({
            initiatives: initiatives,
            carbon_footprint: carbonResult,
            waste_profile: wasteResult,
            total_potential_savings: initiatives.reduce((sum, i) => sum + i.potential_savings, 0)
        });
        
    } catch (error) {
        console.error('Sustainability initiatives error:', error);
        res.status(500).json({ error: error.message });
    }
});

// REAL PORTFOLIO RECOMMENDATIONS ENDPOINT
app.post('/api/portfolio-recommendations', async (req, res) => {
    try {
        const companyData = req.body;
        
        // Get AI matches
        const matches = await runPythonScript('real_ai_matching_engine.py', {
            action: 'find_symbiotic_matches',
            company_data: companyData,
            top_k: 5
        });
        
        // Get carbon and waste data
        const carbonResult = await runPythonScript('carbon_calculation_engine.py', {
            action: 'calculate_carbon_footprint',
            company_data: companyData
        });
        
        const wasteResult = await runPythonScript('waste_tracking_engine.py', {
            action: 'calculate_waste_profile',
            company_data: companyData
        });
        
        // Generate portfolio recommendations
        const recommendations = [];
        
        // Top matches
        if (matches && matches.length > 0) {
            recommendations.push({
                type: 'symbiotic_matches',
                title: 'Industrial Symbiosis Opportunities',
                description: 'Companies you can partner with for mutual benefit',
                items: matches.slice(0, 3).map(match => ({
                    name: match.company_name,
                    industry: match.industry,
                    match_score: match.match_score,
                    potential_savings: match.potential_savings,
                    description: match.description
                })),
                priority: 'high'
            });
        }
        
        // Carbon reduction opportunities
        if (carbonResult.total_carbon_footprint > 500) {
            recommendations.push({
                type: 'carbon_reduction',
                title: 'Carbon Footprint Reduction',
                description: 'Opportunities to reduce your environmental impact',
                items: [
                    {
                        name: 'Energy Efficiency Upgrade',
                        potential_reduction: carbonResult.energy_emissions * 0.15,
                        cost_savings: carbonResult.energy_emissions * 0.15 * 50,
                        implementation_time: '3-6 months'
                    },
                    {
                        name: 'Renewable Energy Integration',
                        potential_reduction: carbonResult.energy_emissions * 0.25,
                        cost_savings: carbonResult.energy_emissions * 0.25 * 0.12,
                        implementation_time: '6-12 months'
                    }
                ],
                priority: 'medium'
            });
        }
        
        // Waste optimization
        if (wasteResult.total_waste_generated > 300) {
            recommendations.push({
                type: 'waste_optimization',
                title: 'Waste Management Optimization',
                description: 'Improve waste handling and recycling',
                items: [
                    {
                        name: 'Recycling Program Enhancement',
                        potential_reduction: wasteResult.total_waste_generated * 0.20,
                        cost_savings: wasteResult.waste_costs.total_net_cost * 0.20,
                        implementation_time: '2-4 months'
                    },
                    {
                        name: 'Waste-to-Energy Implementation',
                        potential_reduction: wasteResult.total_waste_generated * 0.15,
                        cost_savings: wasteResult.total_waste_generated * 0.15 * 30,
                        implementation_time: '6-9 months'
                    }
                ],
                priority: 'medium'
            });
        }
        
        // Market opportunities
        recommendations.push({
            type: 'market_opportunities',
            title: 'Market Expansion Opportunities',
            description: 'New markets and business opportunities',
            items: [
                {
                    name: 'Sustainable Product Line',
                    market_size: 'Growing 15% annually',
                    entry_cost: 100000,
                    potential_revenue: 500000,
                    timeline: '12-18 months'
                },
                {
                    name: 'Circular Economy Services',
                    market_size: 'Emerging market',
                    entry_cost: 75000,
                    potential_revenue: 300000,
                    timeline: '9-15 months'
                }
            ],
            priority: 'low'
        });
        
        res.json({
            recommendations: recommendations,
            summary: {
                total_opportunities: recommendations.length,
                high_priority: recommendations.filter(r => r.priority === 'high').length,
                estimated_total_value: recommendations.reduce((sum, rec) => {
                    return sum + rec.items.reduce((item_sum, item) => {
                        return item_sum + (item.cost_savings || item.potential_revenue || 0);
                    }, 0);
                }, 0)
            }
        });
        
    } catch (error) {
        console.error('Portfolio recommendations error:', error);
        res.status(500).json({ error: error.message });
    }
});

// REAL AI INSIGHTS ENDPOINT
app.post('/api/ai-insights', async (req, res) => {
    try {
        const companyData = req.body;
        
        // Get comprehensive data
        const carbonResult = await runPythonScript('carbon_calculation_engine.py', {
            action: 'calculate_carbon_footprint',
            company_data: companyData
        });
        
        const wasteResult = await runPythonScript('waste_tracking_engine.py', {
            action: 'calculate_waste_profile',
            company_data: companyData
        });
        
        const matches = await runPythonScript('real_ai_matching_engine.py', {
            action: 'find_symbiotic_matches',
            company_data: companyData,
            top_k: 3
        });
        
        // Generate AI insights
        const insights = [];
        
        // Performance insights
        const efficiency_score = carbonResult.efficiency_metrics.efficiency_score;
        if (efficiency_score < 60) {
            insights.push({
                type: 'performance',
                title: 'Efficiency Improvement Opportunity',
                description: `Your efficiency score of ${efficiency_score} indicates significant room for improvement.`,
                recommendation: 'Focus on energy efficiency and process optimization to improve your score.',
                impact: 'high',
                timeline: '6-12 months'
            });
        }
        
        // Cost optimization insights
        const total_costs = carbonResult.total_carbon_footprint * 50 + wasteResult.waste_costs.total_net_cost;
        if (total_costs > 100000) {
            insights.push({
                type: 'cost_optimization',
                title: 'High Operational Costs',
                description: `Your current operational costs are ${total_costs.toLocaleString()} USD annually.`,
                recommendation: 'Implement cost-saving measures through waste reduction and energy efficiency.',
                impact: 'high',
                timeline: '3-6 months'
            });
        }
        
        // Partnership insights
        if (matches && matches.length > 0) {
            const topMatch = matches[0];
            insights.push({
                type: 'partnership',
                title: 'High-Potential Partnership',
                description: `Strong match found with ${topMatch.company_name} (${topMatch.match_score} score).`,
                recommendation: `Consider reaching out to explore ${topMatch.match_type} opportunities.`,
                impact: 'medium',
                timeline: '1-3 months'
            });
        }
        
        // Sustainability insights
        const sustainability_score = companyData.sustainability_score || 50;
        if (sustainability_score < 70) {
            insights.push({
                type: 'sustainability',
                title: 'Sustainability Enhancement',
                description: `Your sustainability score of ${sustainability_score} can be improved.`,
                recommendation: 'Focus on renewable energy, waste reduction, and circular economy practices.',
                impact: 'medium',
                timeline: '6-18 months'
            });
        }
        
        // Market insights
        insights.push({
            type: 'market',
            title: 'Market Position Analysis',
            description: `You operate in the ${companyData.industry} sector with ${companyData.employee_count} employees.`,
            recommendation: 'Consider expanding into sustainable product lines and circular economy services.',
            impact: 'low',
            timeline: '12-24 months'
        });
        
        res.json({
            insights: insights,
            summary: {
                total_insights: insights.length,
                high_impact: insights.filter(i => i.impact === 'high').length,
                immediate_actions: insights.filter(i => i.timeline.includes('1-3') || i.timeline.includes('3-6')).length
            },
            data_sources: {
                carbon_footprint: carbonResult,
                waste_profile: wasteResult,
                ai_matches: matches
            }
        });
        
    } catch (error) {
        console.error('AI insights error:', error);
        res.status(500).json({ error: error.message });
    }
});

// REAL ONBOARDING FLOW ENDPOINT
app.post('/api/onboarding-flow', async (req, res) => {
    try {
        const companyData = req.body;
        
        // Generate dynamic onboarding flow based on company data
        const onboardingFlow = {
            steps: [
                {
                    id: 'company_info',
                    title: 'Company Information',
                    description: 'Tell us about your company',
                    questions: [
                        {
                            id: 'name',
                            type: 'text',
                            label: 'Company Name',
                            required: true,
                            value: companyData.name || ''
                        },
                        {
                            id: 'industry',
                            type: 'select',
                            label: 'Industry',
                            required: true,
                            options: [
                                'manufacturing', 'textiles', 'food_beverage', 'chemicals',
                                'construction', 'electronics', 'automotive', 'pharmaceuticals'
                            ],
                            value: companyData.industry || ''
                        },
                        {
                            id: 'location',
                            type: 'text',
                            label: 'Location',
                            required: true,
                            value: companyData.location || ''
                        },
                        {
                            id: 'employee_count',
                            type: 'number',
                            label: 'Number of Employees',
                            required: true,
                            value: companyData.employee_count || 0
                        }
                    ]
                },
                {
                    id: 'materials_processes',
                    title: 'Materials & Processes',
                    description: 'What materials do you work with?',
                    questions: [
                        {
                            id: 'materials',
                            type: 'multiselect',
                            label: 'Primary Materials',
                            required: true,
                            options: [
                                'steel', 'aluminum', 'plastic', 'glass', 'cotton', 'polyester',
                                'chemicals', 'cement', 'wood', 'paper', 'textiles', 'electronics'
                            ],
                            value: companyData.materials || []
                        },
                        {
                            id: 'processes',
                            type: 'textarea',
                            label: 'Main Manufacturing Processes',
                            required: false,
                            value: companyData.processes || ''
                        }
                    ]
                },
                {
                    id: 'sustainability_goals',
                    title: 'Sustainability Goals',
                    description: 'What are your sustainability objectives?',
                    questions: [
                        {
                            id: 'carbon_reduction_target',
                            type: 'select',
                            label: 'Carbon Reduction Target',
                            required: false,
                            options: ['10%', '20%', '30%', '50%', 'Net Zero'],
                            value: companyData.carbon_reduction_target || ''
                        },
                        {
                            id: 'waste_reduction_target',
                            type: 'select',
                            label: 'Waste Reduction Target',
                            required: false,
                            options: ['15%', '25%', '35%', '50%', 'Zero Waste'],
                            value: companyData.waste_reduction_target || ''
                        },
                        {
                            id: 'renewable_energy_target',
                            type: 'select',
                            label: 'Renewable Energy Target',
                            required: false,
                            options: ['25%', '50%', '75%', '100%'],
                            value: companyData.renewable_energy_target || ''
                        }
                    ]
                },
                {
                    id: 'partnership_preferences',
                    title: 'Partnership Preferences',
                    description: 'What types of partnerships interest you?',
                    questions: [
                        {
                            id: 'material_exchange',
                            type: 'slider',
                            label: 'Material Exchange Partnerships',
                            required: false,
                            min: 0,
                            max: 100,
                            value: companyData.matching_preferences?.material_exchange * 100 || 50
                        },
                        {
                            id: 'waste_recycling',
                            type: 'slider',
                            label: 'Waste Recycling Partnerships',
                            required: false,
                            min: 0,
                            max: 100,
                            value: companyData.matching_preferences?.waste_recycling * 100 || 70
                        },
                        {
                            id: 'energy_sharing',
                            type: 'slider',
                            label: 'Energy Sharing Partnerships',
                            required: false,
                            min: 0,
                            max: 100,
                            value: companyData.matching_preferences?.energy_sharing * 100 || 40
                        }
                    ]
                }
            ],
            progress: 0,
            current_step: 0
        };
        
        res.json(onboardingFlow);
        
    } catch (error) {
        console.error('Onboarding flow error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Database endpoints
app.post('/api/companies', async (req, res) => {
    try {
        console.log('POST /api/companies - Request body:', JSON.stringify(req.body, null, 2));
        
        // Use service role client to bypass RLS for bulk imports
        const { data, error } = await supabaseServiceClient
            .from('companies')
            .insert([req.body])
            .select();
        
        if (error) {
            console.error('Supabase error in POST /api/companies:', error);
            throw error;
        }
        
        console.log('POST /api/companies - Success:', data[0]);
        res.json(data[0]);
    } catch (error) {
        console.error('POST /api/companies - Error:', error);
        console.error('Error details:', {
            message: error.message,
            code: error.code,
            details: error.details,
            hint: error.hint
        });
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/companies', async (req, res) => {
    try {
        const { data, error } = await supabaseClient
            .from('companies')
            .select('*');
        
        if (error) throw error;
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/companies/:id', async (req, res) => {
    try {
        const { data, error } = await supabaseClient
            .from('companies')
            .select('*')
            .eq('id', req.params.id)
            .single();
        
        if (error) throw error;
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Chat endpoints
app.post('/api/chat', async (req, res) => {
    try {
        const { message, company_data } = req.body;
        
        // Call conversational AI agent
        const chatResult = await runPythonScript('conversational_b2b_agent.py', {
            action: 'process_message',
            message: message,
            company_data: company_data
        });
        
        res.json(chatResult);
    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Plugin endpoints
app.get('/api/plugins', async (req, res) => {
    try {
        const { data, error } = await supabaseClient
            .from('plugins')
            .select('*');
        
        if (error) throw error;
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/plugins', async (req, res) => {
    try {
        const { data, error } = await supabaseClient
            .from('plugins')
            .insert([req.body])
            .select();
        
        if (error) throw error;
        res.json(data[0]);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Analytics endpoints
app.post('/api/analytics', async (req, res) => {
    try {
        const { company_data, analysis_type } = req.body;
        
        // Call advanced analytics engine
        const analyticsResult = await runPythonScript('advanced_analytics_engine.py', {
            action: analysis_type,
            company_data: company_data
        });
        
        res.json(analyticsResult);
    } catch (error) {
        console.error('Analytics error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Multi-hop symbiosis endpoints
app.post('/api/symbiosis-network', async (req, res) => {
    try {
        const { company_data, network_depth } = req.body;
        
        // Call multi-hop symbiosis engine
        const networkResult = await runPythonScript('multi_hop_symbiosis_network.py', {
            action: 'analyze_network',
            company_data: company_data,
            network_depth: network_depth || 3
        });
        
        res.json(networkResult);
    } catch (error) {
        console.error('Symbiosis network error:', error);
        res.status(500).json({ error: error.message });
    }
});

// COMPREHENSIVE MATCH ANALYSIS ENDPOINT
app.post('/api/comprehensive-match-analysis', async (req, res) => {
    try {
        const { buyer_data, seller_data, match_data } = req.body;
        
        // Use AI evolution engine for comprehensive analysis
        const analysisResult = await aiEvolutionEngine.executeAIAnalysis('matchGeneration', {
          company_a: buyer_data,
          company_b: seller_data,
          material_a: match_data.material_a,
          material_b: match_data.material_b
        });
        
        res.json(analysisResult);
    } catch (error) {
        console.error('Comprehensive match analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

// REFINEMENT ANALYSIS ENDPOINT
app.post('/api/refinement-analysis', async (req, res) => {
    try {
        const { material_data, buyer_data } = req.body;
        
        const refinementResult = await runPythonScript('refinement_analysis_engine.py', {
            action: 'assess_material_readiness',
            material_data: material_data,
            buyer_data: buyer_data
        });
        
        res.json(refinementResult);
    } catch (error) {
        console.error('Refinement analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

// FINANCIAL ANALYSIS ENDPOINT
app.post('/api/financial-analysis', async (req, res) => {
    try {
        const { match_data, buyer_data, seller_data, logistics_data, refinement_data } = req.body;
        
        const financialResult = await runPythonScript('financial_analysis_engine.py', {
            action: 'analyze_match_financials',
            match_data: match_data,
            buyer_data: buyer_data,
            seller_data: seller_data,
            logistics_data: logistics_data,
            refinement_data: refinement_data
        });
        
        res.json(financialResult);
    } catch (error) {
        console.error('Financial analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

// ENHANCED MATCHING WITH COMPREHENSIVE ANALYSIS
app.post('/api/enhanced-matching', async (req, res) => {
    try {
        const { buyer_data, seller_data, preferences } = req.body;
        
        // Get initial matches
        const initialMatches = await runPythonScript('real_ai_matching_engine.py', {
            action: 'find_symbiotic_matches',
            buyer_data: buyer_data,
            seller_data: seller_data,
            preferences: preferences,
            top_k: 10
        });
        
        // Perform comprehensive analysis on each match
        const enhancedMatches = [];
        
        for (const match of initialMatches) {
            try {
                const comprehensiveAnalysis = await runPythonScript('comprehensive_match_analyzer.py', {
                    action: 'analyze_match_comprehensive',
                    buyer_data: buyer_data,
                    seller_data: seller_data,
                    match_data: match
                });
                
                enhancedMatches.push({
                    ...match,
                    comprehensive_analysis: comprehensiveAnalysis
                });
            } catch (error) {
                console.error(`Error analyzing match ${match.id}:`, error);
                enhancedMatches.push({
                    ...match,
                    comprehensive_analysis: { error: 'Analysis failed' }
                });
            }
        }
        
        // Sort by overall score
        enhancedMatches.sort((a, b) => {
            const scoreA = a.comprehensive_analysis?.overall_score || 0;
            const scoreB = b.comprehensive_analysis?.overall_score || 0;
            return scoreB - scoreA;
        });
        
        res.json({
            matches: enhancedMatches,
            total_matches: enhancedMatches.length,
            analysis_summary: {
                excellent_matches: enhancedMatches.filter(m => m.comprehensive_analysis?.match_quality === 'Excellent').length,
                good_matches: enhancedMatches.filter(m => m.comprehensive_analysis?.match_quality === 'Good').length,
                moderate_matches: enhancedMatches.filter(m => m.comprehensive_analysis?.match_quality === 'Moderate').length,
                average_overall_score: enhancedMatches.reduce((sum, m) => sum + (m.comprehensive_analysis?.overall_score || 0), 0) / enhancedMatches.length
            }
        });
        
    } catch (error) {
        console.error('Enhanced matching error:', error);
        res.status(500).json({ error: error.message });
    }
});

// DETAILED COST BREAKDOWN ENDPOINT
app.post('/api/cost-breakdown', async (req, res) => {
    try {
        const { buyer_data, seller_data, match_data } = req.body;
        
        // Get logistics analysis
        const logisticsResult = await runPythonScript('logistics_cost_engine.py', {
            action: 'get_route_planning',
            buyer_data: buyer_data,
            seller_data: seller_data,
            match_data: match_data
        });
        
        // Get financial analysis
        const financialResult = await runPythonScript('financial_analysis_engine.py', {
            action: 'analyze_match_financials',
            match_data: match_data,
            buyer_data: buyer_data,
            seller_data: seller_data,
            logistics_data: logisticsResult
        });
        
        // Get refinement analysis
        const refinementResult = await runPythonScript('refinement_analysis_engine.py', {
            action: 'assess_material_readiness',
            material_data: match_data,
            buyer_data: buyer_data
        });
        
        res.json({
            logistics_breakdown: logisticsResult,
            financial_breakdown: financialResult,
            refinement_breakdown: refinementResult,
            summary: {
                total_cost: financialResult?.scenario_comparison?.waste_scenario?.total_cost || 0,
                net_savings: financialResult?.scenario_comparison?.net_savings || 0,
                payback_period: financialResult?.scenario_comparison?.payback_period_months || 0,
                roi_percentage: financialResult?.scenario_comparison?.roi_percentage || 0
            }
        });
        
    } catch (error) {
        console.error('Cost breakdown error:', error);
        res.status(500).json({ error: error.message });
    }
});

// CARBON AND ENVIRONMENTAL ANALYSIS ENDPOINT
app.post('/api/environmental-analysis', async (req, res) => {
    try {
        const { buyer_data, seller_data, match_data } = req.body;
        
        // Get carbon analysis
        const carbonResult = await runPythonScript('carbon_calculation_engine.py', {
            action: 'calculate_carbon_footprint',
            buyer_data: buyer_data,
            seller_data: seller_data,
            match_data: match_data
        });
        
        // Get waste analysis
        const wasteResult = await runPythonScript('waste_tracking_engine.py', {
            action: 'calculate_waste_profile',
            buyer_data: buyer_data,
            seller_data: seller_data,
            match_data: match_data
        });
        
        // Get logistics carbon impact
        const logisticsResult = await runPythonScript('logistics_cost_engine.py', {
            action: 'calculate_carbon_impact',
            buyer_data: buyer_data,
            seller_data: seller_data,
            match_data: match_data
        });
        
        res.json({
            carbon_analysis: carbonResult,
            waste_analysis: wasteResult,
            logistics_carbon: logisticsResult,
            environmental_summary: {
                total_carbon_savings: (carbonResult?.net_carbon_savings_kg || 0) + (logisticsResult?.carbon_savings_vs_truck || 0),
                waste_reduction_potential: wasteResult?.waste_reduction_potential || 0,
                recycling_potential: wasteResult?.recycling_potential || 0,
                environmental_score: calculateEnvironmentalScore(carbonResult, wasteResult, logisticsResult)
            }
        });
        
    } catch (error) {
        console.error('Environmental analysis error:', error);
        res.status(500).json({ error: error.message });
    }
});

// EQUIPMENT RECOMMENDATIONS ENDPOINT
app.post('/api/equipment-recommendations', async (req, res) => {
    try {
        const { material_data, buyer_data } = req.body;
        
        const refinementResult = await runPythonScript('refinement_analysis_engine.py', {
            action: 'assess_material_readiness',
            material_data: material_data,
            buyer_data: buyer_data
        });
        
        const equipmentRecommendations = refinementResult?.equipment_recommendations || [];
        
        // Add ROI analysis for each equipment
        const enhancedRecommendations = equipmentRecommendations.map(equipment => ({
            ...equipment,
            roi_analysis: {
                payback_years: equipment.roi_analysis?.payback_years || 0,
                roi_percentage: equipment.roi_analysis?.roi_percentage || 0,
                annual_savings: equipment.roi_analysis?.annual_savings || 0,
                break_even_months: equipment.roi_analysis?.break_even_months || 0
            }
        }));
        
        res.json({
            equipment_recommendations: enhancedRecommendations,
            total_equipment_cost: enhancedRecommendations.reduce((sum, eq) => sum + eq.purchase_cost, 0),
            average_payback_period: enhancedRecommendations.reduce((sum, eq) => sum + eq.roi_analysis.payback_years, 0) / enhancedRecommendations.length,
            recommendations: refinementResult?.recommendations || []
        });
        
    } catch (error) {
        console.error('Equipment recommendations error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Helper function to calculate environmental score
function calculateEnvironmentalScore(carbonResult, wasteResult, logisticsResult) {
    let score = 0;
    
    if (carbonResult?.net_carbon_savings_kg > 0) {
        score += 0.4;
    }
    
    if (wasteResult?.waste_reduction_potential > 0) {
        score += 0.3;
    }
    
    if (logisticsResult?.carbon_savings_vs_truck > 0) {
        score += 0.3;
    }
    
    return Math.min(1.0, score);
}

// Company data endpoint
app.get('/api/companies/current', (req, res) => {
  // Mock company data - in a real app, this would come from the database
  const mockCompanyData = {
    id: '1',
    name: 'EcoTech Solutions',
    industry: 'Manufacturing',
    location: 'San Francisco, CA',
    employee_count: 250,
    annual_revenue: 15000000,
    sustainability_score: 85,
    matches_count: 12,
    savings_achieved: 450000,
    carbon_reduced: 1250
  };
  
  res.json(mockCompanyData);
});

// AI insights endpoint
app.get('/api/ai-insights', async (req, res) => {
  try {
    const { company_id } = req.query;
    
    if (!company_id) {
      // Return mock data if no company_id provided
      const mockInsights = {
        insights: [
          {
            id: '1',
            type: 'opportunity',
            title: 'New Waste Stream Partnership',
            description: 'Local brewery can use your organic waste for biogas production',
            impact: 'high',
            estimated_savings: 75000,
            carbon_reduction: 200,
            action_required: true,
            priority: 'high'
          },
          {
            id: '2',
            type: 'savings',
            title: 'Energy Efficiency Upgrade',
            description: 'Switch to LED lighting could save $25,000 annually',
            impact: 'medium',
            estimated_savings: 25000,
            carbon_reduction: 50,
            action_required: true,
            priority: 'medium'
          },
          {
            id: '3',
            type: 'match',
            title: 'Material Exchange Opportunity',
            description: 'Construction company needs your excess steel scrap',
            impact: 'high',
            estimated_savings: 120000,
            carbon_reduction: 300,
            action_required: true,
            priority: 'urgent'
          }
        ]
      };
      
      return res.json(mockInsights);
    }

    // Query database for real insights
    const { data: insights, error } = await supabase
      .from('ai_insights')
      .select('*')
      .eq('company_id', company_id)
      .order('created_at', { ascending: false })
      .limit(10);

    if (error) {
      console.error('Error fetching AI insights:', error);
      return res.status(500).json({ error: 'Failed to fetch AI insights' });
    }

    // Transform database insights to frontend format
    const transformedInsights = insights.map(insight => ({
      id: insight.id,
      type: insight.insight_type || 'opportunity',
      title: insight.title || 'AI Insight',
      description: insight.description || 'AI-generated insight for your business',
      impact: insight.impact || 'medium',
      estimated_savings: insight.estimated_savings ? 
        parseInt(insight.estimated_savings.replace(/[^0-9]/g, '')) : 0,
      carbon_reduction: insight.carbon_reduction ? 
        parseInt(insight.carbon_reduction.replace(/[^0-9]/g, '')) : 0,
      action_required: insight.action_required || false,
      priority: insight.confidence > 80 ? 'high' : insight.confidence > 60 ? 'medium' : 'low'
    }));

    res.json({ insights: transformedInsights });
  } catch (error) {
    console.error('AI insights error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Portfolio data endpoint
app.get('/api/portfolio', (req, res) => {
  const mockPortfolio = {
    portfolio: {
      company_overview: {
        size_category: 'Medium Enterprise',
        industry_position: 'Innovation Leader',
        sustainability_rating: 'A+',
        growth_potential: 'High'
      },
      achievements: {
        total_savings: 450000,
        carbon_reduced: 1250,
        partnerships_formed: 8,
        waste_diverted: 850
      },
      recommendations: [
        {
          category: 'Waste Management',
          suggestions: [
            'Implement zero-waste program',
            'Partner with local recycling facilities',
            'Optimize packaging materials'
          ],
          priority: 'high'
        },
        {
          category: 'Energy Efficiency',
          suggestions: [
            'Install solar panels',
            'Upgrade HVAC systems',
            'Implement smart building controls'
          ],
          priority: 'medium'
        },
        {
          category: 'Supply Chain',
          suggestions: [
            'Source local suppliers',
            'Use recycled materials',
            'Implement circular procurement'
          ],
          priority: 'high'
        }
      ],
      recent_activity: [
        {
          date: '2024-01-15',
          action: 'Completed waste audit',
          impact: 'Identified 30% reduction opportunity'
        },
        {
          date: '2024-01-10',
          action: 'Formed partnership with GreenTech',
          impact: 'Annual savings of $75,000'
        },
        {
          date: '2024-01-05',
          action: 'Implemented energy monitoring',
          impact: '15% reduction in energy costs'
        }
      ]
    }
  };
  
  res.json(mockPortfolio);
});

// AI Onboarding API endpoints
app.post('/api/ai-onboarding/initial-questions', async (req, res) => {
  try {
    const { companyProfile } = req.body;
    
    // Use DeepSeek Reasoner API to generate initial intelligent questions
    const prompt = `
    You are an AI assistant for an Industrial Symbiosis Management platform. 
    Generate 3-4 initial intelligent questions to start the onboarding process for a new company.
    
    Company Profile (if available):
    - Name: ${companyProfile?.name || 'Not provided'}
    - Industry: ${companyProfile?.industry || 'Not provided'}
    - Location: ${companyProfile?.location || 'Not provided'}
    - Employee Count: ${companyProfile?.employee_count || 'Not provided'}
    
    Generate initial questions that:
    1. Are welcoming and easy to answer
    2. Help establish basic company profile
    3. Identify industry and scale
    4. Understand current sustainability practices
    5. Set the foundation for deeper symbiosis analysis
    
    Return the questions in this JSON format:
    {
      "questions": [
        {
          "title": "Step title",
          "description": "Step description",
          "fields": [
            {
              "id": "unique_id",
              "type": "text|select|textarea|multiselect",
              "label": "Question text",
              "placeholder": "Optional placeholder",
              "options": ["option1", "option2"], // for select/multiselect
              "required": true/false,
              "reasoning": "Why this question is important"
            }
          ]
        }
      ]
    }
    `;

    const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.DEEPSEEK_API_KEY}`
      },
      body: JSON.stringify({
        model: 'deepseek-reasoner',
        messages: [
          {
            role: 'system',
            content: 'You are an expert in industrial symbiosis and circular economy. Generate intelligent, welcoming initial questions to help companies start their symbiosis journey.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 1000,
        temperature: 0.7,
        response_format: { type: 'json_object' }
      })
    });

    if (!response.ok) {
      throw new Error('Failed to generate initial AI questions');
    }

    const data = await response.json();
    const aiResponse = data.choices[0].message.content;
    
    // Parse the AI response to extract questions
    let questions = [];
    try {
      const parsed = JSON.parse(aiResponse);
      questions = parsed.questions || [];
    } catch (e) {
      // Fallback questions if AI response parsing fails
      questions = [
        {
          title: "Welcome to SymbioFlows",
          description: "Let's start by understanding your company's basic profile to identify the best symbiosis opportunities.",
          fields: [
            {
              id: 'company_name',
              type: 'text',
              label: 'What is your company name?',
              placeholder: 'Enter your company name',
              required: true,
              reasoning: 'Your company name helps us personalize your experience and track your progress.'
            },
            {
              id: 'industry',
              type: 'select',
              label: 'What industry are you in?',
              options: [
                'Manufacturing',
                'Food & Beverage',
                'Textiles & Apparel',
                'Chemicals',
                'Metals & Mining',
                'Construction',
                'Electronics',
                'Automotive',
                'Pharmaceuticals',
                'Energy',
                'Agriculture',
                'Other'
              ],
              required: true,
              reasoning: 'Your industry determines the types of waste streams and resource needs we should focus on.'
            },
            {
              id: 'employee_count',
              type: 'select',
              label: 'How many employees does your company have?',
              options: [
                '1-10',
                '11-50',
                '51-200',
                '201-1000',
                '1000+'
              ],
              required: true,
              reasoning: 'Company size helps us understand your scale and potential impact of symbiosis opportunities.'
            }
          ]
        },
        {
          title: "Current Sustainability Practices",
          description: "Understanding your current practices helps us identify immediate opportunities for improvement.",
          fields: [
            {
              id: 'sustainability_goals',
              type: 'multiselect',
              label: 'What are your sustainability goals?',
              options: [
                'Reduce waste',
                'Lower costs',
                'Improve efficiency',
                'Reduce carbon footprint',
                'Meet regulatory requirements',
                'Enhance brand reputation',
                'Access new markets',
                'Innovation leadership'
              ],
              reasoning: 'Your sustainability goals help us prioritize the most relevant symbiosis opportunities.'
            },
            {
              id: 'current_waste_management',
              type: 'select',
              label: 'How do you currently manage waste?',
              options: [
                'Landfill disposal',
                'Recycling programs',
                'Waste-to-energy',
                'Composting',
                'Partner with waste management companies',
                'Other'
              ],
              reasoning: 'Current waste management practices help us identify potential cost savings and new revenue streams.'
            }
          ]
        }
      ];
    }

    res.json({ questions });
  } catch (error) {
    console.error('Error generating initial AI questions:', error);
    res.status(500).json({ error: 'Failed to generate initial AI questions' });
  }
});

app.post('/api/ai-onboarding/questions', async (req, res) => {
  try {
    const { companyProfile, currentStep, existingData } = req.body;
    
    // Use DeepSeek Reasoner API to generate intelligent questions
    const prompt = `
    You are an AI assistant for an Industrial Symbiosis Management platform. 
    Based on the company profile below, generate 2-4 intelligent questions that will help identify the best symbiosis opportunities.
    
    Company Profile:
    - Name: ${companyProfile.name}
    - Industry: ${companyProfile.industry}
    - Location: ${companyProfile.location}
    - Employee Count: ${companyProfile.employee_count}
    - Products: ${companyProfile.products}
    - Main Materials: ${companyProfile.main_materials}
    - Production Volume: ${companyProfile.production_volume}
    
    Generate questions that:
    1. Are relevant to their specific industry and materials
    2. Help identify waste streams and resource needs
    3. Understand sustainability goals
    4. Are concise and easy to answer
    5. Will lead to actionable symbiosis opportunities
    
    Return the questions in this JSON format:
    {
      "questions": [
        {
          "id": "unique_id",
          "type": "text|select|textarea|multiselect",
          "label": "Question text",
          "placeholder": "Optional placeholder",
          "options": ["option1", "option2"], // for select/multiselect
          "required": true/false,
          "reasoning": "Why this question is important"
        }
      ]
    }
    `;

    const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.DEEPSEEK_API_KEY}`
      },
      body: JSON.stringify({
        model: 'deepseek-reasoner',
        messages: [
          {
            role: 'system',
            content: 'You are an expert in industrial symbiosis and circular economy. Generate intelligent, relevant questions to help companies find symbiosis opportunities.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 1000,
        temperature: 0.7,
        response_format: { type: 'json_object' }
      })
    });

    if (!response.ok) {
      throw new Error('Failed to generate AI questions');
    }

    const data = await response.json();
    const aiResponse = data.choices[0].message.content;
    
    // Parse the AI response to extract questions
    let questions = [];
    try {
      const parsed = JSON.parse(aiResponse);
      questions = parsed.questions || [];
    } catch (e) {
      // Fallback questions if AI response parsing fails
      questions = [
        {
          id: 'waste_streams',
          type: 'multiselect',
          label: 'What waste streams do you generate?',
          options: [
            'Organic waste',
            'Plastic waste',
            'Metal scrap',
            'Chemical waste',
            'Electronic waste',
            'Textile waste',
            'Food waste',
            'Construction waste',
            'Paper waste',
            'Glass waste'
          ],
          required: true,
          reasoning: 'Based on your industry and materials, these are the most likely waste streams for symbiosis opportunities.'
        },
        {
          id: 'resource_needs',
          type: 'multiselect',
          label: 'What resources do you need?',
          options: [
            'Raw materials',
            'Energy',
            'Water',
            'Chemicals',
            'Packaging materials',
            'Transportation',
            'Waste disposal',
            'Recycled materials',
            'Bio-based materials',
            'Renewable energy'
          ],
          required: true,
          reasoning: 'Identifying your resource needs helps find companies that can provide them as waste streams.'
        }
      ];
    }

    res.json({ questions });
  } catch (error) {
    console.error('Error generating AI questions:', error);
    res.status(500).json({ error: 'Failed to generate AI questions' });
  }
});

app.post('/api/ai-onboarding/complete', async (req, res) => {
  try {
    const { companyProfile, onboardingData } = req.body;
    
    // Get the authenticated user's email from the session/token
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    if (authError || !user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    // Ensure email is included in the company profile
    const enrichedCompanyProfile = {
      ...companyProfile,
      email: user.email, // Always include the authenticated user's email
      onboarding_completed: true,
      updated_at: new Date().toISOString()
    };

    // Save or update the company profile in the database
    const { data: savedCompany, error: saveError } = await supabase
      .from('companies')
      .upsert([{
        id: user.id, // Use the user's ID as the company ID
        ...enrichedCompanyProfile
      }], {
        onConflict: 'id',
        ignoreDuplicates: false
      })
      .select()
      .single();

    if (saveError) {
      console.error('Error saving company profile:', saveError);
      return res.status(500).json({ error: 'Failed to save company profile' });
    }

    // Use DeepSeek Reasoner API to analyze the complete profile and create recommendations
    const prompt = `
    You are an AI assistant for an Industrial Symbiosis Management platform.
    Analyze the following company profile and create a comprehensive symbiosis analysis:
    
    Company Profile:
    ${JSON.stringify(enrichedCompanyProfile, null, 2)}
    
    Onboarding Data:
    ${JSON.stringify(onboardingData, null, 2)}
    
    Provide a detailed analysis including:
    1. Key waste streams and their potential value
    2. Resource needs that could be met through symbiosis
    3. Recommended symbiosis opportunities
    4. Potential partners to seek
    5. Estimated cost savings and environmental impact
    6. Implementation roadmap
    7. Material exchange opportunities
    8. Specific company suggestions for partnerships
    
    Return the analysis in this JSON format:
    {
      "analysis": {
        "waste_streams": [
          {
            "name": "waste stream name",
            "quantity": "estimated quantity",
            "value": "potential value",
            "potential_uses": ["use 1", "use 2"]
          }
        ],
        "resource_needs": [
          {
            "name": "resource name",
            "current_cost": "estimated cost",
            "potential_sources": ["source 1", "source 2"]
          }
        ],
        "opportunities": [
          {
            "title": "opportunity title",
            "description": "detailed description",
            "estimated_savings": "amount",
            "carbon_reduction": "tons CO2",
            "implementation_time": "timeframe",
            "difficulty": "easy/medium/hard"
          }
        ],
        "potential_partners": [
          {
            "company_type": "type of company",
            "location": "preferred location",
            "waste_they_can_use": ["waste 1", "waste 2"],
            "resources_they_can_provide": ["resource 1", "resource 2"],
            "estimated_partnership_value": "value"
          }
        ],
        "material_listings": [
          {
            "material": "material name",
            "current_status": "waste/resource",
            "quantity": "available/needed",
            "value": "estimated value",
            "potential_exchanges": ["exchange 1", "exchange 2"]
          }
        ],
        "ai_insights": {
          "symbiosis_score": "Percentage indicating potential for symbiosis (0-100)",
          "estimated_savings": "Estimated annual cost savings from symbiosis",
          "carbon_reduction": "Estimated CO2 reduction in tons/year",
          "top_opportunities": ["Opportunity 1", "Opportunity 2", "Opportunity 3"],
          "recommended_partners": ["Partner type 1", "Partner type 2", "Partner type 3"],
          "implementation_roadmap": ["Step 1", "Step 2", "Step 3"]
        }
      }
    }
    `;

    const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.DEEPSEEK_API_KEY}`
      },
      body: JSON.stringify({
        model: 'deepseek-reasoner',
        messages: [
          {
            role: 'system',
            content: 'You are an expert in industrial symbiosis and circular economy. Analyze company data and generate detailed, actionable recommendations and listings.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 2000,
        temperature: 0.7,
        response_format: { type: 'json_object' }
      })
    });

    if (!response.ok) {
      throw new Error('Failed to complete onboarding analysis');
    }

    const data = await response.json();
    const aiResponse = data.choices[0].message.content;
    
    // Parse the AI response
    let analysis = {};
    try {
      const parsed = JSON.parse(aiResponse);
      analysis = parsed;
    } catch (e) {
      // Fallback analysis if parsing fails
      analysis = {
        analysis: {
          waste_streams: ['Based on industry analysis'],
          resource_needs: ['Identified from profile'],
          opportunities: ['Multiple symbiosis pathways available'],
          potential_partners: ['Local companies in similar industries'],
          estimated_savings: '15-30% reduction in waste disposal costs',
          environmental_impact: 'Significant reduction in carbon footprint',
          roadmap: ['Immediate opportunities', 'Medium-term partnerships', 'Long-term circular economy goals']
        },
        recommendations: ['Start with waste stream analysis', 'Identify local partners', 'Implement pilot projects'],
        next_steps: ['Complete marketplace profile', 'Connect with potential partners', 'Schedule consultation']
      };
    }

    // Store the analysis in the database
    if (analysis.analysis) {
      const { error: analysisError } = await supabase
        .from('ai_insights')
        .insert({
          company_id: user.id,
          symbiosis_score: analysis.analysis.ai_insights?.symbiosis_score || '75%',
          estimated_savings: analysis.analysis.ai_insights?.estimated_savings || '$25K-50K annually',
          carbon_reduction: analysis.analysis.ai_insights?.carbon_reduction || '10-20 tons CO2',
          top_opportunities: analysis.analysis.ai_insights?.top_opportunities || [],
          recommended_partners: analysis.analysis.ai_insights?.recommended_partners || [],
          implementation_roadmap: analysis.analysis.ai_insights?.implementation_roadmap || []
        });

      if (analysisError) {
        console.error('Error saving AI insights:', analysisError);
      }
    }

    console.log('Onboarding completed successfully for company:', savedCompany.name);
    console.log('Analysis stored:', analysis);

    res.json({ 
      success: true, 
      analysis,
      companyProfile: enrichedCompanyProfile,
      message: 'Onboarding completed successfully. Your profile has been created and analyzed for symbiosis opportunities.'
    });
  } catch (error) {
    console.error('Error completing onboarding:', error);
    res.status(500).json({ error: 'Failed to complete onboarding' });
  }
});

// AI Listings Orchestration Endpoint
app.post('/api/v1/companies/:id/generate-listings', async (req, res) => {
  try {
    const { id: company_id } = req.params;
    
    // Fetch complete company profile
    const { data: company, error: fetchError } = await supabase
      .from('companies')
      .select('*')
      .eq('id', company_id)
      .single();

    if (fetchError || !company) {
      return res.status(404).json({ error: 'Company not found' });
    }

    // Call Python inference service
    const path = require('path');
    
    const options = {
      mode: 'text', // Use text mode to manually parse JSON
      pythonPath: 'python',
      pythonOptions: ['-u'],
      scriptPath: path.join(__dirname),
      args: [JSON.stringify(company)]
    };

    const rawResults = await new Promise((resolve, reject) => {
      PythonShell.run('listing_inference_service.py', options, (err, results) => {
        if (err) reject(err);
        else resolve(results && results.length > 0 ? results[0] : '');
      });
    });

    let results;
    try {
      results = JSON.parse(rawResults);
    } catch (parseErr) {
      console.error('Failed to parse Python output as JSON:', rawResults);
      throw new Error('AI inference service returned invalid JSON.');
    }

    // Insert materials (outputs)
    const materialsToInsert = results.predicted_outputs.map(item => ({
      company_id,
      name: item.name,
      description: item.description,
      category: item.category,
      quantity_estimate: item.quantity_estimate,
      potential_value: item.potential_value,
      quality_grade: item.quality_grade,
      potential_uses: item.potential_uses,
      symbiosis_opportunities: item.symbiosis_opportunities,
      embeddings: item.embeddings ? JSON.stringify(item.embeddings) : null,
      ai_generated: true,
      created_at: new Date().toISOString()
    }));

    const { error: materialsError } = await supabase
      .from('materials')
      .insert(materialsToInsert);

    if (materialsError) {
      console.error('Materials insertion error:', materialsError);
    }

    // Insert requirements (inputs)
    const requirementsToInsert = results.predicted_inputs.map(item => ({
      company_id,
      name: item.name,
      description: item.description,
      category: item.category,
      quantity_needed: item.quantity_needed,
      current_cost: item.current_cost,
      priority: item.priority,
      potential_sources: item.potential_sources,
      symbiosis_opportunities: item.symbiosis_opportunities,
      embeddings: item.embeddings ? JSON.stringify(item.embeddings) : null,
      ai_generated: true,
      created_at: new Date().toISOString()
    }));

    const { error: requirementsError } = await supabase
      .from('requirements')
      .insert(requirementsToInsert);

    if (requirementsError) {
      console.error('Requirements insertion error:', requirementsError);
    }

    // Store AI insights
    if (results.ai_insights) {
      const { error: insightsError } = await supabase
        .from('ai_insights')
        .upsert({
          company_id,
          symbiosis_score: results.ai_insights.symbiosis_score,
          estimated_savings: results.ai_insights.estimated_savings,
          carbon_reduction: results.ai_insights.carbon_reduction,
          top_opportunities: results.ai_insights.top_opportunities,
          recommended_partners: results.ai_insights.recommended_partners,
          implementation_roadmap: results.ai_insights.implementation_roadmap,
          created_at: new Date().toISOString()
        });

      if (insightsError) {
        console.error('AI insights insertion error:', insightsError);
      }
    }

    // Trigger GNN match engine
    try {
      await supabase.functions.invoke('gnn-match', {
        body: { company_id, new_listings: true }
      });
    } catch (gnnError) {
      console.error('GNN trigger error:', gnnError);
    }

    res.json({
      success: true,
      materials_count: materialsToInsert.length,
      requirements_count: requirementsToInsert.length,
      ai_insights: results.ai_insights
    });

  } catch (error) {
    console.error('AI Listings Orchestration Error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Real-time notifications endpoint (SSE)
app.get('/api/v1/companies/:id/notifications/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  const { id: company_id } = req.params;

  // Initial fetch
  let lastNotificationId = null;
  const sendNotifications = async () => {
    const { data: notifications } = await supabase
      .from('notifications')
      .select('*')
      .eq('company_id', company_id)
      .order('created_at', { ascending: false })
      .limit(10);
    if (notifications && notifications.length > 0) {
      const newNotifications = lastNotificationId
        ? notifications.filter(n => n.id !== lastNotificationId)
        : notifications;
      newNotifications.forEach(n => {
        res.write(`data: ${JSON.stringify(n)}\n\n`);
      });
      lastNotificationId = notifications[0].id;
    }
  };

  // Poll every 3 seconds (simulate real-time)
  const interval = setInterval(sendNotifications, 3000);
  req.on('close', () => {
    clearInterval(interval);
    res.end();
  });
  // Send initial batch
  sendNotifications();
});

// Adaptive AI Onboarding endpoints - HTTP requests to Python Flask server
app.post('/api/adaptive-onboarding/start', async (req, res) => {
  try {
    const { initial_profile } = req.body;
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return sendResponse(res, { success: false, error: 'Authentication required' }, 401);
    }
    const token = authHeader.split(' ')[1];
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    if (authError || !user) {
      return sendResponse(res, { success: false, error: 'Authentication required' }, 401);
    }
    // Use requestWithRetry for Python Flask call
    const response = await requestWithRetry({
      method: 'POST',
      url: process.env.ADAPTIVE_ONBOARDING_URL || 'http://localhost:5003/api/adaptive-onboarding/start',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': authHeader
      },
      data: {
        user_id: user.id,
        initial_profile: initial_profile || {}
      },
      timeout: 10000
    });
    const onboardingResult = response.data;
    if (onboardingResult.success && onboardingResult.session) {
      return sendResponse(res, {
        success: true,
        data: {
          session_id: onboardingResult.session.session_id,
          initial_questions: onboardingResult.session.initial_questions,
          completion_percentage: onboardingResult.session.completion_percentage
        }
      });
    } else {
      return sendResponse(res, { success: false, error: onboardingResult.error || 'Onboarding failed' }, 500);
    }
  } catch (error) {
    console.error('Error starting adaptive onboarding:', error);
    return sendResponse(res, { success: false, error: 'Failed to start adaptive onboarding' }, 500);
  }
});

app.post('/api/adaptive-onboarding/respond', async (req, res) => {
  try {
    const { session_id, question_id, answer } = req.body;
    
    // Get the authenticated user from the Authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const token = authHeader.split(' ')[1];
    
    // Verify the token with Supabase
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    if (authError || !user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    // Make HTTP request to Python Flask server
    const response = await fetch('http://localhost:5003/api/adaptive-onboarding/respond', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        session_id,
        question_id,
        answer
      })
    });

    if (!response.ok) {
      throw new Error(`Python server responded with ${response.status}`);
    }

    const responseResult = await response.json();
    res.json(responseResult);
    
  } catch (error) {
    console.error('Error processing user response:', error);
    res.status(500).json({ error: 'Failed to process user response' });
  }
});

app.post('/api/adaptive-onboarding/complete', async (req, res) => {
  try {
    const { session_id } = req.body;
    
    // Get the authenticated user from the Authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const token = authHeader.split(' ')[1];
    
    // Verify the token with Supabase
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    if (authError || !user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    // Make HTTP request to Python Flask server
    const response = await fetch('http://localhost:5003/api/adaptive-onboarding/complete', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        session_id
      })
    });

    if (!response.ok) {
      throw new Error(`Python server responded with ${response.status}`);
    }

    const result = await response.json();
    
    // Save company profile to database if available
    if (result.success && result.company_profile) {
      const enrichedCompanyProfile = {
        ...result.company_profile,
        email: user.email,
        onboarding_completed: true,
        updated_at: new Date().toISOString()
      };

      const { data: savedCompany, error: saveError } = await supabase
        .from('companies')
        .upsert([{
          id: user.id,
          ...enrichedCompanyProfile
        }], {
          onConflict: 'id',
          ignoreDuplicates: false
        })
        .select()
        .single();

      if (saveError) {
        console.error('Error saving company profile:', saveError);
      }
    }
    
    res.json(result);
    
  } catch (error) {
    console.error('Error completing adaptive onboarding:', error);
    res.status(500).json({ error: 'Failed to complete adaptive onboarding' });
  }
});

app.get('/api/adaptive-onboarding/status/:session_id', async (req, res) => {
  try {
    const { session_id } = req.params;
    
    // Get the authenticated user from the Authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const token = authHeader.split(' ')[1];
    
    // Verify the token with Supabase
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    if (authError || !user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    // Make HTTP request to Python Flask server
    const response = await fetch(`http://localhost:5003/api/adaptive-onboarding/status/${session_id}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Python server responded with ${response.status}`);
    }

    const statusResult = await response.json();
    res.json(statusResult);
    
  } catch (error) {
    console.error('Error getting session status:', error);
    res.status(500).json({ error: 'Failed to get session status' });
  }
});

app.get('/api/adaptive-onboarding/questions/:session_id', async (req, res) => {
  try {
    const { session_id } = req.params;
    
    // Get the authenticated user from the Authorization header
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const token = authHeader.split(' ')[1];
    
    // Verify the token with Supabase
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    if (authError || !user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    // Make HTTP request to Python Flask server
    const response = await fetch(`http://localhost:5003/api/adaptive-onboarding/questions/${session_id}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Python server responded with ${response.status}`);
    }

    const questionsResult = await response.json();
    res.json(questionsResult);
  } catch (error) {
    console.error('Error getting session questions:', error);
    res.status(500).json({ error: 'Failed to get session questions' });
  }
});

// Export the app for testing
module.exports = app;

// Start the server only if not in test environment
if (process.env.NODE_ENV !== 'test') {
    app.listen(PORT, () => {
        console.log(`🚀 Server running on port ${PORT}`);
        console.log(`📡 API available at http://localhost:${PORT}/api`);
        console.log(`🏥 Health check at http://localhost:${PORT}/api/health`);
    });
}

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    // Check database connectivity
    const { data, error } = await supabase
      .from('users')
      .select('count')
      .limit(1);
    
    if (error) {
      return res.status(503).json({
        status: 'unhealthy',
        database: 'disconnected',
        timestamp: new Date().toISOString(),
        error: error.message
      });
    }
    
    res.json({
      status: 'healthy',
      database: 'connected',
      timestamp: new Date().toISOString(),
      uptime: process.uptime()
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      database: 'error',
      timestamp: new Date().toISOString(),
      error: error.message
    });
  }
});

// AI Portfolio Generation Endpoint (Legacy)
app.post('/api/generate-ai-portfolio', async (req, res) => {
  try {
    const { companyProfile, generateMaterials, generateOpportunities, generateRecommendations } = req.body;

    if (!companyProfile) {
      return res.status(400).json({ error: 'Company profile is required' });
    }

    // Check if DeepSeek API key is configured
    if (!process.env.DEEPSEEK_API_KEY) {
      return res.status(500).json({ 
        error: 'DeepSeek API key not configured',
        details: 'Please set DEEPSEEK_API_KEY environment variable'
      });
    }

    // Initialize AI Portfolio Generator
    const { AIPortfolioGenerator } = require('./ai-portfolio-generator');
    const generator = new AIPortfolioGenerator();

    console.log('🤖 Starting AI portfolio generation for:', companyProfile.company_name);

    // Generate comprehensive portfolio
    const portfolio = await generator.generatePortfolio(companyProfile);

    res.json({
      success: true,
      materials: portfolio.materials,
      opportunities: portfolio.opportunities,
      recommendations: portfolio.recommendations,
      summary: portfolio.summary,
      analysis: portfolio.analysis
    });

  } catch (error) {
    console.error('❌ AI Portfolio generation error:', error);
    res.status(500).json({ 
      error: 'Failed to generate AI portfolio',
      details: error.message 
    });
  }
});

// New AI Portfolio Generation Endpoint (Comprehensive)
app.post('/api/ai/portfolio/generate', async (req, res) => {
  try {
    const { companyProfile, generateMaterials, generateOpportunities, generateRecommendations, user_id } = req.body;

    if (!companyProfile) {
      return res.status(400).json({ error: 'Company profile is required' });
    }

    // Check if DeepSeek API key is configured
    if (!process.env.DEEPSEEK_API_KEY) {
      return res.status(500).json({ 
        error: 'DeepSeek API key not configured',
        details: 'Please set DEEPSEEK_API_KEY environment variable'
      });
    }

    // Initialize AI Portfolio Generator
    const { AIPortfolioGenerator } = require('./ai-portfolio-generator');
    const generator = new AIPortfolioGenerator();

    console.log('🤖 Starting comprehensive AI portfolio generation for:', companyProfile.company_name);

    // Generate comprehensive portfolio
    const portfolio = await generator.generatePortfolio(companyProfile);

    // Save to database if user_id is provided
    if (user_id) {
      try {
        await savePortfolioToDatabase(user_id, portfolio);
        console.log('✅ Portfolio saved to database for user:', user_id);
      } catch (dbError) {
        console.error('⚠️ Failed to save portfolio to database:', dbError);
        // Continue with response even if database save fails
      }
    }

    res.json({
      success: true,
      data: {
        materials: portfolio.materials,
        opportunities: portfolio.opportunities,
        recommendations: portfolio.recommendations,
        summary: portfolio.summary,
        analysis: portfolio.analysis
      }
    });

  } catch (error) {
    console.error('❌ Comprehensive AI Portfolio generation error:', error);
    res.status(500).json({ 
      error: 'Failed to generate AI portfolio',
      details: error.message 
    });
  }
});

// AI Company Analysis Endpoint
app.post('/api/ai/company/analyze', async (req, res) => {
  try {
    const { companyProfile } = req.body;

    if (!companyProfile) {
      return res.status(400).json({ error: 'Company profile is required' });
    }

    // Check if DeepSeek API key is configured
    if (!process.env.DEEPSEEK_API_KEY) {
      return res.status(500).json({ 
        error: 'DeepSeek API key not configured',
        details: 'Please set DEEPSEEK_API_KEY environment variable'
      });
    }

    // Initialize AI Portfolio Generator
    const { AIPortfolioGenerator } = require('./ai-portfolio-generator');
    const generator = new AIPortfolioGenerator();

    console.log('🤖 Starting company analysis for:', companyProfile.company_name);

    // Analyze company profile
    const analysis = await generator.analyzeCompanyProfile(companyProfile);

    res.json({
      success: true,
      data: analysis
    });

  } catch (error) {
    console.error('❌ Company analysis error:', error);
    res.status(500).json({ 
      error: 'Failed to analyze company profile',
      details: error.message 
    });
  }
});

// Shipping and Material Exchange Endpoints
app.post('/api/shipping/calculate-rates', async (req, res) => {
  try {
    const { materialId, buyerLocation } = req.body;
    
    if (!materialId || !buyerLocation) {
      return res.status(400).json({ error: 'Material ID and buyer location are required' });
    }

    const rates = await shippingService.calculateShippingCosts(materialId, buyerLocation);
    res.json({ success: true, rates });
  } catch (error) {
    console.error('Error calculating shipping rates:', error);
    res.status(500).json({ error: 'Failed to calculate shipping rates' });
  }
});

app.post('/api/shipping/create-exchange', async (req, res) => {
  try {
    const { from_company_id, to_company_id, material_id, quantity, from_address, to_address, package_details } = req.body;
    
    // Validate required fields
    if (!from_company_id || !to_company_id || !material_id || !quantity || !from_address || !to_address) {
      return res.status(400).json({ error: 'All required fields must be provided' });
    }

    const exchangeData = {
      from_company_id,
      to_company_id,
      material_id,
      quantity,
      from_address,
      to_address,
      package_details
    };

    const result = await shippingService.createMaterialExchange(exchangeData);
    res.json({ success: true, exchange: result });
  } catch (error) {
    console.error('Error creating material exchange:', error);
    res.status(500).json({ error: 'Failed to create material exchange' });
  }
});

app.get('/api/shipping/track/:trackingNumber', async (req, res) => {
  try {
    const { trackingNumber } = req.params;
    const { carrier } = req.query;
    
    if (!trackingNumber) {
      return res.status(400).json({ error: 'Tracking number is required' });
    }

    const tracking = await shippingService.trackShipment(trackingNumber, carrier);
    res.json({ success: true, tracking });
  } catch (error) {
    console.error('Error tracking shipment:', error);
    res.status(500).json({ error: 'Failed to track shipment' });
  }
});

app.post('/api/shipping/validate-address', async (req, res) => {
  try {
    const address = req.body;
    
    if (!address) {
      return res.status(400).json({ error: 'Address is required' });
    }

    const validation = await shippingService.validateAddress(address);
    res.json({ success: true, validation });
  } catch (error) {
    console.error('Error validating address:', error);
    res.status(500).json({ error: 'Failed to validate address' });
  }
});

app.get('/api/shipping/history/:companyId', async (req, res) => {
  try {
    const { companyId } = req.params;
    
    if (!companyId) {
      return res.status(400).json({ error: 'Company ID is required' });
    }

    const history = await shippingService.getShippingHistory(companyId);
    res.json({ success: true, history });
  } catch (error) {
    console.error('Error getting shipping history:', error);
    res.status(500).json({ error: 'Failed to get shipping history' });
  }
});

app.put('/api/shipping/exchange/:exchangeId/status', async (req, res) => {
  try {
    const { exchangeId } = req.params;
    const { status } = req.body;
    
    if (!exchangeId || !status) {
      return res.status(400).json({ error: 'Exchange ID and status are required' });
    }

    // Update exchange status in database
    const { data: exchange, error } = await supabase
      .from('material_exchanges')
      .update({ 
        status,
        updated_at: new Date().toISOString(),
        confirmed_at: status === 'confirmed' ? new Date().toISOString() : null,
        shipped_at: status === 'shipped' ? new Date().toISOString() : null,
        delivered_at: status === 'delivered' ? new Date().toISOString() : null
      })
      .eq('id', exchangeId)
      .select()
      .single();

    if (error) throw error;

    res.json({ success: true, exchange });
  } catch (error) {
    console.error('Error updating exchange status:', error);
    res.status(500).json({ error: 'Failed to update exchange status' });
  }
});

// Next-Gen Materials API Endpoints
app.get('/api/materials/:materialName/data', async (req, res) => {
  try {
    const { materialName } = req.params;
    
    if (!materialName) {
      return res.status(400).json({ error: 'Material name is required' });
    }

    const materialData = await materialsService.getMaterialData(materialName);
    res.json({ success: true, material: materialData });
  } catch (error) {
    console.error('Error fetching material data:', error);
    res.status(500).json({ error: 'Failed to fetch material data' });
  }
});

app.get('/api/materials/:materialId/properties', async (req, res) => {
  try {
    const { materialId } = req.params;
    
    if (!materialId) {
      return res.status(400).json({ error: 'Material ID is required' });
    }

    const properties = await materialsService.getMaterialProperties(materialId);
    res.json({ success: true, properties });
  } catch (error) {
    console.error('Error fetching material properties:', error);
    res.status(500).json({ error: 'Failed to fetch material properties' });
  }
});

app.get('/api/materials/:materialId/alternatives', async (req, res) => {
  try {
    const { materialId } = req.params;
    
    if (!materialId) {
      return res.status(400).json({ error: 'Material ID is required' });
    }

    const alternatives = await materialsService.findSustainableAlternatives(materialId);
    res.json({ success: true, alternatives });
  } catch (error) {
    console.error('Error finding alternatives:', error);
    res.status(500).json({ error: 'Failed to find alternatives' });
  }
});

app.post('/api/materials/impact/calculate', async (req, res) => {
  try {
    const { materialId, quantity, unit } = req.body;
    
    if (!materialId || !quantity) {
      return res.status(400).json({ error: 'Material ID and quantity are required' });
    }

    const impact = await materialsService.calculateEnvironmentalImpact(materialId, quantity, unit);
    res.json({ success: true, impact });
  } catch (error) {
    console.error('Error calculating environmental impact:', error);
    res.status(500).json({ error: 'Failed to calculate environmental impact' });
  }
});

app.get('/api/materials/:materialId/circular-opportunities', async (req, res) => {
  try {
    const { materialId } = req.params;
    
    if (!materialId) {
      return res.status(400).json({ error: 'Material ID is required' });
    }

    const opportunities = await materialsService.getCircularEconomyOpportunities(materialId);
    res.json({ success: true, opportunities });
  } catch (error) {
    console.error('Error fetching circular economy opportunities:', error);
    res.status(500).json({ error: 'Failed to fetch circular economy opportunities' });
  }
});

app.post('/api/materials/matching/scientific', async (req, res) => {
  try {
    const { companyProfile, wasteStreams, resourceNeeds } = req.body;
    
    if (!companyProfile) {
      return res.status(400).json({ error: 'Company profile is required' });
    }

    const matches = await materialsService.findScientificMatches(companyProfile, wasteStreams, resourceNeeds);
    res.json({ success: true, matches });
  } catch (error) {
    console.error('Error finding scientific matches:', error);
    res.status(500).json({ error: 'Failed to find scientific matches' });
  }
});

app.get('/api/materials/:materialId/supply-chain', async (req, res) => {
  try {
    const { materialId } = req.params;
    const { location } = req.query;
    
    if (!materialId) {
      return res.status(400).json({ error: 'Material ID is required' });
    }

    const insights = await materialsService.getSupplyChainInsights(materialId, location);
    res.json({ success: true, insights });
  } catch (error) {
    console.error('Error fetching supply chain insights:', error);
    res.status(500).json({ error: 'Failed to fetch supply chain insights' });
  }
});

app.post('/api/portfolio/scientific', async (req, res) => {
  try {
    const { companyData } = req.body;
    
    if (!companyData) {
      return res.status(400).json({ error: 'Company data is required' });
    }

    const portfolio = await materialsService.generateScientificPortfolio(companyData);
    res.json({ success: true, portfolio });
  } catch (error) {
    console.error('Error generating scientific portfolio:', error);
    res.status(500).json({ error: 'Failed to generate scientific portfolio' });
  }
});

// Enhanced AI onboarding with scientific data
app.post('/api/ai-onboarding/scientific-complete', async (req, res) => {
  try {
    const { companyProfile, onboardingData } = req.body;
    
    // Get the authenticated user's email from the session/token
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    if (authError || !user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    // Ensure email is included in the company profile
    const enrichedCompanyProfile = {
      ...companyProfile,
      email: user.email,
      onboarding_completed: true,
      updated_at: new Date().toISOString()
    };

    // Save or update the company profile in the database
    const { data: savedCompany, error: saveError } = await supabase
      .from('companies')
      .upsert([{
        id: user.id,
        ...enrichedCompanyProfile
      }], {
        onConflict: 'id',
        ignoreDuplicates: false
      })
      .select()
      .single();

    if (saveError) {
      console.error('Error saving company profile:', saveError);
      return res.status(500).json({ error: 'Failed to save company profile' });
    }

    // Generate scientific portfolio using Next-Gen Materials API
    const scientificPortfolio = await materialsService.generateScientificPortfolio(enrichedCompanyProfile);

    // Use DeepSeek for additional AI insights
    const prompt = `
    You are an AI assistant for an Industrial Symbiosis Management platform.
    Analyze the following company profile and scientific material data to create comprehensive symbiosis recommendations:
    
    Company Profile:
    ${JSON.stringify(enrichedCompanyProfile, null, 2)}
    
    Scientific Material Data:
    ${JSON.stringify(scientificPortfolio, null, 2)}
    
    Provide enhanced analysis including:
    1. Scientific material compatibility assessments
    2. Sustainability impact calculations
    3. Circular economy opportunities
    4. Supply chain optimization recommendations
    5. Environmental impact projections
    6. Implementation roadmap with scientific backing
    7. Risk assessment for material exchanges
    8. Regulatory compliance considerations
    
    Return the analysis in this JSON format:
    {
      "scientific_analysis": {
        "material_compatibility": [...],
        "sustainability_metrics": {...},
        "circular_economy_opportunities": [...],
        "environmental_impact": {...},
        "supply_chain_optimization": [...],
        "implementation_roadmap": [...],
        "risk_assessment": {...},
        "regulatory_compliance": [...]
      }
    }
    `;

    const response = await fetch('https://api.deepseek.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.DEEPSEEK_API_KEY}`
      },
      body: JSON.stringify({
        model: 'deepseek-reasoner',
        messages: [
          {
            role: 'system',
            content: 'You are an expert in industrial symbiosis, materials science, and circular economy. Analyze company data with scientific material properties to generate evidence-based recommendations.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 3000,
        temperature: 0.7,
        response_format: { type: 'json_object' }
      })
    });

    if (!response.ok) {
      throw new Error('Failed to complete scientific analysis');
    }

    const data = await response.json();
    const aiResponse = data.choices[0].message.content;
    
    // Parse the AI response
    let scientificAnalysis = {};
    try {
      const parsed = JSON.parse(aiResponse);
      scientificAnalysis = parsed;
    } catch (e) {
      // Fallback analysis if parsing fails
      scientificAnalysis = {
        scientific_analysis: {
          material_compatibility: ['Based on scientific material properties'],
          sustainability_metrics: { score: scientificPortfolio.sustainability_score },
          circular_economy_opportunities: scientificPortfolio.circular_opportunities,
          environmental_impact: scientificPortfolio.environmental_impact,
          supply_chain_optimization: ['Optimize based on material properties'],
          implementation_roadmap: ['Scientific approach to implementation'],
          risk_assessment: { low: 'Based on material compatibility' },
          regulatory_compliance: ['Ensure compliance with material regulations']
        }
      };
    }

    // Save scientific insights to database
    const { error: insightsError } = await supabase
      .from('ai_insights')
      .insert({
        company_id: user.id,
        symbiosis_score: scientificPortfolio.symbiosis_potential,
        estimated_savings: `$${scientificPortfolio.recommendations.reduce((sum, rec) => sum + (rec.potential_savings || 0), 0).toFixed(0)} annually`,
        carbon_reduction: `${scientificPortfolio.environmental_impact.reduce((sum, impact) => sum + (impact.carbon_savings || 0), 0).toFixed(0)} kg CO2e`,
        top_opportunities: scientificPortfolio.recommendations.map(rec => rec.title),
        recommended_partners: ['Based on scientific material compatibility'],
        implementation_roadmap: scientificAnalysis.scientific_analysis?.implementation_roadmap || ['Scientific implementation approach'],
        scientific_data: scientificPortfolio,
        sustainability_score: scientificPortfolio.sustainability_score
      });

    if (insightsError) {
      console.error('Error saving scientific insights:', insightsError);
    }

    console.log('Scientific onboarding completed successfully for company:', savedCompany.name);

    res.json({ 
      success: true, 
      scientific_portfolio: scientificPortfolio,
      scientific_analysis: scientificAnalysis,
      companyProfile: enrichedCompanyProfile,
      message: 'Scientific onboarding completed successfully. Your profile has been enhanced with material science data and sustainability metrics.'
    });
  } catch (error) {
    console.error('Error completing scientific onboarding:', error);
    res.status(500).json({ error: 'Failed to complete scientific onboarding' });
  }
});

// Payment Routes
app.post('/api/payments/create-order', async (req, res) => {
  try {
    const { exchangeData } = req.body;
    
    if (!exchangeData) {
      return res.status(400).json({ error: 'Exchange data is required' });
    }

    const order = await paymentService.createOrder(exchangeData);
    res.json(order);
  } catch (error) {
    console.error('Error creating payment order:', error);
    res.status(500).json({ error: 'Failed to create payment order' });
  }
});

app.post('/api/payments/capture/:orderId', async (req, res) => {
  try {
    const { orderId } = req.params;
    
    if (!orderId) {
      return res.status(400).json({ error: 'Order ID is required' });
    }

    const capture = await paymentService.capturePayment(orderId);
    res.json(capture);
  } catch (error) {
    console.error('Error capturing payment:', error);
    res.status(500).json({ error: 'Failed to capture payment' });
  }
});

app.post('/api/payments/create-subscription', async (req, res) => {
  try {
    const { subscriptionData } = req.body;
    
    if (!subscriptionData) {
      return res.status(400).json({ error: 'Subscription data is required' });
    }

    const subscription = await paymentService.createSubscription(subscriptionData);
    res.json(subscription);
  } catch (error) {
    console.error('Error creating subscription:', error);
    res.status(500).json({ error: 'Failed to create subscription' });
  }
});

app.post('/api/payments/refund', async (req, res) => {
  try {
    const { captureId, amount, reason } = req.body;
    
    if (!captureId || !amount) {
      return res.status(400).json({ error: 'Capture ID and amount are required' });
    }

    const refund = await paymentService.processRefund(captureId, amount, reason);
    res.json(refund);
  } catch (error) {
    console.error('Error processing refund:', error);
    res.status(500).json({ error: 'Failed to process refund' });
  }
});

app.get('/api/payments/analytics/:companyId', async (req, res) => {
  try {
    const { companyId } = req.params;
    const { dateRange } = req.query;
    
    if (!companyId) {
      return res.status(400).json({ error: 'Company ID is required' });
    }

    const analytics = await paymentService.getPaymentAnalytics(companyId, dateRange);
    res.json(analytics);
  } catch (error) {
    console.error('Error getting payment analytics:', error);
    res.status(500).json({ error: 'Failed to get payment analytics' });
  }
});

// PayPal Webhook
app.post('/api/payments/webhook', async (req, res) => {
  try {
    const event = req.body;
    
    // Verify webhook signature (implement based on PayPal docs)
    // const isValid = verifyWebhookSignature(req.headers, req.body);
    // if (!isValid) {
    //   return res.status(400).json({ error: 'Invalid webhook signature' });
    // }

    await paymentService.handleWebhook(event);
    res.json({ received: true });
  } catch (error) {
    console.error('Error handling webhook:', error);
    res.status(500).json({ error: 'Failed to handle webhook' });
  }
});

// Height API Endpoints
app.post('/api/height/create-exchange-tracking', async (req, res) => {
  try {
    const { exchangeData } = req.body;
    
    if (!exchangeData) {
      return res.status(400).json({ error: 'Exchange data is required' });
    }

    const tracking = await heightService.createMaterialExchangeTracking(exchangeData);
    res.json({ success: true, tracking });
  } catch (error) {
    console.error('Error creating Height tracking:', error);
    res.status(500).json({ error: 'Failed to create Height tracking' });
  }
});

app.post('/api/height/create-sustainability-tracking', async (req, res) => {
  try {
    const { impactData } = req.body;
    
    if (!impactData) {
      return res.status(400).json({ error: 'Impact data is required' });
    }

    const tracking = await heightService.createSustainabilityTracking(impactData);
    res.json({ success: true, tracking });
  } catch (error) {
    console.error('Error creating sustainability tracking:', error);
    res.status(500).json({ error: 'Failed to create sustainability tracking' });
  }
});

app.get('/api/height/project/:projectId', async (req, res) => {
  try {
    const { projectId } = req.params;
    
    if (!projectId) {
      return res.status(400).json({ error: 'Project ID is required' });
    }

    const projectDetails = await heightService.getProjectDetails(projectId);
    res.json({ success: true, project: projectDetails });
  } catch (error) {
    console.error('Error fetching Height project:', error);
    res.status(500).json({ error: 'Failed to fetch Height project' });
  }
});

app.patch('/api/height/task/:taskId/status', async (req, res) => {
  try {
    const { taskId } = req.params;
    const { status } = req.body;
    
    if (!taskId || !status) {
      return res.status(400).json({ error: 'Task ID and status are required' });
    }

    const updatedTask = await heightService.updateTaskStatus(taskId, status);
    res.json({ success: true, task: updatedTask });
  } catch (error) {
    console.error('Error updating Height task status:', error);
    res.status(500).json({ error: 'Failed to update task status' });
  }
});

app.post('/api/height/task/:taskId/comment', async (req, res) => {
  try {
    const { taskId } = req.params;
    const { comment } = req.body;
    
    if (!taskId || !comment) {
      return res.status(400).json({ error: 'Task ID and comment are required' });
    }

    const newComment = await heightService.addTaskComment(taskId, comment);
    res.json({ success: true, comment: newComment });
  } catch (error) {
    console.error('Error adding Height task comment:', error);
    res.status(500).json({ error: 'Failed to add task comment' });
  }
});

app.get('/api/height/workspace/members', async (req, res) => {
  try {
    const members = await heightService.getWorkspaceMembers();
    res.json({ success: true, members });
  } catch (error) {
    console.error('Error fetching Height workspace members:', error);
    res.status(500).json({ error: 'Failed to fetch workspace members' });
  }
});

// INTELLIGENT MATCHING ENDPOINTS
app.post('/api/intelligent-matching', async (req, res) => {
  try {
    const { companyData, options } = req.body;
    
    console.log('🚀 Starting intelligent matching request');
    
    const result = await intelligentMatchingService.findIntelligentMatches(companyData, options);
    
    res.json(result);
  } catch (error) {
    console.error('Intelligent matching error:', error);
    res.status(500).json({ error: error.message });
  }
});

// API FUSION ENDPOINTS
app.post('/api/materials/translate-shipping', async (req, res) => {
  try {
    const { materialData } = req.body;
    
    const shippingParams = await apiFusionService.translateMaterialToShippingParams(materialData);
    
    res.json({
      success: true,
      shipping_params: shippingParams
    });
  } catch (error) {
    console.error('Material translation error:', error);
    res.status(500).json({ error: error.message });
  }
});

// AI EVOLUTION ENDPOINTS
app.post('/api/ai-analysis', async (req, res) => {
  try {
    const { analysisType, inputData, context } = req.body;
    
    const result = await aiEvolutionEngine.executeAIAnalysis(analysisType, inputData, context);
    
    res.json(result);
  } catch (error) {
    console.error('AI analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/ai-feedback', async (req, res) => {
  try {
    const { feedbackId, feedbackData } = req.body;
    
    const result = await aiEvolutionEngine.collectUserFeedback(feedbackId, feedbackData);
    
    res.json(result);
  } catch (error) {
    console.error('AI feedback error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/ai-feedback-stats', async (req, res) => {
  try {
    const stats = await aiEvolutionEngine.getFeedbackStats();
    
    res.json({
      success: true,
      stats: stats
    });
  } catch (error) {
    console.error('AI feedback stats error:', error);
    res.status(500).json({ error: error.message });
  }
});

// ENHANCED SHIPPING ENDPOINTS
app.post('/api/shipping/rates', async (req, res) => {
  try {
    const { materialData, fromAddress, toAddress } = req.body;
    
    const shippingService = require('./services/shippingService');
    const rates = await shippingService.getShippingRates(materialData, fromAddress, toAddress);
    
    res.json(rates);
  } catch (error) {
    console.error('Shipping rates error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Freightos API endpoints
app.post('/api/freightos/rates', async (req, res) => {
  try {
    const shipmentData = req.body;
    
    const rates = await freightosService.getFreightRates(shipmentData);
    
    res.json(rates);
  } catch (error) {
    console.error('Freightos rates error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/freightos/emissions', async (req, res) => {
  try {
    const shipmentData = req.body;
    
    const emissions = await freightosService.getCO2Emissions(shipmentData);
    
    res.json(emissions);
  } catch (error) {
    console.error('Freightos emissions error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/freightos/network/:region?', async (req, res) => {
  try {
    const region = req.params.region || 'gulf';
    
    const network = await freightosService.getLogisticsNetwork(region);
    
    res.json(network);
  } catch (error) {
    console.error('Freightos network error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/shipping/label', async (req, res) => {
  try {
    const { materialData, fromAddress, toAddress, rateId } = req.body;
    
    const shippingService = require('./services/shippingService');
    const label = await shippingService.createShippingLabel(materialData, fromAddress, toAddress, rateId);
    
    res.json(label);
  } catch (error) {
    console.error('Shipping label error:', error);
    res.status(500).json({ error: error.message });
  }
});

// COMPREHENSIVE MATCH ANALYSIS
app.post('/api/comprehensive-match-analysis', async (req, res) => {
  try {
    const { buyer_data, seller_data, match_data } = req.body;
    
    // Use AI evolution engine for comprehensive analysis
    const analysisResult = await aiEvolutionEngine.executeAIAnalysis('matchGeneration', {
      company_a: buyer_data,
      company_b: seller_data,
      material_a: match_data.material_a,
      material_b: match_data.material_b
    });
    
    res.json(analysisResult);
  } catch (error) {
    console.error('Comprehensive match analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

// MATERIAL ANALYSIS ENDPOINT
app.post('/api/materials/analyze', async (req, res) => {
  try {
    const { materialData } = req.body;
    
    const analysisResult = await aiEvolutionEngine.executeAIAnalysis('materialAnalysis', materialData);
    
    res.json(analysisResult);
  } catch (error) {
    console.error('Material analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

// PORTFOLIO CREATION ENDPOINT
app.post('/api/portfolio/create', async (req, res) => {
  try {
    const { companyData } = req.body;
    
    const portfolioResult = await aiEvolutionEngine.executeAIAnalysis('portfolioCreation', companyData);
    
    res.json(portfolioResult);
  } catch (error) {
    console.error('Portfolio creation error:', error);
    res.status(500).json({ error: error.message });
  }
});

// SUSTAINABILITY ASSESSMENT ENDPOINT
app.post('/api/sustainability/assess', async (req, res) => {
  try {
    const { companyData } = req.body;
    
    const assessmentResult = await aiEvolutionEngine.executeAIAnalysis('sustainabilityAssessment', companyData);
    
    res.json(assessmentResult);
  } catch (error) {
    console.error('Sustainability assessment error:', error);
    res.status(500).json({ error: error.message });
  }
});

// SHIPPING OPTIMIZATION ENDPOINT
app.post('/api/shipping/optimize', async (req, res) => {
  try {
    const { shippingData } = req.body;
    
    const optimizationResult = await aiEvolutionEngine.executeAIAnalysis('shippingOptimization', shippingData);
    
    res.json(optimizationResult);
  } catch (error) {
    console.error('Shipping optimization error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Import new services
const RealDataProcessor = require('./services/realDataProcessor');
const BulkDataImporter = require('./services/bulkDataImporter');
// Temporarily disabled ProductionMonitoring to prevent crashes
// const ProductionMonitoring = require('./services/productionMonitoring');

// Initialize services
const realDataProcessor = new RealDataProcessor();
const bulkDataImporter = new BulkDataImporter();
// const monitoring = ProductionMonitoring.getInstance();

// Add monitoring middleware
// app.use(monitoring.httpRequestMiddleware());

// REAL DATA PROCESSING ENDPOINTS

// Process single real company data
app.post('/api/real-data/process-company', async (req, res) => {
  try {
    const { companyData } = req.body;
    
    if (!companyData) {
      return res.status(400).json({ error: 'Company data is required' });
    }

    const result = await realDataProcessor.processRealCompanyData(companyData);
    
    res.json({
      success: true,
      message: 'Company data processed successfully',
      data: result
    });
  } catch (error) {
    console.error('Real data processing error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Bulk import 50 real company profiles
app.post('/api/real-data/bulk-import', async (req, res) => {
  try {
    const { companies, options } = req.body;
    
    if (!companies || !Array.isArray(companies) || companies.length === 0) {
      return res.status(400).json({ error: 'Companies array is required' });
    }

    if (companies.length > 100) {
      return res.status(400).json({ error: 'Maximum 100 companies per import' });
    }

    // Start bulk import
    const result = await bulkDataImporter.importRealCompanyData(companies, options);
    
    res.json({
      success: true,
      message: `Bulk import completed. ${result.summary.successful} companies processed successfully.`,
      data: result
    });
  } catch (error) {
    console.error('Bulk import error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get bulk import status
app.get('/api/real-data/import-status/:importId', async (req, res) => {
  try {
    const { importId } = req.params;
    
    const { data, error } = await supabase
      .from('bulk_imports')
      .select('*')
      .eq('import_id', importId)
      .single();

    if (error) {
      return res.status(404).json({ error: 'Import not found' });
    }

    res.json({
      success: true,
      data: data
    });
  } catch (error) {
    console.error('Import status error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get high-value targets for outreach
app.get('/api/real-data/high-value-targets', async (req, res) => {
  try {
    const { limit = 10, min_savings = 50000 } = req.query;
    
    const { data, error } = await supabase
      .from('companies')
      .select('*')
      .gte('business_metrics->potential_savings', min_savings)
      .order('business_metrics->potential_savings', { ascending: false })
      .limit(parseInt(limit));

    if (error) {
      throw error;
    }

    const targets = data.map(company => ({
      id: company.id,
      name: company.name,
      industry: company.industry,
      location: company.location,
      potential_savings: company.business_metrics?.potential_savings,
      symbiosis_score: company.business_metrics?.symbiosis_score,
      carbon_reduction: company.business_metrics?.carbon_reduction_potential,
      data_quality_score: company.data_quality_score,
      contact_info: company.contact_info
    }));

    res.json({
      success: true,
      data: targets
    });
  } catch (error) {
    console.error('High-value targets error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get symbiosis network analysis
app.get('/api/real-data/symbiosis-network', async (req, res) => {
  try {
    const { data: companies, error } = await supabase
      .from('companies')
      .select('*')
      .not('ai_insights', 'is', null);

    if (error) {
      throw error;
    }

    const networkAnalysis = await bulkDataImporter.generateBulkInsights({
      successful: companies.map(c => ({ result: { result: c } }))
    });

    res.json({
      success: true,
      data: networkAnalysis
    });
  } catch (error) {
    console.error('Symbiosis network error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get market analysis
app.get('/api/real-data/market-analysis', async (req, res) => {
  try {
    const { data: companies, error } = await supabase
      .from('companies')
      .select('*');

    if (error) {
      throw error;
    }

    const marketAnalysis = await bulkDataImporter.generateBulkInsights({
      successful: companies.map(c => ({ result: { result: c } }))
    });

    res.json({
      success: true,
      data: marketAnalysis.market_analysis
    });
  } catch (error) {
    console.error('Market analysis error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get production monitoring metrics (temporarily disabled)
// app.get('/api/monitoring/metrics', async (req, res) => {
//   try {
//     const metrics = await monitoring.getMetrics();
//     res.set('Content-Type', 'text/plain');
//     res.send(metrics);
//   } catch (error) {
//     console.error('Metrics error:', error);
//     res.status(500).json({ error: error.message });
//   }
// });

// Get system health (temporarily disabled)
// app.get('/api/monitoring/health', async (req, res) => {
//   try {
//     const health = await monitoring.healthCheck();
//     res.json(health);
//   } catch (error) {
//     console.error('Health check error:', error);
//     res.status(500).json({ error: error.message });
//   }
// });

// Import revolutionary AI listings generator
const RevolutionaryAIListingsGenerator = require('./services/revolutionaryAIListingsGenerator');

// Initialize revolutionary AI listings generator
const aiListingsGenerator = new RevolutionaryAIListingsGenerator();

// REVOLUTIONARY AI LISTINGS GENERATION ENDPOINTS

// Generate AI listings for a specific company
app.post('/api/ai/generate-listings/:companyId', async (req, res) => {
  try {
    const { companyId } = req.params;
    
    // Get company data
    const { data: company, error: companyError } = await supabase
      .from('companies')
      .select('*')
      .eq('id', companyId)
      .single();

    if (companyError || !company) {
      return res.status(404).json({ error: 'Company not found' });
    }

    // Generate AI listings
    const result = await aiListingsGenerator.generateCompanyListings(company);
    
    res.json({
      success: true,
      message: 'AI listings generated successfully',
      data: result
    });
  } catch (error) {
    console.error('AI listings generation error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Generate AI listings for all companies
app.post('/api/ai/generate-all-listings', async (req, res) => {
  try {
    const { data: companies, error: companiesError } = await supabase
      .from('companies')
      .select('*');

    if (companiesError) {
      throw companiesError;
    }

    if (!companies || companies.length === 0) {
      return res.status(404).json({ error: 'No companies found' });
    }

    const results = [];
    const errors = [];

    // Generate listings for each company
    for (const company of companies) {
      try {
        const result = await aiListingsGenerator.generateCompanyListings(company);
        results.push({
          company_id: company.id,
          company_name: company.name,
          success: true,
          data: result
        });
      } catch (error) {
        errors.push({
          company_id: company.id,
          company_name: company.name,
          error: error.message
        });
      }
    }

    const summary = {
      total_companies: companies.length,
      successful: results.length,
      failed: errors.length,
      total_listings_generated: results.reduce((sum, r) => sum + r.data.generation_summary.total_listings, 0),
      total_potential_value: results.reduce((sum, r) => sum + r.data.business_metrics.total_potential_value, 0)
    };

    res.json({
      success: true,
      message: `Generated listings for ${results.length}/${companies.length} companies`,
      summary,
      results,
      errors
    });
  } catch (error) {
    console.error('Bulk AI listings generation error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get AI listings statistics
app.get('/api/ai/listings-stats', async (req, res) => {
  try {
    const { data: listings, error } = await supabase
      .from('materials')
      .select('*')
      .eq('ai_generated', true);

    if (error) throw error;

    const stats = {
      total_ai_listings: listings.length,
      waste_listings: listings.filter(l => l.type === 'waste').length,
      requirement_listings: listings.filter(l => l.type === 'requirement').length,
      average_confidence_score: listings.reduce((sum, l) => sum + (l.confidence_score || 0), 0) / listings.length,
      total_potential_value: listings.reduce((sum, l) => sum + (l.potential_value || 0), 0),
      average_sustainability_score: listings.reduce((sum, l) => sum + (l.sustainability_score || 0), 0) / listings.length
    };

    res.json({
      success: true,
      data: stats
    });
  } catch (error) {
    console.error('AI listings stats error:', error);
    res.status(500).json({ error: error.message });
  }
});

// ==================== ADMIN ENDPOINTS ====================

// Get all users for admin
app.get('/api/admin/users', async (req, res) => {
  try {
    const { data: users, error } = await supabase
      .from('users')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) throw error;
    res.json({ success: true, users });
  } catch (error) {
    console.error('Error fetching users:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Admin user upgrade endpoint
app.post('/api/admin/upgrade-user', async (req, res) => {
  try {
    const { userId, subscriptionType } = req.body;
    
    if (!userId || !subscriptionType) {
      return res.status(400).json({ 
        success: false, 
        error: 'User ID and subscription type are required' 
      });
    }

    // Update user subscription
    const { data, error } = await supabase
      .from('users')
      .update({ 
        subscription_type: subscriptionType,
        subscription_status: 'active',
        updated_at: new Date().toISOString()
      })
      .eq('id', userId)
      .select();

    if (error) throw error;

    res.json({ 
      success: true, 
      message: `User upgraded to ${subscriptionType}`,
      user: data[0]
    });
  } catch (error) {
    console.error('Error upgrading user:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Admin downgrade user endpoint
app.post('/api/admin/downgrade-user', async (req, res) => {
  try {
    const { userId } = req.body;
    
    if (!userId) {
      return res.status(400).json({ 
        success: false, 
        error: 'User ID is required' 
      });
    }

    // Downgrade user to free
    const { data, error } = await supabase
      .from('users')
      .update({ 
        subscription_type: 'free',
        subscription_status: 'active',
        updated_at: new Date().toISOString()
      })
      .eq('id', userId)
      .select();

    if (error) throw error;

    res.json({ 
      success: true, 
      message: 'User downgraded to free',
      user: data[0]
    });
  } catch (error) {
    console.error('Error downgrading user:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get system statistics for admin dashboard
app.get('/api/admin/stats', async (req, res) => {
  try {
    // Get counts from different tables
    const [
      { count: usersCount },
      { count: companiesCount },
      { count: materialsCount },
      { count: matchesCount }
    ] = await Promise.all([
      supabase.from('users').select('*', { count: 'exact', head: true }),
      supabase.from('companies').select('*', { count: 'exact', head: true }),
      supabase.from('materials').select('*', { count: 'exact', head: true }),
      supabase.from('matches').select('*', { count: 'exact', head: true })
    ]);

    res.json({
      success: true,
      stats: {
        total_users: usersCount,
        total_companies: companiesCount,
        total_materials: materialsCount,
        total_matches: matchesCount
      }
    });
  } catch (error) {
    console.error('Error fetching admin stats:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Server startup handled above in conditional block

// ============================================================================
// ADVANCED PYTHON AI SERVICE ENDPOINTS
// ============================================================================

// GNN Reasoning Engine Endpoints
app.post('/api/ai/gnn/create-graph', async (req, res) => {
    try {
        const { participants, graph_type = 'industrial' } = req.body;
        
        if (!participants || !Array.isArray(participants)) {
            return res.status(400).json({ error: 'Participants array is required' });
        }

        const gnnResult = await runPythonScript('gnn_reasoning_engine.py', {
            action: 'create_graph',
            participants: participants,
            graph_type: graph_type
        });

        res.json({
            success: true,
            graph_data: gnnResult.graph_data,
            node_count: gnnResult.node_count,
            edge_count: gnnResult.edge_count,
            message: 'Industrial graph created successfully'
        });

    } catch (error) {
        console.error('GNN Graph Creation Error:', error);
        res.status(500).json({ error: 'Failed to create industrial graph' });
    }
});

app.post('/api/ai/gnn/train-model', async (req, res) => {
    try {
        const { 
            graph_data, 
            model_name = 'default', 
            model_type = 'GCN', 
            task_type = 'node_classification' 
        } = req.body;

        const trainingResult = await runPythonScript('gnn_reasoning_engine.py', {
            action: 'train_model',
            graph_data: graph_data,
            model_name: model_name,
            model_type: model_type,
            task_type: task_type
        });

        res.json({
            success: true,
            model_name: model_name,
            training_metrics: trainingResult.metrics,
            model_performance: trainingResult.performance,
            message: 'GNN model trained successfully'
        });

    } catch (error) {
        console.error('GNN Training Error:', error);
        res.status(500).json({ error: 'Failed to train GNN model' });
    }
});

app.post('/api/ai/gnn/infer', async (req, res) => {
    try {
        const { 
            graph_data, 
            model_name = 'default', 
            inference_type = 'node_embeddings' 
        } = req.body;

        const inferenceResult = await runPythonScript('gnn_reasoning_engine.py', {
            action: 'infer',
            graph_data: graph_data,
            model_name: model_name,
            inference_type: inference_type
        });

        res.json({
            success: true,
            embeddings: inferenceResult.embeddings,
            predictions: inferenceResult.predictions,
            confidence_scores: inferenceResult.confidence_scores,
            message: 'GNN inference completed successfully'
        });

    } catch (error) {
        console.error('GNN Inference Error:', error);
        res.status(500).json({ error: 'Failed to perform GNN inference' });
    }
});

app.get('/api/ai/gnn/models', async (req, res) => {
    try {
        const modelsResult = await runPythonScript('gnn_reasoning_engine.py', {
            action: 'list_models'
        });

        res.json({
            success: true,
            available_models: modelsResult.models,
            model_count: modelsResult.model_count,
            message: 'GNN models retrieved successfully'
        });

    } catch (error) {
        console.error('GNN Models Error:', error);
        res.status(500).json({ error: 'Failed to retrieve GNN models' });
    }
});

// Revolutionary AI Matching Endpoints
app.post('/api/ai/revolutionary/match', async (req, res) => {
    try {
        const { 
            query_company, 
            candidate_companies, 
            algorithm = 'ensemble', 
            top_k = 10, 
            confidence_threshold = 0.5 
        } = req.body;

        if (!query_company || !candidate_companies) {
            return res.status(400).json({ error: 'Query company and candidate companies are required' });
        }

        const matchResult = await runPythonScript('revolutionary_ai_matching.py', {
            action: 'find_matches',
            query_company: query_company,
            candidate_companies: candidate_companies,
            algorithm: algorithm,
            top_k: top_k,
            confidence_threshold: confidence_threshold
        });

        res.json({
            success: true,
            matches: matchResult.candidates,
            total_candidates: matchResult.total_candidates,
            matching_time: matchResult.matching_time,
            algorithm_used: matchResult.algorithm_used,
            message: 'Revolutionary AI matching completed successfully'
        });

    } catch (error) {
        console.error('Revolutionary AI Matching Error:', error);
        res.status(500).json({ error: 'Failed to perform revolutionary AI matching' });
    }
});

app.post('/api/ai/revolutionary/predict-compatibility', async (req, res) => {
    try {
        const { buyer, seller } = req.body;

        if (!buyer || !seller) {
            return res.status(400).json({ error: 'Buyer and seller data are required' });
        }

        const compatibilityResult = await runPythonScript('revolutionary_ai_matching.py', {
            action: 'predict_compatibility',
            buyer: buyer,
            seller: seller
        });

        res.json({
            success: true,
            compatibility_score: compatibilityResult.compatibility_score,
            explanation: compatibilityResult.explanation,
            confidence_level: compatibilityResult.confidence_level,
            risk_assessment: compatibilityResult.risk_assessment,
            opportunity_score: compatibilityResult.opportunity_score,
            roi_prediction: compatibilityResult.roi_prediction,
            message: 'Compatibility prediction completed successfully'
        });

    } catch (error) {
        console.error('Compatibility Prediction Error:', error);
        res.status(500).json({ error: 'Failed to predict compatibility' });
    }
});

app.post('/api/ai/revolutionary/feedback', async (req, res) => {
    try {
        const { match_id, user_id, rating, feedback = '' } = req.body;

        if (!match_id || !user_id || !rating) {
            return res.status(400).json({ error: 'Match ID, user ID, and rating are required' });
        }

        const feedbackResult = await runPythonScript('revolutionary_ai_matching.py', {
            action: 'record_feedback',
            match_id: match_id,
            user_id: user_id,
            rating: rating,
            feedback: feedback
        });

        res.json({
            success: true,
            feedback_recorded: feedbackResult.recorded,
            model_updated: feedbackResult.model_updated,
            message: 'Feedback recorded successfully'
        });

    } catch (error) {
        console.error('Feedback Recording Error:', error);
        res.status(500).json({ error: 'Failed to record feedback' });
    }
});

// Knowledge Graph Endpoints
app.post('/api/ai/knowledge-graph/build', async (req, res) => {
    try {
        const { companies, materials, relationships } = req.body;

        const kgResult = await runPythonScript('knowledge_graph.py', {
            action: 'build_graph',
            companies: companies,
            materials: materials,
            relationships: relationships
        });

        res.json({
            success: true,
            graph_nodes: kgResult.node_count,
            graph_edges: kgResult.edge_count,
            knowledge_base: kgResult.knowledge_base,
            message: 'Knowledge graph built successfully'
        });

    } catch (error) {
        console.error('Knowledge Graph Build Error:', error);
        res.status(500).json({ error: 'Failed to build knowledge graph' });
    }
});

app.post('/api/ai/knowledge-graph/query', async (req, res) => {
    try {
        const { query, query_type = 'semantic' } = req.body;

        if (!query) {
            return res.status(400).json({ error: 'Query is required' });
        }

        const queryResult = await runPythonScript('knowledge_graph.py', {
            action: 'query_graph',
            query: query,
            query_type: query_type
        });

        res.json({
            success: true,
            results: queryResult.results,
            confidence: queryResult.confidence,
            reasoning_path: queryResult.reasoning_path,
            message: 'Knowledge graph query completed successfully'
        });

    } catch (error) {
        console.error('Knowledge Graph Query Error:', error);
        res.status(500).json({ error: 'Failed to query knowledge graph' });
    }
});

// Federated Meta Learning Endpoints
app.post('/api/ai/federated/initialize', async (req, res) => {
    try {
        const { participants, model_config } = req.body;

        const federatedResult = await runPythonScript('federated_meta_learning.py', {
            action: 'initialize',
            participants: participants,
            model_config: model_config
        });

        res.json({
            success: true,
            session_id: federatedResult.session_id,
            participants_count: federatedResult.participants_count,
            model_architecture: federatedResult.model_architecture,
            message: 'Federated learning session initialized successfully'
        });

    } catch (error) {
        console.error('Federated Learning Init Error:', error);
        res.status(500).json({ error: 'Failed to initialize federated learning' });
    }
});

app.post('/api/ai/federated/train', async (req, res) => {
    try {
        const { session_id, local_data, round_number } = req.body;

        if (!session_id || !local_data) {
            return res.status(400).json({ error: 'Session ID and local data are required' });
        }

        const trainingResult = await runPythonScript('federated_meta_learning.py', {
            action: 'train_round',
            session_id: session_id,
            local_data: local_data,
            round_number: round_number
        });

        res.json({
            success: true,
            round_completed: trainingResult.round_completed,
            model_improvement: trainingResult.model_improvement,
            convergence_status: trainingResult.convergence_status,
            message: 'Federated training round completed successfully'
        });

    } catch (error) {
        console.error('Federated Training Error:', error);
        res.status(500).json({ error: 'Failed to complete federated training round' });
    }
});

// Multi-hop Symbiosis Network Endpoints
app.post('/api/ai/multi-hop/analyze', async (req, res) => {
    try {
        const { companies, max_hops = 3, analysis_type = 'symbiosis' } = req.body;

        if (!companies || !Array.isArray(companies)) {
            return res.status(400).json({ error: 'Companies array is required' });
        }

        const multiHopResult = await runPythonScript('multi_hop_symbiosis_network.py', {
            action: 'analyze_network',
            companies: companies,
            max_hops: max_hops,
            analysis_type: analysis_type
        });

        res.json({
            success: true,
            network_graph: multiHopResult.network_graph,
            symbiosis_paths: multiHopResult.symbiosis_paths,
            value_chain_analysis: multiHopResult.value_chain_analysis,
            circular_opportunities: multiHopResult.circular_opportunities,
            message: 'Multi-hop symbiosis analysis completed successfully'
        });

    } catch (error) {
        console.error('Multi-hop Analysis Error:', error);
        res.status(500).json({ error: 'Failed to perform multi-hop symbiosis analysis' });
    }
});

app.post('/api/ai/multi-hop/optimize', async (req, res) => {
    try {
        const { network_data, optimization_goal = 'maximize_value' } = req.body;

        const optimizationResult = await runPythonScript('multi_hop_symbiosis_network.py', {
            action: 'optimize_network',
            network_data: network_data,
            optimization_goal: optimization_goal
        });

        res.json({
            success: true,
            optimized_paths: optimizationResult.optimized_paths,
            value_improvement: optimizationResult.value_improvement,
            efficiency_gains: optimizationResult.efficiency_gains,
            implementation_plan: optimizationResult.implementation_plan,
            message: 'Network optimization completed successfully'
        });

    } catch (error) {
        console.error('Network Optimization Error:', error);
        res.status(500).json({ error: 'Failed to optimize network' });
    }
});

// Advanced AI Integration Endpoints
app.post('/api/ai/integration/comprehensive-analysis', async (req, res) => {
    try {
        const { 
            company_data, 
            analysis_type = 'full', 
            include_gnn = true, 
            include_knowledge_graph = true 
        } = req.body;

        if (!company_data) {
            return res.status(400).json({ error: 'Company data is required' });
        }

        const analysisResult = await runPythonScript('advanced_ai_integration.py', {
            action: 'comprehensive_analysis',
            company_data: company_data,
            analysis_type: analysis_type,
            include_gnn: include_gnn,
            include_knowledge_graph: include_knowledge_graph
        });

        res.json({
            success: true,
            analysis_results: analysisResult.results,
            recommendations: analysisResult.recommendations,
            risk_assessment: analysisResult.risk_assessment,
            opportunity_analysis: analysisResult.opportunity_analysis,
            implementation_roadmap: analysisResult.implementation_roadmap,
            message: 'Comprehensive AI analysis completed successfully'
        });

    } catch (error) {
        console.error('Comprehensive Analysis Error:', error);
        res.status(500).json({ error: 'Failed to perform comprehensive analysis' });
    }
});

// AI Service Health and Status Endpoints
app.get('/api/ai/services/status', (req, res) => {
    try {
        res.json({
            success: true,
            services_status: {
                'gnn_reasoning_engine.py': { status: 'available' },
                'revolutionary_ai_matching.py': { status: 'available' },
                'knowledge_graph.py': { status: 'available' },
                'federated_meta_learning.py': { status: 'available' },
                'multi_hop_symbiosis_network.py': { status: 'available' },
                'advanced_ai_integration.py': { status: 'available' }
            },
            timestamp: new Date().toISOString(),
            message: 'AI services status retrieved successfully'
        });
    } catch (error) {
        console.error('AI Services Status Error:', error);
        res.status(500).json({ error: 'Failed to retrieve AI services status' });
    }
});

app.post('/api/ai/services/restart', async (req, res) => {
    try {
        const { service_name } = req.body;

        if (!service_name) {
            return res.status(400).json({ error: 'Service name is required' });
        }

        const restartResult = await runPythonScript(service_name, {
            action: 'restart_service'
        });

        res.json({
            success: true,
            service_restarted: restartResult.restarted,
            service_name: service_name,
            message: `AI service ${service_name} restarted successfully`
        });

    } catch (error) {
        console.error('AI Service Restart Error:', error);
        res.status(500).json({ error: 'Failed to restart AI service' });
    }
});

// Faster Python execution - direct import instead of spawning
let adaptiveOnboarding = null;

async function initializePythonServices() {
    try {
        // Import Python modules directly if possible
        const { spawn } = require('child_process');
        
        // Initialize adaptive onboarding service
        const initResult = await runPythonScript('adaptive_onboarding_init.py', [
            'initialize'
        ]);
        
        console.log('✅ Python services initialized');
        return true;
    } catch (error) {
        console.log('⚠️ Using fallback Python execution:', error.message);
        return false;
    }
}

// Initialize Python services on startup
initializePythonServices();

// Temporarily disable problematic production monitoring middleware
// app.use(productionMonitoring.httpRequestMiddleware());

// LOGISTICS PREVIEW ENDPOINT - Complete the perfect user flow
app.post('/api/logistics-preview', async (req, res) => {
  try {
    const { origin, destination, material, weight_kg, company_profile } = req.body;
    
    console.log('🚛 Generating logistics preview for:', material);
    
    // Get logistics analysis from Freightos
    const logisticsAnalysis = await freightosService.getFreightEstimate({
      origin: origin,
      destination: destination,
      weight: weight_kg,
      volume: weight_kg * 0.001, // Rough estimate
      commodity: material,
      mode: 'truck', // Start with truck
      container_type: weight_kg > 1000 ? '40ft' : '20ft',
      hazardous: false
    });

    // Get alternative transport modes
    const alternativeModes = await freightosService.getAlternativeModes(
      origin, destination, weight_kg, weight_kg * 0.001
    );

    // Calculate ROI and feasibility
    const materialValue = weight_kg * 2.5; // Estimated $2.5/kg
    const logisticsCost = logisticsAnalysis.total_cost.total_cost;
    const roiPercentage = ((materialValue - logisticsCost) / logisticsCost) * 100;
    const isFeasible = roiPercentage > 20; // 20% ROI threshold

    const logisticsPreview = {
      origin,
      destination,
      material,
      weight_kg,
      transport_modes: alternativeModes.map(mode => ({
        mode: mode.mode,
        cost: mode.cost,
        transit_time: mode.transit_time,
        carbon_emissions: mode.carbon_footprint,
        reliability: mode.sustainability_score / 100
      })),
      total_cost: logisticsCost,
      total_carbon: logisticsAnalysis.carbon_footprint,
      cost_breakdown: {
        transport: logisticsCost * 0.8,
        handling: logisticsCost * 0.12,
        customs: logisticsCost * 0.06,
        insurance: logisticsCost * 0.02
      },
      recommendations: [
        'Truck transport offers best cost-benefit ratio for this distance',
        'Consider bulk shipping for quantities over 5000kg',
        'Negotiate long-term contracts for better rates',
        'Explore carbon offset options to improve sustainability score'
      ],
      is_feasible: isFeasible,
      roi_percentage: Math.max(0, roiPercentage)
    };

    res.json(logisticsPreview);
    
  } catch (error) {
    console.error('Logistics preview error:', error);
    
    // Fallback logistics preview
    const fallbackPreview = {
      origin: req.body.origin,
      destination: req.body.destination,
      material: req.body.material,
      weight_kg: req.body.weight_kg,
      transport_modes: [
        {
          mode: 'Truck',
          cost: 2500,
          transit_time: 2,
          carbon_emissions: 150,
          reliability: 0.95
        },
        {
          mode: 'Sea',
          cost: 1800,
          transit_time: 5,
          carbon_emissions: 80,
          reliability: 0.90
        },
        {
          mode: 'Air',
          cost: 8500,
          transit_time: 1,
          carbon_emissions: 400,
          reliability: 0.99
        }
      ],
      total_cost: 2500,
      total_carbon: 150,
      cost_breakdown: {
        transport: 2000,
        handling: 300,
        customs: 150,
        insurance: 50
      },
      recommendations: [
        'Truck transport offers best cost-benefit ratio',
        'Consider bulk shipping for larger quantities',
        'Negotiate long-term contracts for better rates'
      ],
      is_feasible: true,
      roi_percentage: 85
    };
    
    res.json(fallbackPreview);
  }
});

module.exports = app;

// AI Onboarding endpoints
app.post('/api/ai-onboarding/assess-knowledge', async (req, res) => {
  try {
    const { companyProfile } = req.body;

    if (!companyProfile) {
      return res.status(400).json({ error: 'Company profile is required' });
    }

    // Call the enhanced AI onboarding questions generator
    const options = {
      mode: 'json',
      pythonPath: 'python3',
      pythonOptions: ['-u'],
      scriptPath: './',
      args: ['assess-knowledge', JSON.stringify(companyProfile)]
    };

    const result = await new Promise((resolve, reject) => {
      PythonShell.run('ai_onboarding_questions_generator.py', options, (err, results) => {
        if (err) {
          console.error('Python script error:', err);
          reject(err);
        } else {
          try {
            const data = JSON.parse(results[0]);
            resolve(data);
          } catch (parseError) {
            console.error('JSON parse error:', parseError);
            reject(parseError);
          }
        }
      });
    });

    res.json(result);
  } catch (error) {
    console.error('Error assessing knowledge:', error);
    res.status(500).json({ 
      error: 'Failed to assess knowledge gaps',
      details: error.message 
    });
  }
});

app.post('/api/ai-onboarding/generate-questions', async (req, res) => {
  try {
    const { companyProfile, knowledgeAssessment } = req.body;

    if (!companyProfile) {
      return res.status(400).json({ error: 'Company profile is required' });
    }

    // Call the AI onboarding questions generator
    const options = {
      mode: 'json',
      pythonPath: 'python3',
      pythonOptions: ['-u'],
      scriptPath: './',
      args: ['generate-questions', JSON.stringify({ companyProfile, knowledgeAssessment })]
    };

    const result = await new Promise((resolve, reject) => {
      PythonShell.run('ai_onboarding_questions_generator.py', options, (err, results) => {
        if (err) {
          console.error('Python script error:', err);
          reject(err);
        } else {
          try {
            const data = JSON.parse(results[0]);
            resolve(data);
          } catch (parseError) {
            console.error('JSON parse error:', parseError);
            reject(parseError);
          }
        }
      });
    });

    res.json(result);
  } catch (error) {
    console.error('Error generating questions:', error);
    res.status(500).json({ 
      error: 'Failed to generate questions',
      details: error.message 
    });
  }
});

app.post('/api/ai-onboarding/generate-listings', async (req, res) => {
  try {
    const { companyProfile, answers } = req.body;

    if (!companyProfile || !answers) {
      return res.status(400).json({ error: 'Company profile and answers are required' });
    }

    // Call the AI onboarding questions generator to generate material listings
    const options = {
      mode: 'json',
      pythonPath: 'python3',
      pythonOptions: ['-u'],
      scriptPath: './',
      args: ['generate-listings', JSON.stringify({ companyProfile, answers })]
    };

    const result = await new Promise((resolve, reject) => {
      PythonShell.run('ai_onboarding_questions_generator.py', options, (err, results) => {
        if (err) {
          console.error('Python script error:', err);
          reject(err);
        } else {
          try {
            const data = JSON.parse(results[0]);
            resolve(data);
          } catch (parseError) {
            console.error('JSON parse error:', parseError);
            reject(parseError);
          }
        }
      });
    });

    res.json(result);
  } catch (error) {
    console.error('Error generating listings:', error);
    res.status(500).json({ 
      error: 'Failed to generate material listings',
      details: error.message 
    });
  }
});

module.exports = app;

const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'Industrial AI Marketplace API',
      version: '1.0.0',
      description: 'API documentation for the Industrial AI Marketplace backend.'
    },
    servers: [
      { url: 'http://localhost:3000', description: 'Development server' }
    ]
  },
  apis: ['./app.js'], // You can add more files for endpoint annotations
};

const swaggerSpec = swaggerJsdoc(swaggerOptions);
app.use('/api/docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec));

// =========================
// AI MATCHING ENDPOINTS
// =========================

/**
 * POST /api/ai-matching/semantic-search
 * Body: { material_id: string, top_k: number }
 * Returns: { matches: [...] }
 */
app.post('/api/ai-matching/semantic-search', async (req, res) => {
  try {
    const { material_id, top_k } = req.body;
    if (!material_id || typeof material_id !== 'string') {
      return sendResponse(res, { success: false, error: 'material_id is required' }, 400);
    }
    // TODO: Integrate with semantic search engine (Python or Node)
    // Placeholder: return empty matches
    return sendResponse(res, { success: true, data: { matches: [] } });
  } catch (error) {
    console.error('Semantic search error:', error);
    return sendResponse(res, { success: false, error: error.message }, 500);
  }
});

/**
 * POST /api/ai-matching/llm-analysis
 * Body: { material1_id: string, material2_id: string }
 * Returns: { analysis: ... }
 */
app.post('/api/ai-matching/llm-analysis', async (req, res) => {
  try {
    const { material1_id, material2_id } = req.body;
    if (!material1_id || !material2_id) {
      return sendResponse(res, { success: false, error: 'Both material1_id and material2_id are required' }, 400);
    }
    // TODO: Integrate with LLM analysis service
    // Placeholder: return dummy analysis
    return sendResponse(res, { success: true, data: { analysis: { compatibility: 'unknown', notes: 'LLM analysis not implemented' } } });
  } catch (error) {
    console.error('LLM analysis error:', error);
    return sendResponse(res, { success: false, error: error.message }, 500);
  }
});

/**
 * POST /api/ai-matching/gnn-scoring
 * Body: { material1_id: string, material2_id: string }
 * Returns: { score: number }
 */
app.post('/api/ai-matching/gnn-scoring', async (req, res) => {
  try {
    const { material1_id, material2_id } = req.body;
    if (!material1_id || !material2_id) {
      return sendResponse(res, { success: false, error: 'Both material1_id and material2_id are required' }, 400);
    }
    // TODO: Integrate with GNN scoring service
    // Placeholder: return dummy score
    return sendResponse(res, { success: true, data: { score: Math.random() } });
  } catch (error) {
    console.error('GNN scoring error:', error);
    return sendResponse(res, { success: false, error: error.message }, 500);
  }
});

/**
 * POST /api/ai-matching/comprehensive
 * Body: { material_id: string, match_type: string, preferences: object }
 * Returns: { matches: [...] }
 */
app.post('/api/ai-matching/comprehensive', async (req, res) => {
  try {
    const { material_id, match_type, preferences } = req.body;
    if (!material_id) {
      return sendResponse(res, { success: false, error: 'material_id is required' }, 400);
    }
    // TODO: Integrate with comprehensive matching engine
    // Placeholder: return empty matches
    return sendResponse(res, { success: true, data: { matches: [] } });
  } catch (error) {
    console.error('Comprehensive matching error:', error);
    return sendResponse(res, { success: false, error: error.message }, 500);
  }
});

/**
 * POST /api/ai-matching/multi-hop
 * Body: { material_id: string, max_hops: number }
 * Returns: { matches: [...] }
 */
app.post('/api/ai-matching/multi-hop', async (req, res) => {
  try {
    const { material_id, max_hops } = req.body;
    if (!material_id) {
      return sendResponse(res, { success: false, error: 'material_id is required' }, 400);
    }
    // TODO: Integrate with multi-hop symbiosis engine
    // Placeholder: return empty matches
    return sendResponse(res, { success: true, data: { matches: [] } });
  } catch (error) {
    console.error('Multi-hop matching error:', error);
    return sendResponse(res, { success: false, error: error.message }, 500);
  }
});

/**
 * GET /api/ai-matching/insights/:id
 * Returns: { insights: ... }
 */
app.get('/api/ai-matching/insights/:id', async (req, res) => {
  try {
    const { id } = req.params;
    if (!id) {
      return sendResponse(res, { success: false, error: 'Match ID is required' }, 400);
    }
    // TODO: Integrate with insights engine
    // Placeholder: return dummy insights
    return sendResponse(res, { success: true, data: { insights: { matchId: id, details: 'Insights not implemented' } } });
  } catch (error) {
    console.error('Insights error:', error);
    return sendResponse(res, { success: false, error: error.message }, 500);
  }
});
