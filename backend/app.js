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

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:5173',
  credentials: true
}));

// Body parsing middleware - MUST come before routes
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

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

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version || '1.0.0'
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
    
    res.json({ success: true });
  } catch (error) {
    console.error('Logging error:', error);
    res.status(500).json({ error: 'Failed to log message' });
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

// Helper function to run Python scripts
function runPythonScript(scriptPath, data) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [scriptPath, JSON.stringify(data)]);
        
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
    });
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

// REAL AI MATCHING ENDPOINT
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
        const { data, error } = await supabaseClient
            .from('companies')
            .insert([req.body])
            .select();
        
        if (error) throw error;
        res.json(data[0]);
    } catch (error) {
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
        
        // Call comprehensive match analyzer
        const analysisResult = await runPythonScript('comprehensive_match_analyzer.py', {
            action: 'analyze_match_comprehensive',
            buyer_data: buyer_data,
            seller_data: seller_data,
            match_data: match_data
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

// Start the server
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
  console.log(`📡 API available at http://localhost:${PORT}/api`);
  console.log(`🏥 Health check at http://localhost:${PORT}/api/health`);
});
