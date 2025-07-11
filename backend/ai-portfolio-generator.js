// Load environment variables from .env file
require('dotenv').config();

const axios = require('axios');
const { supabase } = require('./supabase');

// Initialize DeepSeek R1 API
const DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions';
const DEEPSEEK_API_KEY = process.env.DEEPSEEK_API_KEY;

class AIPortfolioGenerator {
  constructor() {
    this.materialCategories = {
      'Manufacturing': ['metal_scrap', 'plastic_waste', 'chemical_waste', 'packaging_waste', 'wood_waste'],
      'Construction': ['construction_waste', 'metal_scrap', 'wood_waste', 'concrete_waste', 'packaging_waste'],
      'Food & Beverage': ['organic_waste', 'packaging_waste', 'water_waste', 'energy_waste', 'food_scraps'],
      'Chemicals': ['chemical_waste', 'hazardous_waste', 'water_waste', 'energy_waste', 'solvents'],
      'Electronics': ['electronic_waste', 'metal_scrap', 'plastic_waste', 'chemical_waste', 'batteries'],
      'Automotive': ['metal_scrap', 'plastic_waste', 'rubber_waste', 'chemical_waste', 'oil_waste'],
      'Textiles': ['fabric_scraps', 'dye_waste', 'packaging_waste', 'water_waste', 'fiber_waste'],
      'Pharmaceuticals': ['chemical_waste', 'biomedical_waste', 'packaging_waste', 'water_waste', 'expired_drugs'],
      'Oil & Gas': ['produced_water', 'drill_cuttings', 'chemical_waste', 'solid_waste', 'oil_sludge'],
      'Water Treatment': ['sludge', 'chemical_waste', 'filter_media', 'biosolids', 'residual_chemicals'],
      'Logistics': ['packaging_waste', 'vehicle_parts', 'fuel_waste', 'tyre_waste', 'scrap_metal'],
      'Healthcare': ['medical_waste', 'plastic_waste', 'chemical_waste', 'general_waste', 'pharmaceutical_waste'],
      'Tourism & Hospitality': ['food_waste', 'plastic_waste', 'paper_waste', 'glass_waste', 'organic_waste'],
      'Agriculture': ['organic_waste', 'water_waste', 'packaging_waste', 'chemical_waste', 'biomass'],
      'Mining': ['tailings', 'rock_waste', 'chemical_waste', 'water_waste', 'dust_waste']
    };

    this.requirementCategories = {
      'Manufacturing': ['raw_materials', 'energy', 'water', 'chemicals', 'packaging'],
      'Construction': ['building_materials', 'energy', 'water', 'equipment', 'transport'],
      'Food & Beverage': ['ingredients', 'energy', 'water', 'packaging', 'refrigeration'],
      'Chemicals': ['raw_chemicals', 'energy', 'water', 'catalysts', 'equipment'],
      'Electronics': ['components', 'energy', 'water', 'chemicals', 'packaging'],
      'Automotive': ['parts', 'energy', 'water', 'chemicals', 'transport'],
      'Textiles': ['fibers', 'energy', 'water', 'dyes', 'packaging'],
      'Pharmaceuticals': ['active_ingredients', 'energy', 'water', 'chemicals', 'packaging'],
      'Oil & Gas': ['equipment', 'energy', 'water', 'chemicals', 'transport'],
      'Water Treatment': ['chemicals', 'energy', 'equipment', 'filtration_media', 'transport'],
      'Logistics': ['fuel', 'energy', 'equipment', 'packaging', 'transport'],
      'Healthcare': ['medical_supplies', 'energy', 'water', 'chemicals', 'equipment'],
      'Tourism & Hospitality': ['food', 'energy', 'water', 'cleaning_supplies', 'equipment'],
      'Agriculture': ['seeds', 'energy', 'water', 'fertilizers', 'equipment'],
      'Mining': ['equipment', 'energy', 'water', 'chemicals', 'transport']
    };
  }

  async callDeepSeekR1(prompt, temperature = 0.3, maxTokens = 1000) {
    try {
      if (!DEEPSEEK_API_KEY) {
        throw new Error('DeepSeek API key not configured');
      }

      const response = await axios.post(DEEPSEEK_API_URL, {
        model: 'deepseek-reasoner', // Using DeepSeek R1 reasoning model
        messages: [
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: temperature,
        max_tokens: maxTokens,
        stream: false
      }, {
        headers: {
          'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
          'Content-Type': 'application/json'
        }
      });

      return response.data.choices[0].message.content;
    } catch (error) {
      console.error('DeepSeek R1 API Error:', error.response?.data || error.message);
      throw new Error(`DeepSeek R1 API call failed: ${error.message}`);
    }
  }

  async generatePortfolio(companyProfile) {
    try {
      console.log('🤖 Starting AI portfolio generation with DeepSeek R1 for:', companyProfile.company_name);

      // 1. Analyze company profile
      const analysis = await this.analyzeCompanyProfile(companyProfile);

      // 2. Generate material listings
      const materials = await this.generateMaterialListings(companyProfile, analysis);

      // 3. Generate symbiosis opportunities
      const opportunities = await this.generateSymbiosisOpportunities(companyProfile, analysis);

      // 4. Generate recommendations
      const recommendations = await this.generateRecommendations(companyProfile, analysis);

      // 5. Generate portfolio summary
      const summary = await this.generatePortfolioSummary(companyProfile, analysis, materials, opportunities);

      return {
        materials,
        opportunities,
        recommendations,
        summary,
        analysis
      };

    } catch (error) {
      console.error('❌ AI Portfolio generation failed:', error);
      throw error;
    }
  }

  async analyzeCompanyProfile(profile) {
    const prompt = `
You are an expert industrial symbiosis analyst. Analyze this company profile for industrial symbiosis opportunities using advanced reasoning:

Company: ${profile.company_name}
Industry: ${profile.industry}
Location: ${profile.location}
Products: ${profile.products_services}
Materials: ${profile.main_materials}
Processes: ${profile.production_processes}
Waste Streams: ${profile.current_waste_streams}
Waste Quantities: ${profile.waste_quantities}
Resource Needs: ${profile.resource_needs}
Sustainability Goals: ${profile.sustainability_goals?.join(', ')}

Using your reasoning capabilities, analyze the industrial symbiosis potential and provide detailed insights in JSON format:

{
  "waste_potential": "High/Medium/Low",
  "resource_efficiency_score": 1-10,
  "symbiosis_opportunities": ["opportunity1", "opportunity2"],
  "key_materials": ["material1", "material2"],
  "waste_categories": ["category1", "category2"],
  "potential_partners": ["partner_type1", "partner_type2"],
  "estimated_savings": "$X-$Y annually",
  "environmental_impact": "X tons CO2 reduction potential",
  "reasoning": "Detailed explanation of your analysis"
}
`;

    const response = await this.callDeepSeekR1(prompt, 0.3, 1000);
    
    try {
      return JSON.parse(response);
    } catch (parseError) {
      console.error('Failed to parse analysis JSON:', response);
      // Return fallback analysis
      return {
        waste_potential: "Medium",
        resource_efficiency_score: 5,
        symbiosis_opportunities: ["Material exchange", "Waste recycling"],
        key_materials: ["General materials"],
        waste_categories: ["General waste"],
        potential_partners: ["Manufacturing companies"],
        estimated_savings: "$10K-$50K annually",
        environmental_impact: "10-40 tons CO2 reduction potential",
        reasoning: "Analysis based on industry standards and waste management practices"
      };
    }
  }

  async generateMaterialListings(profile, analysis) {
    const materials = [];

    // Generate waste materials based on industry and waste streams
    const wasteStreams = profile.current_waste_streams.split(',').map(s => s.trim());
    const quantities = profile.waste_quantities.split(',').map(q => q.trim());

    for (let i = 0; i < wasteStreams.length; i++) {
      const waste = wasteStreams[i];
      const quantity = quantities[i] || 'Variable';

      if (waste) {
        materials.push({
          material_name: waste,
          quantity: quantity,
          unit: this.extractUnit(quantity),
          type: 'waste',
          description: `${waste} from ${profile.company_name} ${profile.industry} operations`,
          availability: 'Available',
          price_per_unit: this.calculatePrice(waste, 'waste'),
          quality_grade: this.assessQuality(waste),
          potential_uses: this.generatePotentialUses(waste),
          ai_generated: true,
          created_at: new Date().toISOString()
        });
      }
    }

    // Generate requirement materials based on industry
    const requirements = this.requirementCategories[profile.industry] || [];
    for (const req of requirements.slice(0, 3)) { // Limit to 3 requirements
      materials.push({
        material_name: req.replace('_', ' '),
        quantity: this.generateRealisticQuantity(req),
        unit: this.getUnitForMaterial(req),
        type: 'requirement',
        description: `${req.replace('_', ' ')} needed for ${profile.company_name} operations`,
        availability: 'Needed',
        price_per_unit: this.calculatePrice(req, 'requirement'),
        quality_grade: 'A',
        potential_sources: this.generatePotentialSources(req),
        ai_generated: true,
        created_at: new Date().toISOString()
      });
    }

    return materials;
  }

  async generateSymbiosisOpportunities(profile, analysis) {
    const opportunities = [];

    // Generate opportunities based on analysis
    const opportunityTypes = [
      'Material Exchange',
      'Waste Recycling',
      'Energy Sharing',
      'Water Reuse',
      'Joint Infrastructure',
      'Technology Transfer'
    ];

    for (const type of opportunityTypes.slice(0, 4)) { // Limit to 4 opportunities
      opportunities.push({
        title: `${type} Partnership`,
        description: `Potential ${type.toLowerCase()} partnership for ${profile.company_name}`,
        type: type,
        potential_partners: this.generatePotentialPartners(type, profile.industry),
        estimated_savings: this.generateSavingsEstimate(type),
        environmental_impact: this.generateEnvironmentalImpact(type),
        implementation_timeline: this.generateTimeline(type),
        difficulty_level: this.assessDifficulty(type),
        ai_generated: true,
        created_at: new Date().toISOString()
      });
    }

    return opportunities;
  }

  async generateRecommendations(profile, analysis) {
    const prompt = `
You are an expert industrial symbiosis consultant. Using advanced reasoning, provide 5 specific, actionable recommendations for this company's industrial symbiosis strategy:

Company: ${profile.company_name}
Industry: ${profile.industry}
Waste Streams: ${profile.current_waste_streams}
Analysis: ${JSON.stringify(analysis)}

Consider the company's specific context, industry best practices, and implementation feasibility. Provide detailed recommendations in JSON format:

[
  {
    "title": "Recommendation title",
    "description": "Detailed description with reasoning",
    "priority": "High/Medium/Low",
    "estimated_impact": "Specific impact description",
    "implementation_steps": ["step1", "step2", "step3"],
    "reasoning": "Why this recommendation is important for this specific company"
  }
]
`;

    const response = await this.callDeepSeekR1(prompt, 0.4, 1500);
    
    try {
      return JSON.parse(response);
    } catch (parseError) {
      console.error('Failed to parse recommendations JSON:', response);
      // Return fallback recommendations
      return [
        {
          title: "Implement waste segregation",
          description: "Separate different types of waste for better recycling opportunities",
          priority: "High",
          estimated_impact: "Improve recycling rates by 30%",
          implementation_steps: ["Audit current waste streams", "Install segregation bins", "Train staff"],
          reasoning: "Waste segregation is fundamental for effective recycling and material recovery"
        },
        {
          title: "Explore material exchange partnerships",
          description: "Find companies that can use your waste as raw materials",
          priority: "Medium",
          estimated_impact: "Reduce waste disposal costs by 40%",
          implementation_steps: ["Identify potential partners", "Assess material compatibility", "Establish partnerships"],
          reasoning: "Material exchange creates mutual benefits and reduces environmental impact"
        }
      ];
    }
  }

  async generatePortfolioSummary(profile, analysis, materials, opportunities) {
    const prompt = `
You are an expert industrial symbiosis analyst. Create a comprehensive, professional portfolio summary for this company's industrial symbiosis profile using advanced reasoning:

Company: ${profile.company_name}
Industry: ${profile.industry}
Analysis: ${JSON.stringify(analysis)}
Materials Count: ${materials.length}
Opportunities Count: ${opportunities.length}

Write a compelling, business-focused summary (2-3 paragraphs) that highlights:
- Company's unique symbiosis potential based on their specific profile
- Key opportunities identified with reasoning
- Estimated benefits and competitive advantages
- Strategic next steps for implementation

Use your reasoning capabilities to provide insights that go beyond basic analysis.
`;

    const response = await this.callDeepSeekR1(prompt, 0.5, 500);
    return response;
  }

  // Helper methods
  extractUnit(quantity) {
    if (quantity.includes('kg')) return 'kg';
    if (quantity.includes('tons')) return 'tons';
    if (quantity.includes('liters')) return 'liters';
    if (quantity.includes('m³')) return 'cubic meters';
    return 'units';
  }

  calculatePrice(material, type) {
    const basePrice = type === 'waste' ? 0.5 : 10;
    const variation = Math.random() * 0.8 + 0.2; // 20% to 100% variation
    return Math.round(basePrice * variation * 100) / 100;
  }

  assessQuality(material) {
    const grades = ['A', 'B', 'C'];
    return grades[Math.floor(Math.random() * grades.length)];
  }

  generatePotentialUses(material) {
    const uses = {
      'plastic': ['Recycling', 'Energy recovery', 'Material recovery'],
      'metal': ['Recycling', 'Refining', 'Manufacturing'],
      'organic': ['Composting', 'Biogas', 'Animal feed'],
      'chemical': ['Treatment', 'Recovery', 'Neutralization'],
      'water': ['Treatment', 'Reuse', 'Cooling']
    };

    for (const [type, useList] of Object.entries(uses)) {
      if (material.toLowerCase().includes(type)) {
        return useList;
      }
    }
    return ['Recycling', 'Energy recovery', 'Material recovery'];
  }

  generateRealisticQuantity(material) {
    const quantities = [100, 500, 1000, 2000, 5000, 10000];
    return quantities[Math.floor(Math.random() * quantities.length)];
  }

  getUnitForMaterial(material) {
    if (material.includes('energy')) return 'kWh';
    if (material.includes('water')) return 'liters';
    return 'kg';
  }

  generatePotentialSources(material) {
    return ['Local suppliers', 'Regional markets', 'Industrial partners'];
  }

  generatePotentialPartners(type, industry) {
    const partners = {
      'Material Exchange': ['Manufacturing companies', 'Recycling facilities'],
      'Waste Recycling': ['Waste management companies', 'Recycling centers'],
      'Energy Sharing': ['Energy companies', 'Industrial facilities'],
      'Water Reuse': ['Water treatment plants', 'Industrial facilities']
    };
    return partners[type] || ['Industrial partners'];
  }

  generateSavingsEstimate(type) {
    const ranges = {
      'Material Exchange': '$10K-$50K',
      'Waste Recycling': '$5K-$25K',
      'Energy Sharing': '$20K-$100K',
      'Water Reuse': '$15K-$75K'
    };
    return ranges[type] || '$10K-$50K';
  }

  generateEnvironmentalImpact(type) {
    const impacts = {
      'Material Exchange': '5-20 tons CO2 reduction',
      'Waste Recycling': '10-40 tons CO2 reduction',
      'Energy Sharing': '20-80 tons CO2 reduction',
      'Water Reuse': '15-60 tons CO2 reduction'
    };
    return impacts[type] || '10-40 tons CO2 reduction';
  }

  generateTimeline(type) {
    const timelines = {
      'Material Exchange': '1-3 months',
      'Waste Recycling': '2-6 months',
      'Energy Sharing': '3-12 months',
      'Water Reuse': '2-8 months'
    };
    return timelines[type] || '3-6 months';
  }

  assessDifficulty(type) {
    const difficulties = {
      'Material Exchange': 'Medium',
      'Waste Recycling': 'Low',
      'Energy Sharing': 'High',
      'Water Reuse': 'Medium'
    };
    return difficulties[type] || 'Medium';
  }
}

module.exports = { AIPortfolioGenerator }; 