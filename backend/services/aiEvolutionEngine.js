const axios = require('axios');
const { supabase } = require('../supabase');

class AIEvolutionEngine {
  constructor() {
    this.deepseekApiKey = process.env.DEEPSEEK_API_KEY;
    this.deepseekModel = 'deepseek-chat';
    this.baseUrl = 'https://api.deepseek.com/v1/chat/completions';
    
    // Structured prompt templates
    this.promptTemplates = {
      materialAnalysis: this.getMaterialAnalysisPrompt(),
      matchGeneration: this.getMatchGenerationPrompt(),
      portfolioCreation: this.getPortfolioCreationPrompt(),
      sustainabilityAssessment: this.getSustainabilityAssessmentPrompt(),
      shippingOptimization: this.getShippingOptimizationPrompt()
    };
    
    // Feedback collection system
    this.feedbackCategories = [
      'match_accuracy',
      'material_compatibility',
      'savings_estimation',
      'implementation_feasibility',
      'environmental_impact',
      'overall_quality'
    ];
  }

  /**
   * Execute AI analysis with structured prompts and feedback collection
   */
  async executeAIAnalysis(analysisType, inputData, context = {}) {
    try {
      console.log(`🤖 Executing AI analysis: ${analysisType}`);
      
      // Get the appropriate prompt template
      const promptTemplate = this.promptTemplates[analysisType];
      if (!promptTemplate) {
        throw new Error(`Unknown analysis type: ${analysisType}`);
      }

      // Prepare the prompt with input data
      const prompt = this.preparePrompt(promptTemplate, inputData, context);
      
      // Execute the AI call
      const response = await this.callDeepSeekAPI(prompt);
      
      // Parse and structure the response
      const structuredResponse = this.parseAIResponse(response, analysisType);
      
      // Log the interaction for feedback collection
      await this.logAIInteraction(analysisType, inputData, structuredResponse, context);
      
      return {
        success: true,
        analysis_type: analysisType,
        result: structuredResponse,
        confidence_score: this.calculateConfidenceScore(structuredResponse),
        feedback_id: await this.generateFeedbackId(analysisType, inputData)
      };
      
    } catch (error) {
      console.error(`❌ AI analysis error (${analysisType}):`, error);
      
      // Log the error for analysis
      await this.logAIError(analysisType, inputData, error, context);
      
      return {
        success: false,
        analysis_type: analysisType,
        error: error.message,
        fallback_result: this.getFallbackResult(analysisType, inputData)
      };
    }
  }

  /**
   * Material analysis prompt template
   */
  getMaterialAnalysisPrompt() {
    return {
      system: `You are an expert industrial materials scientist specializing in circular economy and industrial symbiosis. Your task is to analyze industrial materials and identify their potential for reuse, recycling, and symbiotic partnerships.

CRITICAL REQUIREMENTS:
1. Provide analysis in structured JSON format
2. Include material properties, compatibility scores, and sustainability metrics
3. Identify potential applications and partner industries
4. Calculate environmental impact and economic value
5. Suggest optimization strategies

RESPONSE FORMAT:
{
  "material_properties": {
    "density_kg_per_m3": number,
    "melting_point_celsius": number,
    "chemical_composition": string,
    "hazard_classification": string,
    "recycling_code": string
  },
  "sustainability_metrics": {
    "circular_economy_potential": number (0-100),
    "environmental_impact_score": number (0-100),
    "carbon_footprint_reduction": number (kg CO2e),
    "water_savings": number (liters)
  },
  "compatibility_analysis": {
    "compatible_materials": [string],
    "incompatible_materials": [string],
    "processing_requirements": [string],
    "quality_considerations": [string]
  },
  "potential_applications": [
    {
      "industry": string,
      "application": string,
      "value_proposition": string,
      "implementation_complexity": "low|medium|high"
    }
  ],
  "economic_analysis": {
    "estimated_value_per_ton": number,
    "processing_cost_per_ton": number,
    "market_demand": "low|medium|high",
    "price_volatility": "low|medium|high"
  },
  "recommendations": [
    {
      "category": string,
      "description": string,
      "priority": "high|medium|low",
      "expected_impact": string
    }
  ]
}`,
      user: (inputData) => `Analyze this industrial material for circular economy potential:

Material Name: ${inputData.material_name}
Category: ${inputData.category}
Current State: ${inputData.state}
Quantity: ${inputData.quantity} ${inputData.unit}
Quality Grade: ${inputData.quality_grade}
Description: ${inputData.description}
Company Industry: ${inputData.company_industry}
Location: ${inputData.location}

Provide a comprehensive analysis focusing on:
1. Material properties and characteristics
2. Sustainability and environmental impact
3. Compatibility with other materials and industries
4. Potential applications and value propositions
5. Economic feasibility and market considerations
6. Specific recommendations for optimization`
    };
  }

  /**
   * Match generation prompt template
   */
  getMatchGenerationPrompt() {
    return {
      system: `You are an expert industrial symbiosis matchmaker with deep knowledge of circular economy principles and industrial processes. Your task is to identify optimal symbiotic partnerships between companies based on their material flows, waste streams, and resource needs.

CRITICAL REQUIREMENTS:
1. Analyze material compatibility and waste-to-input potential
2. Consider geographic proximity and logistics feasibility
3. Calculate economic and environmental benefits
4. Assess implementation complexity and risks
5. Provide detailed match rationale and recommendations

RESPONSE FORMAT:
{
  "match_analysis": {
    "overall_score": number (0-100),
    "match_type": "material_exchange|waste_recycling|energy_sharing|water_reuse|logistics_sharing",
    "confidence_level": "high|medium|low"
  },
  "compatibility_scores": {
    "material_compatibility": number (0-100),
    "waste_synergy": number (0-100),
    "energy_synergy": number (0-100),
    "geographic_proximity": number (0-100),
    "logistics_feasibility": number (0-100)
  },
  "economic_analysis": {
    "potential_savings_company_a": number,
    "potential_savings_company_b": number,
    "implementation_cost": number,
    "payback_period_months": number,
    "roi_percentage": number
  },
  "environmental_impact": {
    "carbon_reduction_kg_co2e": number,
    "waste_diverted_kg": number,
    "water_saved_liters": number,
    "energy_saved_kwh": number
  },
  "implementation_roadmap": [
    {
      "phase": string,
      "timeline_weeks": number,
      "activities": [string],
      "resources_required": [string],
      "success_metrics": [string]
    }
  ],
  "risk_assessment": {
    "technical_risks": [string],
    "logistical_risks": [string],
    "regulatory_risks": [string],
    "mitigation_strategies": [string]
  },
  "recommendations": [
    {
      "category": string,
      "description": string,
      "priority": "high|medium|low",
      "expected_outcome": string
    }
  ]
}`,
      user: (inputData) => `Generate symbiotic match analysis between these companies:

COMPANY A:
- Name: ${inputData.company_a.name}
- Industry: ${inputData.company_a.industry}
- Location: ${inputData.company_a.location}
- Materials: ${inputData.company_a.materials.join(', ')}
- Waste Streams: ${inputData.company_a.waste_streams.join(', ')}
- Resource Needs: ${inputData.company_a.resource_needs.join(', ')}

COMPANY B:
- Name: ${inputData.company_b.name}
- Industry: ${inputData.company_b.industry}
- Location: ${inputData.company_b.location}
- Materials: ${inputData.company_b.materials.join(', ')}
- Waste Streams: ${inputData.company_b.waste_streams.join(', ')}
- Resource Needs: ${inputData.company_b.resource_needs.join(', ')}

MATERIAL CONTEXT:
- Material A: ${inputData.material_a.material_name} (${inputData.material_a.type})
- Material B: ${inputData.material_b.material_name} (${inputData.material_b.type})

Analyze the potential for symbiotic partnership focusing on material exchange, waste recycling, and resource optimization.`
    };
  }

  /**
   * Portfolio creation prompt template
   */
  getPortfolioCreationPrompt() {
    return {
      system: `You are an expert industrial portfolio analyst specializing in circular economy optimization. Your task is to create comprehensive industrial symbiosis portfolios that maximize resource efficiency, cost savings, and environmental benefits.

CRITICAL REQUIREMENTS:
1. Analyze current material flows and waste streams
2. Identify optimization opportunities and potential partnerships
3. Calculate economic and environmental impacts
4. Provide implementation strategies and timelines
5. Include risk assessment and mitigation plans

RESPONSE FORMAT:
{
  "portfolio_summary": {
    "total_potential_savings": number,
    "carbon_reduction_potential": number,
    "waste_reduction_potential": number,
    "partnership_opportunities": number,
    "implementation_timeline_months": number
  },
  "material_analysis": [
    {
      "material_name": string,
      "current_status": "waste|byproduct|resource",
      "optimization_potential": number (0-100),
      "potential_applications": [string],
      "estimated_value": number,
      "processing_requirements": [string]
    }
  ],
  "partnership_opportunities": [
    {
      "partner_industry": string,
      "synergy_type": string,
      "potential_savings": number,
      "implementation_complexity": "low|medium|high",
      "timeline_months": number
    }
  ],
  "implementation_roadmap": [
    {
      "phase": string,
      "duration_weeks": number,
      "key_activities": [string],
      "success_metrics": [string],
      "resource_requirements": [string]
    }
  ],
  "risk_mitigation": {
    "technical_risks": [string],
    "market_risks": [string],
    "regulatory_risks": [string],
    "mitigation_strategies": [string]
  },
  "success_metrics": {
    "financial_metrics": [string],
    "environmental_metrics": [string],
    "operational_metrics": [string],
    "sustainability_metrics": [string]
  }
}`,
      user: (inputData) => `Create a comprehensive industrial symbiosis portfolio for this company:

COMPANY PROFILE:
- Name: ${inputData.company_name}
- Industry: ${inputData.industry}
- Location: ${inputData.location}
- Employee Count: ${inputData.employee_count}
- Annual Revenue: ${inputData.annual_revenue}

CURRENT OPERATIONS:
- Products: ${inputData.products}
- Main Materials: ${inputData.main_materials}
- Production Volume: ${inputData.production_volume}
- Process Description: ${inputData.process_description}

WASTE & RESOURCES:
- Current Waste Management: ${inputData.current_waste_management}
- Waste Quantity: ${inputData.waste_quantity} ${inputData.waste_unit}
- Resource Needs: ${inputData.resource_needs}
- Energy Consumption: ${inputData.energy_consumption}

SUSTAINABILITY GOALS: ${inputData.sustainability_goals.join(', ')}

Create a comprehensive portfolio that maximizes circular economy opportunities and identifies optimal symbiotic partnerships.`
    };
  }

  /**
   * Sustainability assessment prompt template
   */
  getSustainabilityAssessmentPrompt() {
    return {
      system: `You are an expert sustainability analyst specializing in industrial environmental impact assessment and circular economy optimization. Your task is to evaluate the sustainability performance of industrial operations and provide actionable recommendations for improvement.

CRITICAL REQUIREMENTS:
1. Assess current environmental impact across multiple dimensions
2. Identify improvement opportunities and optimization strategies
3. Calculate potential environmental and economic benefits
4. Provide prioritized action plans with timelines
5. Include compliance and certification recommendations

RESPONSE FORMAT:
{
  "sustainability_score": {
    "overall_score": number (0-100),
    "environmental_score": number (0-100),
    "social_score": number (0-100),
    "economic_score": number (0-100),
    "circular_economy_score": number (0-100)
  },
  "environmental_impact": {
    "carbon_footprint_kg_co2e": number,
    "water_consumption_liters": number,
    "waste_generated_kg": number,
    "energy_consumption_kwh": number,
    "air_emissions_kg": number
  },
  "improvement_opportunities": [
    {
      "category": string,
      "description": string,
      "potential_impact": number (0-100),
      "implementation_cost": number,
      "payback_period_months": number,
      "priority": "high|medium|low"
    }
  ],
  "circular_economy_metrics": {
    "material_efficiency": number (0-100),
    "waste_reduction_potential": number (0-100),
    "renewable_energy_usage": number (0-100),
    "water_recycling_rate": number (0-100),
    "supply_chain_sustainability": number (0-100)
  },
  "compliance_status": {
    "current_certifications": [string],
    "recommended_certifications": [string],
    "regulatory_requirements": [string],
    "compliance_gaps": [string]
  },
  "action_plan": [
    {
      "phase": string,
      "timeline_months": number,
      "actions": [string],
      "expected_outcomes": [string],
      "success_metrics": [string]
    }
  ]
}`,
      user: (inputData) => `Conduct a comprehensive sustainability assessment for this industrial operation:

COMPANY INFORMATION:
- Name: ${inputData.company_name}
- Industry: ${inputData.industry}
- Location: ${inputData.location}
- Employee Count: ${inputData.employee_count}

OPERATIONAL DATA:
- Annual Revenue: ${inputData.annual_revenue}
- Energy Consumption: ${inputData.energy_consumption}
- Water Usage: ${inputData.water_usage}
- Waste Generation: ${inputData.waste_generation}
- Carbon Footprint: ${inputData.carbon_footprint}

CURRENT PRACTICES:
- Waste Management: ${inputData.waste_management}
- Recycling Practices: ${inputData.recycling_practices}
- Environmental Certifications: ${inputData.certifications}
- Sustainability Goals: ${inputData.sustainability_goals.join(', ')}

Provide a comprehensive sustainability assessment with actionable recommendations for improvement.`
    };
  }

  /**
   * Shipping optimization prompt template
   */
  getShippingOptimizationPrompt() {
    return {
      system: `You are an expert logistics and shipping optimization specialist focusing on industrial material transport and circular economy logistics. Your task is to optimize shipping and logistics for industrial material exchanges while minimizing environmental impact and costs.

CRITICAL REQUIREMENTS:
1. Analyze material properties and shipping requirements
2. Optimize packaging, routing, and carrier selection
3. Calculate costs, emissions, and delivery times
4. Identify consolidation and backhaul opportunities
5. Provide sustainability-focused logistics solutions

RESPONSE FORMAT:
{
  "shipping_optimization": {
    "optimal_carrier": string,
    "service_level": string,
    "estimated_cost": number,
    "delivery_time_days": number,
    "carbon_footprint_kg_co2e": number
  },
  "packaging_analysis": {
    "recommended_packaging": string,
    "packaging_cost": number,
    "material_efficiency": number (0-100),
    "reusability_score": number (0-100),
    "environmental_impact": "low|medium|high"
  },
  "route_optimization": {
    "optimal_route": string,
    "distance_km": number,
    "fuel_efficiency": number (0-100),
    "backhaul_opportunities": [string],
    "consolidation_potential": number (0-100)
  },
  "cost_breakdown": {
    "transportation_cost": number,
    "packaging_cost": number,
    "handling_cost": number,
    "insurance_cost": number,
    "total_cost": number
  },
  "sustainability_metrics": {
    "carbon_intensity": number (kg CO2e per ton-km),
    "fuel_efficiency": number (liters per 100km),
    "renewable_energy_usage": number (0-100),
    "waste_minimization": number (0-100)
  },
  "recommendations": [
    {
      "category": string,
      "description": string,
      "potential_savings": number,
      "implementation_complexity": "low|medium|high",
      "environmental_benefit": string
    }
  ]
}`,
      user: (inputData) => `Optimize shipping and logistics for this industrial material exchange:

MATERIAL DETAILS:
- Material: ${inputData.material_name}
- Type: ${inputData.material_type}
- Quantity: ${inputData.quantity} ${inputData.unit}
- Weight: ${inputData.weight_kg} kg
- Volume: ${inputData.volume_cubic_meters} m³
- Special Handling: ${inputData.special_handling.join(', ')}

SHIPPING REQUIREMENTS:
- From: ${inputData.from_location}
- To: ${inputData.to_location}
- Delivery Timeline: ${inputData.delivery_timeline}
- Budget Constraint: ${inputData.budget_constraint}

ENVIRONMENTAL CONSIDERATIONS:
- Carbon Reduction Target: ${inputData.carbon_reduction_target}
- Sustainability Priority: ${inputData.sustainability_priority}

Provide optimized shipping and logistics solutions that balance cost, speed, and environmental impact.`
    };
  }

  /**
   * Prepare prompt with input data and context
   */
  preparePrompt(template, inputData, context) {
    const systemPrompt = template.system;
    const userPrompt = template.user(inputData);
    
    return {
      model: this.deepseekModel,
      messages: [
        {
          role: 'system',
          content: systemPrompt
        },
        {
          role: 'user',
          content: userPrompt
        }
      ],
      temperature: 0.3,
      max_tokens: 4000,
      response_format: { type: 'json_object' }
    };
  }

  /**
   * Call DeepSeek API
   */
  async callDeepSeekAPI(prompt) {
    try {
      const response = await axios({
        method: 'POST',
        url: this.baseUrl,
        headers: {
          'Authorization': `Bearer ${this.deepseekApiKey}`,
          'Content-Type': 'application/json'
        },
        data: prompt,
        timeout: 60000
      });

      return response.data.choices[0].message.content;
    } catch (error) {
      console.error('DeepSeek API error:', error.response?.data || error.message);
      throw new Error(`DeepSeek API error: ${error.response?.data?.error?.message || error.message}`);
    }
  }

  /**
   * Parse AI response into structured format
   */
  parseAIResponse(response, analysisType) {
    try {
      // Try to parse as JSON
      const parsed = JSON.parse(response);
      
      // Validate the response structure based on analysis type
      return this.validateResponseStructure(parsed, analysisType);
    } catch (error) {
      console.error('Response parsing error:', error);
      throw new Error('Invalid JSON response from AI');
    }
  }

  /**
   * Validate response structure
   */
  validateResponseStructure(response, analysisType) {
    // Add validation logic based on analysis type
    // For now, return the response as-is
    return response;
  }

  /**
   * Calculate confidence score
   */
  calculateConfidenceScore(response) {
    // Implement confidence scoring logic based on response quality
    // For now, return a default score
    return 0.85;
  }

  /**
   * Log AI interaction for feedback collection
   */
  async logAIInteraction(analysisType, inputData, response, context) {
    try {
      await supabase.from('ai_interactions').insert({
        analysis_type: analysisType,
        input_data: inputData,
        response_data: response,
        context: context,
        confidence_score: this.calculateConfidenceScore(response),
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to log AI interaction:', error);
    }
  }

  /**
   * Log AI errors for analysis
   */
  async logAIError(analysisType, inputData, error, context) {
    try {
      await supabase.from('ai_errors').insert({
        analysis_type: analysisType,
        input_data: inputData,
        error_message: error.message,
        error_stack: error.stack,
        context: context,
        timestamp: new Date().toISOString()
      });
    } catch (logError) {
      console.error('Failed to log AI error:', logError);
    }
  }

  /**
   * Generate feedback ID for user feedback collection
   */
  async generateFeedbackId(analysisType, inputData) {
    const feedbackId = `feedback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    try {
      await supabase.from('feedback_requests').insert({
        feedback_id: feedbackId,
        analysis_type: analysisType,
        input_data: inputData,
        status: 'pending',
        created_at: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to create feedback request:', error);
    }
    
    return feedbackId;
  }

  /**
   * Collect user feedback for AI improvement
   */
  async collectUserFeedback(feedbackId, feedbackData) {
    try {
      const { data, error } = await supabase
        .from('user_feedback')
        .insert({
          feedback_id: feedbackId,
          user_ratings: feedbackData.ratings,
          qualitative_feedback: feedbackData.comments,
          improvement_suggestions: feedbackData.suggestions,
          overall_satisfaction: feedbackData.satisfaction,
          timestamp: new Date().toISOString()
        });

      if (error) throw error;

      // Update feedback request status
      await supabase
        .from('feedback_requests')
        .update({ status: 'completed' })
        .eq('feedback_id', feedbackId);

      return { success: true, feedback_id: feedbackId };
    } catch (error) {
      console.error('Failed to collect user feedback:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Real AI analysis required - no fallbacks allowed
   */
  getFallbackResult(analysisType, inputData) {
    throw new Error(`❌ Real AI analysis required for ${analysisType}. API or model failure detected.`);
  }

  /**
   * Get feedback statistics for AI improvement
   */
  async getFeedbackStats() {
    try {
      const { data: feedback } = await supabase
        .from('user_feedback')
        .select('*');

      if (!feedback || feedback.length === 0) {
        return { total_feedback: 0, average_satisfaction: 0 };
      }

      const totalFeedback = feedback.length;
      const averageSatisfaction = feedback.reduce((sum, f) => sum + f.overall_satisfaction, 0) / totalFeedback;

      return {
        total_feedback: totalFeedback,
        average_satisfaction: averageSatisfaction,
        feedback_distribution: this.calculateFeedbackDistribution(feedback)
      };
    } catch (error) {
      console.error('Error getting feedback stats:', error);
      return { total_feedback: 0, average_satisfaction: 0 };
    }
  }

  /**
   * Calculate feedback distribution
   */
  calculateFeedbackDistribution(feedback) {
    const distribution = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
    
    feedback.forEach(f => {
      const rating = Math.round(f.overall_satisfaction);
      if (distribution[rating] !== undefined) {
        distribution[rating]++;
      }
    });

    return distribution;
  }
}

module.exports = new AIEvolutionEngine(); 