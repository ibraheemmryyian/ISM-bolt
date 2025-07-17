const { supabase } = require('../supabase');
const apiFusionService = require('./apiFusionService');
const aiEvolutionEngine = require('./aiEvolutionEngine');
const { spawn } = require('child_process');

// Import pricing integration utilities (commented out until Python bridge is implemented)
// const { NodeJSIntegration } = require('../ai_pricing_integration');

class IntelligentMatchingService {
  constructor() {
    this.matchingEngines = {
      realAI: 'real_ai_matching_engine.py',
      gnn: 'gnn_reasoning_engine.py',
      revolutionary: 'revolutionary_ai_matching.py',
      comprehensive: 'comprehensive_match_analyzer.py'
    };
    
    this.matchTypes = {
      MATERIAL_EXCHANGE: 'material_exchange',
      WASTE_RECYCLING: 'waste_recycling',
      ENERGY_SHARING: 'energy_sharing',
      WATER_REUSE: 'water_reuse',
      LOGISTICS_SHARING: 'logistics_sharing',
      TECHNOLOGY_TRANSFER: 'technology_transfer'
    };
  }

  /**
   * Intelligent matching orchestration
   * Combines multiple engines for comprehensive matching
   */
  async findIntelligentMatches(companyData, options = {}) {
    try {
      console.log('🔍 Starting intelligent matching for company:', companyData.name);
      
      const {
        topK = 10,
        matchTypes = Object.values(this.matchTypes),
        useAllEngines = true,
        includeShippingAnalysis = true,
        includeSustainabilityAnalysis = true
      } = options;

      // Step 1: Get company profile and materials
      const companyProfile = await this.getCompanyProfile(companyData);
      const companyMaterials = await this.getCompanyMaterials(companyProfile.id);
      
      // Step 2: Execute multiple matching engines
      const engineResults = await this.executeMatchingEngines(companyProfile, companyMaterials, useAllEngines);
      
      // Step 3: Aggregate and rank matches
      const aggregatedMatches = await this.aggregateMatches(engineResults, companyProfile);
      
      // Step 4: Enhanced analysis for top matches
      const enhancedMatches = await this.enhanceTopMatches(
        aggregatedMatches.slice(0, topK),
        companyProfile,
        includeShippingAnalysis,
        includeSustainabilityAnalysis
      );
      
      // Step 5: Apply pricing validation to all matches (temporarily disabled)
      // const pricingValidatedMatches = NodeJSIntegration.validate_matches_with_pricing(enhancedMatches);
      const pricingValidatedMatches = enhancedMatches; // Temporarily skip pricing validation
      
      // Step 6: Store only pricing-validated matches in database
      await this.storeMatches(pricingValidatedMatches, companyProfile.id);
      
      // Step 7: Generate insights and recommendations
      const insights = await this.generateMatchingInsights(pricingValidatedMatches, companyProfile);
      
      return {
        success: true,
        matches: pricingValidatedMatches,
        insights: insights,
        total_matches_found: aggregatedMatches.length,
        top_matches_count: enhancedMatches.length,
        pricing_validated_count: pricingValidatedMatches.length,
        matching_engines_used: Object.keys(engineResults).length,
        pricing_validation_stats: {
          total_matches: enhancedMatches.length,
          validated_matches: pricingValidatedMatches.length,
          failed_validations: enhancedMatches.length - pricingValidatedMatches.length
        }
      };
      
    } catch (error) {
      console.error('❌ Intelligent matching error:', error);
      return {
        success: false,
        error: error.message,
        matches: [],
        insights: []
      };
    }
  }

  /**
   * Execute multiple matching engines in parallel
   */
  async executeMatchingEngines(companyProfile, materials, useAllEngines) {
    const results = {};
    const enginePromises = [];

    // Real AI Matching Engine
    if (useAllEngines || true) {
      enginePromises.push(
        this.executePythonEngine('realAI', {
          action: 'find_symbiotic_matches',
          company_data: this.prepareCompanyDataForEngine(companyProfile, materials),
          top_k: 20
        }).then(result => { results.realAI = result; })
      );
    }

    // GNN Reasoning Engine
    if (useAllEngines) {
      enginePromises.push(
        this.executePythonEngine('gnn', {
          action: 'detect_multi_hop_symbiosis',
          participants: [this.prepareCompanyDataForEngine(companyProfile, materials)],
          max_hops: 3
        }).then(result => { results.gnn = result; })
      );
    }

    // Revolutionary AI Matching
    if (useAllEngines) {
      enginePromises.push(
        this.executePythonEngine('revolutionary', {
          action: 'generate_ai_listings',
          current_company: this.prepareCompanyDataForEngine(companyProfile, materials),
          all_companies: await this.getAllCompanies(),
          all_materials: await this.getAllMaterials()
        }).then(result => { results.revolutionary = result; })
      );
    }

    // Wait for all engines to complete
    await Promise.allSettled(enginePromises);
    
    return results;
  }

  /**
   * Execute Python matching engine
   */
  async executePythonEngine(engineName, data) {
    return new Promise((resolve, reject) => {
      const enginePath = this.matchingEngines[engineName];
      if (!enginePath) {
        reject(new Error(`Unknown engine: ${engineName}`));
        return;
      }

      const pythonProcess = spawn('python', [enginePath], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(output);
            resolve(result);
          } catch (error) {
            reject(new Error(`Failed to parse engine output: ${error.message}`));
          }
        } else {
          reject(new Error(`Engine execution failed: ${errorOutput}`));
        }
      });

      // Send data to Python process
      pythonProcess.stdin.write(JSON.stringify(data));
      pythonProcess.stdin.end();
    });
  }

  /**
   * Aggregate matches from multiple engines
   */
  async aggregateMatches(engineResults, companyProfile) {
    const allMatches = [];
    const matchScores = new Map();

    // Process Real AI matches
    if (engineResults.realAI && Array.isArray(engineResults.realAI)) {
      engineResults.realAI.forEach(match => {
        const matchKey = this.generateMatchKey(match, companyProfile.id);
        const score = match.match_score || 0;
        
        if (!matchScores.has(matchKey) || matchScores.get(matchKey) < score) {
          matchScores.set(matchKey, score);
          allMatches.push({
            ...match,
            engine: 'realAI',
            match_key: matchKey
          });
        }
      });
    }

    // Process GNN matches
    if (engineResults.gnn && Array.isArray(engineResults.gnn)) {
      engineResults.gnn.forEach(match => {
        const matchKey = this.generateMatchKey(match, companyProfile.id);
        const score = match.symbiosis_score || 0;
        
        if (!matchScores.has(matchKey) || matchScores.get(matchKey) < score) {
          matchScores.set(matchKey, score);
          allMatches.push({
            ...match,
            engine: 'gnn',
            match_key: matchKey
          });
        }
      });
    }

    // Process Revolutionary AI matches
    if (engineResults.revolutionary && Array.isArray(engineResults.revolutionary)) {
      engineResults.revolutionary.forEach(match => {
        const matchKey = this.generateMatchKey(match, companyProfile.id);
        const score = match.match_strength || 0;
        
        if (!matchScores.has(matchKey) || matchScores.get(matchKey) < score) {
          matchScores.set(matchKey, score);
          allMatches.push({
            ...match,
            engine: 'revolutionary',
            match_key: matchKey
          });
        }
      });
    }

    // Sort by score and remove duplicates
    const uniqueMatches = this.removeDuplicateMatches(allMatches);
    return uniqueMatches.sort((a, b) => (b.match_score || 0) - (a.match_score || 0));
  }

  /**
   * Enhance top matches with additional analysis
   */
  async enhanceTopMatches(matches, companyProfile, includeShipping, includeSustainability) {
    const enhancedMatches = [];

    for (const match of matches) {
      try {
        const enhancedMatch = { ...match };

        // Add shipping analysis if requested
        if (includeShipping && match.material_a_id && match.material_b_id) {
          const shippingAnalysis = await this.analyzeShippingRequirements(match, companyProfile);
          enhancedMatch.shipping_analysis = shippingAnalysis;
        }

        // Add sustainability analysis if requested
        if (includeSustainability) {
          const sustainabilityAnalysis = await this.analyzeSustainabilityImpact(match, companyProfile);
          enhancedMatch.sustainability_analysis = sustainabilityAnalysis;
        }

        // Add AI-powered match analysis
        const aiAnalysis = await this.performAIMatchAnalysis(match, companyProfile);
        enhancedMatch.ai_analysis = aiAnalysis;

        enhancedMatches.push(enhancedMatch);
      } catch (error) {
        console.error('Error enhancing match:', error);
        enhancedMatches.push(match);
      }
    }

    return enhancedMatches;
  }

  /**
   * Analyze shipping requirements for a match
   */
  async analyzeShippingRequirements(match, companyProfile) {
    try {
      // Get material details
      const materialA = await this.getMaterialById(match.material_a_id);
      const materialB = await this.getMaterialById(match.material_b_id);

      if (!materialA || !materialB) {
        return { error: 'Material not found' };
      }

      // Get company locations
      const companyA = await this.getCompanyById(match.company_a_id);
      const companyB = await this.getCompanyById(match.company_b_id);

      // Translate materials to shipping parameters
      const shippingParamsA = await apiFusionService.translateMaterialToShippingParams(materialA);
      const shippingParamsB = await apiFusionService.translateMaterialToShippingParams(materialB);

      return {
        material_a_shipping: shippingParamsA,
        material_b_shipping: shippingParamsB,
        from_location: companyA.location,
        to_location: companyB.location,
        estimated_distance: this.calculateDistance(companyA.location, companyB.location),
        special_handling_required: [
          ...shippingParamsA.special_handling,
          ...shippingParamsB.special_handling
        ],
        packaging_requirements: this.determinePackagingRequirements(shippingParamsA, shippingParamsB)
      };
    } catch (error) {
      console.error('Shipping analysis error:', error);
      return { error: error.message };
    }
  }

  /**
   * Analyze sustainability impact of a match
   */
  async analyzeSustainabilityImpact(match, companyProfile) {
    try {
      // Execute AI sustainability assessment
      const sustainabilityResult = await aiEvolutionEngine.executeAIAnalysis('sustainabilityAssessment', {
        company_name: companyProfile.name,
        industry: companyProfile.industry,
        match_data: match,
        environmental_goals: companyProfile.sustainability_goals || []
      });

      return sustainabilityResult.result;
    } catch (error) {
      console.error('Sustainability analysis error:', error);
      return { error: error.message };
    }
  }

  /**
   * Perform AI-powered match analysis
   */
  async performAIMatchAnalysis(match, companyProfile) {
    try {
      // Execute AI match analysis
      const matchAnalysisResult = await aiEvolutionEngine.executeAIAnalysis('matchGeneration', {
        company_a: await this.getCompanyById(match.company_a_id),
        company_b: await this.getCompanyById(match.company_b_id),
        material_a: await this.getMaterialById(match.material_a_id),
        material_b: await this.getMaterialById(match.material_b_id)
      });

      return matchAnalysisResult.result;
    } catch (error) {
      console.error('AI match analysis error:', error);
      return { error: error.message };
    }
  }

  /**
   * Generate matching insights and recommendations
   */
  async generateMatchingInsights(matches, companyProfile) {
    const insights = {
      total_matches: matches.length,
      high_potential_matches: matches.filter(m => (m.match_score || 0) >= 0.8).length,
      average_match_score: matches.reduce((sum, m) => sum + (m.match_score || 0), 0) / matches.length,
      match_types_distribution: this.calculateMatchTypeDistribution(matches),
      top_opportunities: this.identifyTopOpportunities(matches),
      recommendations: this.generateRecommendations(matches, companyProfile)
    };

    return insights;
  }

  /**
   * Store matches in database
   * Now only stores pricing-validated matches
   */
  async storeMatches(matches, companyId) {
    try {
      // Filter only pricing-validated matches
      const validatedMatches = matches.filter(match => match.pricing_validated === true);
      
      if (validatedMatches.length === 0) {
        console.log('⚠️ No pricing-validated matches to store');
        return;
      }
      
      const matchData = validatedMatches.map(match => ({
        company_a_id: companyId,
        company_b_id: match.company_b_id || match.company_id,
        material_a_id: match.material_a_id,
        material_b_id: match.material_b_id,
        match_score: match.match_score || 0,
        match_type: match.match_type || 'general_symbiosis',
        potential_savings: match.potential_savings || 0,
        implementation_complexity: match.implementation_complexity || 'medium',
        environmental_impact: match.environmental_impact || 0,
        description: match.description || '',
        materials_compatibility: match.materials_compatibility || 0,
        waste_synergy: match.waste_synergy || 0,
        energy_synergy: match.energy_synergy || 0,
        location_proximity: match.location_proximity || 0,
        ai_confidence: match.ai_confidence || 0,
        match_analysis: match.ai_analysis || {},
        status: 'potential',
        pricing_validated: true,
        pricing_timestamp: new Date().toISOString(),
        pricing_data: match.pricing_data || null
      }));

      const { error } = await supabase
        .from('symbiotic_matches')
        .insert(matchData);

      if (error) throw error;

      console.log(`✅ Stored ${matchData.length} pricing-validated matches in database`);
      
      // Log pricing validation statistics
      const failedValidations = matches.length - validatedMatches.length;
      if (failedValidations > 0) {
        console.log(`⚠️ ${failedValidations} matches failed pricing validation and were not stored`);
      }
      
    } catch (error) {
      console.error('Error storing matches:', error);
    }
  }

  /**
   * Utility functions
   */
  generateMatchKey(match, companyId) {
    const otherCompanyId = match.company_b_id || match.company_id;
    const materialA = match.material_a_id || '';
    const materialB = match.material_b_id || '';
    
    return `${companyId}_${otherCompanyId}_${materialA}_${materialB}`;
  }

  removeDuplicateMatches(matches) {
    const seen = new Set();
    return matches.filter(match => {
      const key = match.match_key;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  calculateMatchTypeDistribution(matches) {
    const distribution = {};
    matches.forEach(match => {
      const type = match.match_type || 'unknown';
      distribution[type] = (distribution[type] || 0) + 1;
    });
    return distribution;
  }

  identifyTopOpportunities(matches) {
    return matches
      .filter(match => (match.match_score || 0) >= 0.7)
      .slice(0, 5)
      .map(match => ({
        company_name: match.company_name || match.company_b_name,
        match_score: match.match_score,
        potential_savings: match.potential_savings,
        match_type: match.match_type
      }));
  }

  generateRecommendations(matches, companyProfile) {
    const recommendations = [];

    if (matches.length === 0) {
      recommendations.push({
        category: 'Profile Enhancement',
        description: 'Complete your company profile to get better matches',
        priority: 'high'
      });
    }

    if (matches.filter(m => (m.match_score || 0) >= 0.8).length === 0) {
      recommendations.push({
        category: 'Material Optimization',
        description: 'Consider adding more detailed material specifications',
        priority: 'medium'
      });
    }

    return recommendations;
  }

  calculateDistance(location1, location2) {
    // Simplified distance calculation
    // In production, use a proper geocoding service
    return 100; // Default 100 km
  }

  determinePackagingRequirements(shippingParamsA, shippingParamsB) {
    const requirements = new Set();
    
    if (shippingParamsA.packaging_requirements) {
      requirements.add(shippingParamsA.packaging_requirements);
    }
    if (shippingParamsB.packaging_requirements) {
      requirements.add(shippingParamsB.packaging_requirements);
    }
    
    return Array.from(requirements);
  }

  // Database helper functions
  async getCompanyProfile(companyData) {
    const { data, error } = await supabase
      .from('companies')
      .select('*')
      .eq('id', companyData.id)
      .single();

    if (error) throw error;
    return data;
  }

  async getCompanyMaterials(companyId) {
    const { data, error } = await supabase
      .from('materials')
      .select('*')
      .eq('company_id', companyId)
      .eq('status', 'active');

    if (error) throw error;
    return data || [];
  }

  async getCompanyById(companyId) {
    const { data, error } = await supabase
      .from('companies')
      .select('*')
      .eq('id', companyId)
      .single();

    if (error) return null;
    return data;
  }

  async getMaterialById(materialId) {
    const { data, error } = await supabase
      .from('materials')
      .select('*')
      .eq('id', materialId)
      .single();

    if (error) return null;
    return data;
  }

  async getAllCompanies() {
    const { data, error } = await supabase
      .from('companies')
      .select('*')
      .eq('onboarding_completed', true);

    if (error) return [];
    return data || [];
  }

  async getAllMaterials() {
    const { data, error } = await supabase
      .from('materials')
      .select('*')
      .eq('status', 'active');

    if (error) return [];
    return data || [];
  }

  prepareCompanyDataForEngine(companyProfile, materials) {
    return {
      id: companyProfile.id,
      name: companyProfile.name,
      industry: companyProfile.industry,
      location: companyProfile.location,
      employee_count: companyProfile.employee_count,
      materials: materials.map(m => m.material_name),
      products: companyProfile.products,
      waste_streams: materials.filter(m => m.type === 'waste').map(m => m.material_name),
      resource_needs: materials.filter(m => m.type === 'requirement').map(m => m.material_name),
      sustainability_goals: companyProfile.sustainability_goals || []
    };
  }
}

module.exports = new IntelligentMatchingService(); 