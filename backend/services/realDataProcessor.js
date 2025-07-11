const { supabase } = require('../supabase');
const EnhancedMaterialsService = require('./enhancedMaterialsService');
const RegulatoryComplianceService = require('./regulatoryComplianceService');
const ProductionMonitoring = require('./productionMonitoring');
const crypto = require('crypto');

class RealDataProcessor {
  constructor() {
    this.materialsService = new EnhancedMaterialsService();
    this.complianceService = new RegulatoryComplianceService();
    this.monitoring = ProductionMonitoring.getInstance();
    this.processingQueue = new Map();
    this.dataValidationRules = this.setupValidationRules();
  }

  /**
   * Process real company data with comprehensive validation and enrichment
   */
  async processRealCompanyData(companyData) {
    const tracking = this.monitoring.trackAIRequest('real-data-processor', 'company-processing');
    const requestId = crypto.randomUUID();
    
    try {
      this.monitoring.logBusinessEvent('real_company_data_received', companyData.id, requestId, {
        company_name: companyData.name,
        industry: companyData.industry,
        data_quality_score: this.calculateDataQualityScore(companyData)
      });

      // Step 1: Validate and clean data
      const validatedData = await this.validateAndCleanData(companyData);
      
      // Step 2: Enrich with external APIs
      const enrichedData = await this.enrichWithExternalData(validatedData);
      
      // Step 3: Generate AI insights
      const aiInsights = await this.generateAIInsights(enrichedData);
      
      // Step 4: Calculate business metrics
      const businessMetrics = await this.calculateBusinessMetrics(enrichedData, aiInsights);
      
      // Step 5: Store in database
      const storedData = await this.storeCompanyData(enrichedData, aiInsights, businessMetrics);
      
      // Step 6: Generate immediate recommendations
      const recommendations = await this.generateImmediateRecommendations(storedData);
      
      // Step 7: Track success metrics
      this.trackProcessingSuccess(companyData, storedData, recommendations);

      tracking.success();
      
      return {
        success: true,
        company_id: storedData.id,
        data_quality_score: validatedData.quality_score,
        ai_insights_count: Object.keys(aiInsights).length,
        recommendations_count: recommendations.length,
        processing_time: Date.now() - tracking.startTime,
        next_steps: this.generateNextSteps(storedData, recommendations)
      };

    } catch (error) {
      tracking.error('processing_error');
      this.monitoring.error('Real company data processing failed', {
        company: companyData.name,
        error: error.message,
        requestId
      });
      throw error;
    }
  }

  /**
   * Validate and clean incoming company data
   */
  async validateAndCleanData(companyData) {
    const tracking = this.monitoring.trackAIRequest('real-data-processor', 'data-validation');
    
    try {
      const validationResults = {
        required_fields: this.validateRequiredFields(companyData),
        data_types: this.validateDataTypes(companyData),
        business_logic: this.validateBusinessLogic(companyData),
        duplicates: await this.checkForDuplicates(companyData),
        quality_score: 0
      };

      // Calculate overall quality score
      validationResults.quality_score = this.calculateQualityScore(validationResults);
      
      // Clean data based on validation results
      const cleanedData = this.cleanData(companyData, validationResults);
      
      tracking.success();
      return {
        ...cleanedData,
        validation_results: validationResults,
        quality_score: validationResults.quality_score
      };
    } catch (error) {
      tracking.error('validation_error');
      throw error;
    }
  }

  /**
   * Enrich company data with external APIs
   */
  async enrichWithExternalData(companyData) {
    const tracking = this.monitoring.trackAIRequest('real-data-processor', 'data-enrichment');
    
    try {
      const enrichmentTasks = [];

      // Enrich materials data
      if (companyData.waste_streams) {
        for (const waste of companyData.waste_streams) {
          enrichmentTasks.push(
            this.enrichMaterialData(waste, companyData)
          );
        }
      }

      if (companyData.resource_needs) {
        for (const need of companyData.resource_needs) {
          enrichmentTasks.push(
            this.enrichMaterialData(need, companyData)
          );
        }
      }

      // Enrich company profile
      enrichmentTasks.push(
        this.enrichCompanyProfile(companyData)
      );

      // Enrich location data
      enrichmentTasks.push(
        this.enrichLocationData(companyData)
      );

      // Wait for all enrichments to complete
      const enrichmentResults = await Promise.allSettled(enrichmentTasks);
      
      // Merge enrichment results
      const enrichedData = this.mergeEnrichmentResults(companyData, enrichmentResults);
      
      tracking.success();
      return enrichedData;
    } catch (error) {
      tracking.error('enrichment_error');
      throw error;
    }
  }

  /**
   * Enrich material data with Next-Gen Materials API
   */
  async enrichMaterialData(material, companyContext) {
    try {
      const materialAnalysis = await this.materialsService.getComprehensiveMaterialAnalysis(
        material.name,
        {
          industry: companyContext.industry,
          location: companyContext.location,
          intended_use: material.intended_use,
          quantity: material.quantity,
          unit: material.unit
        }
      );

      return {
        material_id: material.id,
        enriched_data: materialAnalysis,
        enrichment_timestamp: new Date().toISOString()
      };
    } catch (error) {
      this.monitoring.warn('Material enrichment failed', {
        material: material.name,
        error: error.message
      });
      return {
        material_id: material.id,
        enriched_data: null,
        error: error.message
      };
    }
  }

  /**
   * Generate comprehensive AI insights
   */
  async generateAIInsights(enrichedData) {
    const tracking = this.monitoring.trackAIRequest('real-data-processor', 'ai-insights');
    
    try {
      const insights = {
        symbiosis_opportunities: await this.identifySymbiosisOpportunities(enrichedData),
        circular_economy_potential: await this.analyzeCircularEconomyPotential(enrichedData),
        sustainability_metrics: await this.calculateSustainabilityMetrics(enrichedData),
        financial_analysis: await this.performFinancialAnalysis(enrichedData),
        risk_assessment: await this.performRiskAssessment(enrichedData),
        competitive_analysis: await this.performCompetitiveAnalysis(enrichedData),
        market_opportunities: await this.identifyMarketOpportunities(enrichedData),
        regulatory_compliance: await this.analyzeRegulatoryCompliance(enrichedData)
      };

      tracking.success();
      return insights;
    } catch (error) {
      tracking.error('insights_error');
      throw error;
    }
  }

  /**
   * Calculate business metrics
   */
  async calculateBusinessMetrics(enrichedData, aiInsights) {
    const tracking = this.monitoring.trackAIRequest('real-data-processor', 'business-metrics');
    
    try {
      const metrics = {
        total_waste_value: this.calculateTotalWasteValue(enrichedData),
        potential_savings: this.calculatePotentialSavings(enrichedData, aiInsights),
        carbon_reduction_potential: this.calculateCarbonReductionPotential(enrichedData),
        symbiosis_score: this.calculateSymbiosisScore(enrichedData, aiInsights),
        market_position: this.calculateMarketPosition(enrichedData),
        growth_potential: this.calculateGrowthPotential(enrichedData, aiInsights),
        risk_score: this.calculateRiskScore(enrichedData, aiInsights),
        compliance_score: this.calculateComplianceScore(enrichedData, aiInsights)
      };

      tracking.success();
      return metrics;
    } catch (error) {
      tracking.error('metrics_error');
      throw error;
    }
  }

  /**
   * Store company data in database
   */
  async storeCompanyData(enrichedData, aiInsights, businessMetrics) {
    const tracking = this.monitoring.trackDatabaseQuery('store_company_data');
    
    try {
      // Store main company data
      const { data: company, error: companyError } = await supabase
        .from('companies')
        .upsert({
          id: enrichedData.id,
          name: enrichedData.name,
          industry: enrichedData.industry,
          location: enrichedData.location,
          size: enrichedData.size,
          contact_info: enrichedData.contact_info,
          business_description: enrichedData.business_description,
          enriched_data: enrichedData,
          ai_insights: aiInsights,
          business_metrics: businessMetrics,
          data_quality_score: enrichedData.quality_score,
          processing_status: 'completed',
          last_updated: new Date().toISOString()
        })
        .select()
        .single();

      if (companyError) throw companyError;

      // Store materials data
      if (enrichedData.waste_streams) {
        for (const waste of enrichedData.waste_streams) {
          await this.storeMaterialData(waste, company.id, 'waste');
        }
      }

      if (enrichedData.resource_needs) {
        for (const need of enrichedData.resource_needs) {
          await this.storeMaterialData(need, company.id, 'need');
        }
      }

      // Store AI insights
      await this.storeAIInsights(aiInsights, company.id);

      tracking.success();
      return company;
    } catch (error) {
      tracking.error();
      throw error;
    }
  }

  /**
   * Generate immediate recommendations
   */
  async generateImmediateRecommendations(storedData) {
    const tracking = this.monitoring.trackAIRequest('real-data-processor', 'recommendations');
    
    try {
      const recommendations = [];

      // High-value symbiosis opportunities
      if (storedData.ai_insights.symbiosis_opportunities) {
        const topOpportunities = storedData.ai_insights.symbiosis_opportunities
          .filter(opp => opp.potential_value > 10000) // $10k+ opportunities
          .slice(0, 5);

        for (const opportunity of topOpportunities) {
          recommendations.push({
            type: 'symbiosis_opportunity',
            priority: 'high',
            title: `Symbiosis Opportunity: ${opportunity.partner_industry}`,
            description: opportunity.description,
            potential_value: opportunity.potential_value,
            implementation_time: opportunity.implementation_time,
            risk_level: opportunity.risk_level,
            action_items: opportunity.action_items
          });
        }
      }

      // Regulatory compliance actions
      if (storedData.ai_insights.regulatory_compliance) {
        const complianceIssues = storedData.ai_insights.regulatory_compliance
          .filter(issue => issue.severity === 'high' || issue.severity === 'critical');

        for (const issue of complianceIssues) {
          recommendations.push({
            type: 'regulatory_compliance',
            priority: 'critical',
            title: `Compliance Issue: ${issue.regulation}`,
            description: issue.description,
            deadline: issue.deadline,
            penalty: issue.penalty,
            action_items: issue.remediation_steps
          });
        }
      }

      // Sustainability improvements
      if (storedData.business_metrics.carbon_reduction_potential > 1000) {
        recommendations.push({
          type: 'sustainability',
          priority: 'medium',
          title: 'High Carbon Reduction Potential',
          description: `Potential to reduce ${storedData.business_metrics.carbon_reduction_potential} kg CO2e annually`,
          potential_savings: storedData.business_metrics.potential_savings,
          action_items: [
            'Implement waste segregation program',
            'Explore material recovery options',
            'Consider energy efficiency measures'
          ]
        });
      }

      // Financial optimization
      if (storedData.business_metrics.potential_savings > 50000) {
        recommendations.push({
          type: 'financial_optimization',
          priority: 'high',
          title: 'Significant Cost Savings Opportunity',
          description: `Potential annual savings of $${storedData.business_metrics.potential_savings.toLocaleString()}`,
          roi_percentage: (storedData.business_metrics.potential_savings / 100000) * 100,
          action_items: [
            'Conduct detailed waste audit',
            'Identify high-value material streams',
            'Explore partnership opportunities'
          ]
        });
      }

      tracking.success();
      return recommendations;
    } catch (error) {
      tracking.error('recommendations_error');
      throw error;
    }
  }

  /**
   * Setup validation rules
   */
  setupValidationRules() {
    return {
      required_fields: [
        'name', 'industry', 'location', 'size', 'contact_info'
      ],
      data_types: {
        name: 'string',
        industry: 'string',
        location: 'string',
        size: 'string',
        contact_info: 'object',
        waste_streams: 'array',
        resource_needs: 'array'
      },
      business_logic: {
        min_company_size: 10,
        max_waste_streams: 50,
        max_resource_needs: 50,
        valid_industries: [
          'manufacturing', 'chemical', 'food_beverage', 'textiles',
          'automotive', 'aerospace', 'electronics', 'construction',
          'pharmaceutical', 'mining', 'energy', 'agriculture'
        ]
      }
    };
  }

  /**
   * Validate required fields
   */
  validateRequiredFields(data) {
    const results = {};
    for (const field of this.dataValidationRules.required_fields) {
      results[field] = {
        present: data.hasOwnProperty(field) && data[field] !== null && data[field] !== undefined,
        value: data[field]
      };
    }
    return results;
  }

  /**
   * Calculate quality score
   */
  calculateQualityScore(validationResults) {
    let score = 0;
    let totalChecks = 0;

    // Required fields (40% weight)
    const requiredFields = Object.values(validationResults.required_fields);
    const presentFields = requiredFields.filter(field => field.present).length;
    score += (presentFields / requiredFields.length) * 40;
    totalChecks += 40;

    // Data types (30% weight)
    const dataTypeChecks = Object.keys(validationResults.data_types || {}).length;
    score += (dataTypeChecks / dataTypeChecks) * 30;
    totalChecks += 30;

    // Business logic (30% weight)
    const businessLogicChecks = Object.keys(validationResults.business_logic || {}).length;
    score += (businessLogicChecks / businessLogicChecks) * 30;
    totalChecks += 30;

    return Math.round((score / totalChecks) * 100);
  }

  /**
   * Track processing success
   */
  trackProcessingSuccess(originalData, storedData, recommendations) {
    this.monitoring.logBusinessEvent('company_data_processed_successfully', storedData.id, null, {
      company_name: originalData.name,
      data_quality_score: storedData.data_quality_score,
      recommendations_count: recommendations.length,
      processing_time: Date.now() - this.processingQueue.get(storedData.id)?.startTime
    });

    // Update business metrics
    this.monitoring.trackSymbiosisOpportunity(
      storedData.location || 'unknown',
      storedData.industry || 'unknown',
      recommendations.filter(r => r.type === 'symbiosis_opportunity').length
    );
  }

  /**
   * Generate next steps
   */
  generateNextSteps(storedData, recommendations) {
    return [
      {
        step: 1,
        action: 'Review AI Insights',
        description: 'Analyze the generated AI insights for accuracy and relevance',
        priority: 'high',
        estimated_time: '30 minutes'
      },
      {
        step: 2,
        action: 'Prioritize Recommendations',
        description: 'Sort recommendations by potential value and implementation complexity',
        priority: 'high',
        estimated_time: '15 minutes'
      },
      {
        step: 3,
        action: 'Contact Company',
        description: 'Reach out to discuss findings and next steps',
        priority: 'medium',
        estimated_time: '1 hour'
      },
      {
        step: 4,
        action: 'Schedule Follow-up',
        description: 'Plan detailed analysis and implementation strategy',
        priority: 'medium',
        estimated_time: '30 minutes'
      }
    ];
  }
}

module.exports = RealDataProcessor; 