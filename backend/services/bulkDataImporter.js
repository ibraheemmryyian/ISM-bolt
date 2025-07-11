const RealDataProcessor = require('./realDataProcessor');
const { supabase } = require('../supabase');
const ProductionMonitoring = require('./productionMonitoring');
const fs = require('fs').promises;
const path = require('path');

class BulkDataImporter {
  constructor() {
    this.dataProcessor = new RealDataProcessor();
    this.monitoring = ProductionMonitoring.getInstance();
    this.importQueue = [];
    this.processingStatus = new Map();
    this.batchSize = 5; // Process 5 companies at a time
    this.maxRetries = 3;
  }

  /**
   * Import 50 real company profiles with comprehensive processing
   */
  async importRealCompanyData(companyDataArray, options = {}) {
    const importId = `import_${Date.now()}`;
    const tracking = this.monitoring.trackAIRequest('bulk-importer', 'import-50-companies');
    
    try {
      this.monitoring.logBusinessEvent('bulk_import_started', null, importId, {
        total_companies: companyDataArray.length,
        import_id: importId,
        options
      });

      // Validate input data
      const validationResults = await this.validateBulkData(companyDataArray);
      
      if (!validationResults.isValid) {
        throw new Error(`Bulk data validation failed: ${validationResults.errors.join(', ')}`);
      }

      // Initialize import tracking
      this.initializeImportTracking(importId, companyDataArray.length);

      // Process companies in batches
      const results = await this.processCompaniesInBatches(companyDataArray, importId);

      // Generate import summary
      const summary = this.generateImportSummary(results, importId);

      // Store import results
      await this.storeImportResults(importId, results, summary);

      // Generate actionable insights
      const insights = await this.generateBulkInsights(results);

      tracking.success();

      return {
        success: true,
        import_id: importId,
        summary,
        insights,
        next_actions: this.generateNextActions(summary, insights)
      };

    } catch (error) {
      tracking.error('import_error');
      this.monitoring.error('Bulk import failed', {
        import_id: importId,
        error: error.message,
        companies_processed: this.processingStatus.get(importId)?.processed || 0
      });
      throw error;
    }
  }

  /**
   * Validate bulk data before processing
   */
  async validateBulkData(companyDataArray) {
    const tracking = this.monitoring.trackAIRequest('bulk-importer', 'bulk-validation');
    
    try {
      const errors = [];
      const warnings = [];
      let validCompanies = 0;

      for (let i = 0; i < companyDataArray.length; i++) {
        const company = companyDataArray[i];
        const validation = await this.validateSingleCompany(company, i);

        if (validation.isValid) {
          validCompanies++;
        } else {
          errors.push(`Company ${i + 1} (${company.name || 'Unknown'}): ${validation.errors.join(', ')}`);
        }

        if (validation.warnings.length > 0) {
          warnings.push(`Company ${i + 1} (${company.name || 'Unknown'}): ${validation.warnings.join(', ')}`);
        }
      }

      const isValid = validCompanies === companyDataArray.length;

      tracking.success();
      return {
        isValid,
        validCompanies,
        totalCompanies: companyDataArray.length,
        errors,
        warnings,
        successRate: (validCompanies / companyDataArray.length) * 100
      };
    } catch (error) {
      tracking.error('validation_error');
      throw error;
    }
  }

  /**
   * Process companies in batches for optimal performance
   */
  async processCompaniesInBatches(companyDataArray, importId) {
    const results = {
      successful: [],
      failed: [],
      skipped: [],
      processing_times: []
    };

    const batches = this.createBatches(companyDataArray, this.batchSize);

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      const batchStartTime = Date.now();

      this.monitoring.info(`Processing batch ${batchIndex + 1}/${batches.length}`, {
        import_id: importId,
        batch_size: batch.length,
        companies: batch.map(c => c.name || c.id)
      });

      // Process batch concurrently
      const batchPromises = batch.map(async (company, index) => {
        const companyIndex = batchIndex * this.batchSize + index;
        return this.processCompanyWithRetry(company, companyIndex, importId);
      });

      const batchResults = await Promise.allSettled(batchPromises);

      // Process batch results
      for (let i = 0; i < batchResults.length; i++) {
        const result = batchResults[i];
        const company = batch[i];
        const companyIndex = batchIndex * this.batchSize + i;

        if (result.status === 'fulfilled' && result.value.success) {
          results.successful.push({
            company: company,
            result: result.value,
            index: companyIndex
          });
        } else {
          const error = result.status === 'rejected' ? result.reason : result.value.error;
          results.failed.push({
            company: company,
            error: error,
            index: companyIndex
          });
        }
      }

      const batchProcessingTime = Date.now() - batchStartTime;
      results.processing_times.push(batchProcessingTime);

      // Update progress
      this.updateImportProgress(importId, (batchIndex + 1) * this.batchSize);

      // Log batch completion
      this.monitoring.info(`Batch ${batchIndex + 1} completed`, {
        import_id: importId,
        processing_time: batchProcessingTime,
        success_count: batchResults.filter(r => r.status === 'fulfilled' && r.value.success).length,
        failure_count: batchResults.filter(r => r.status === 'rejected' || !r.value.success).length
      });

      // Small delay between batches to prevent overwhelming the system
      if (batchIndex < batches.length - 1) {
        await this.delay(1000);
      }
    }

    return results;
  }

  /**
   * Process single company with retry logic
   */
  async processCompanyWithRetry(company, index, importId) {
    let lastError;
    
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        const result = await this.dataProcessor.processRealCompanyData(company);
        
        this.monitoring.logBusinessEvent('company_processed_successfully', company.id, importId, {
          company_name: company.name,
          attempt: attempt,
          index: index
        });

        return {
          success: true,
          result: result,
          attempt: attempt
        };

      } catch (error) {
        lastError = error;
        
        this.monitoring.warn('Company processing attempt failed', {
          company: company.name,
          attempt: attempt,
          error: error.message,
          import_id: importId
        });

        if (attempt < this.maxRetries) {
          // Exponential backoff
          await this.delay(Math.pow(2, attempt) * 1000);
        }
      }
    }

    // All retries failed
    this.monitoring.error('Company processing failed after all retries', {
      company: company.name,
      error: lastError.message,
      import_id: importId
    });

    return {
      success: false,
      error: lastError.message,
      attempts: this.maxRetries
    };
  }

  /**
   * Generate bulk insights from all processed companies
   */
  async generateBulkInsights(results) {
    const tracking = this.monitoring.trackAIRequest('bulk-importer', 'bulk-insights');
    
    try {
      const successfulCompanies = results.successful.map(r => r.result.result);
      
      const insights = {
        market_analysis: await this.analyzeMarket(successfulCompanies),
        symbiosis_network: await this.analyzeSymbiosisNetwork(successfulCompanies),
        industry_trends: await this.analyzeIndustryTrends(successfulCompanies),
        regional_opportunities: await this.analyzeRegionalOpportunities(successfulCompanies),
        high_value_targets: await this.identifyHighValueTargets(successfulCompanies),
        risk_assessment: await this.assessBulkRisks(successfulCompanies),
        competitive_landscape: await this.analyzeCompetitiveLandscape(successfulCompanies)
      };

      tracking.success();
      return insights;
    } catch (error) {
      tracking.error('insights_error');
      throw error;
    }
  }

  /**
   * Analyze market opportunities across all companies
   */
  async analyzeMarket(companies) {
    const marketData = {
      total_waste_value: 0,
      total_potential_savings: 0,
      industry_distribution: {},
      material_distribution: {},
      regional_distribution: {},
      high_value_streams: []
    };

    for (const company of companies) {
      // Aggregate financial data
      marketData.total_waste_value += company.business_metrics?.total_waste_value || 0;
      marketData.total_potential_savings += company.business_metrics?.potential_savings || 0;

      // Industry distribution
      const industry = company.enriched_data?.industry || 'unknown';
      marketData.industry_distribution[industry] = (marketData.industry_distribution[industry] || 0) + 1;

      // Regional distribution
      const region = company.enriched_data?.location || 'unknown';
      marketData.regional_distribution[region] = (marketData.regional_distribution[region] || 0) + 1;

      // High value streams
      if (company.business_metrics?.total_waste_value > 100000) {
        marketData.high_value_streams.push({
          company: company.enriched_data?.name,
          industry: industry,
          waste_value: company.business_metrics.total_waste_value,
          potential_savings: company.business_metrics.potential_savings
        });
      }
    }

    // Sort high value streams by potential savings
    marketData.high_value_streams.sort((a, b) => b.potential_savings - a.potential_savings);

    return marketData;
  }

  /**
   * Analyze symbiosis network opportunities
   */
  async analyzeSymbiosisNetwork(companies) {
    const networkData = {
      potential_partnerships: [],
      industry_connections: {},
      material_exchanges: {},
      regional_clusters: {}
    };

    // Find potential partnerships between companies
    for (let i = 0; i < companies.length; i++) {
      for (let j = i + 1; j < companies.length; j++) {
        const companyA = companies[i];
        const companyB = companies[j];

        const partnershipScore = this.calculatePartnershipScore(companyA, companyB);
        
        if (partnershipScore > 0.7) { // High potential partnerships
          networkData.potential_partnerships.push({
            company_a: companyA.enriched_data?.name,
            company_b: companyB.enriched_data?.name,
            score: partnershipScore,
            potential_value: this.calculatePartnershipValue(companyA, companyB),
            synergy_type: this.identifySynergyType(companyA, companyB)
          });
        }
      }
    }

    // Sort partnerships by potential value
    networkData.potential_partnerships.sort((a, b) => b.potential_value - a.potential_value);

    return networkData;
  }

  /**
   * Identify high-value targets for immediate outreach
   */
  async identifyHighValueTargets(companies) {
    const targets = companies
      .filter(company => {
        const metrics = company.business_metrics;
        return (
          metrics?.potential_savings > 50000 || // High savings potential
          metrics?.symbiosis_score > 0.8 || // High symbiosis potential
          metrics?.carbon_reduction_potential > 1000 // High environmental impact
        );
      })
      .map(company => ({
        company_name: company.enriched_data?.name,
        industry: company.enriched_data?.industry,
        location: company.enriched_data?.location,
        potential_savings: company.business_metrics?.potential_savings,
        symbiosis_score: company.business_metrics?.symbiosis_score,
        carbon_reduction: company.business_metrics?.carbon_reduction_potential,
        priority_score: this.calculatePriorityScore(company),
        recommended_approach: this.generateRecommendedApproach(company)
      }))
      .sort((a, b) => b.priority_score - a.priority_score);

    return targets.slice(0, 10); // Top 10 targets
  }

  /**
   * Generate next actions based on import results
   */
  generateNextActions(summary, insights) {
    const actions = [];

    if (summary.success_rate > 90) {
      actions.push({
        priority: 'high',
        action: 'Immediate Outreach to High-Value Targets',
        description: `Contact ${insights.high_value_targets.length} companies with highest potential`,
        estimated_value: insights.market_analysis.total_potential_savings,
        timeline: 'This week'
      });
    }

    if (insights.symbiosis_network.potential_partnerships.length > 0) {
      actions.push({
        priority: 'high',
        action: 'Facilitate Partnership Introductions',
        description: `Introduce ${insights.symbiosis_network.potential_partnerships.length} potential partnerships`,
        estimated_value: insights.symbiosis_network.potential_partnerships.reduce((sum, p) => sum + p.potential_value, 0),
        timeline: 'Next 2 weeks'
      });
    }

    actions.push({
      priority: 'medium',
      action: 'Generate Detailed Reports',
      description: 'Create comprehensive analysis reports for each company',
      timeline: 'Next week'
    });

    actions.push({
      priority: 'medium',
      action: 'Schedule Follow-up Calls',
      description: 'Plan detailed discussions with interested companies',
      timeline: 'Next 2 weeks'
    });

    return actions;
  }

  // Helper methods
  createBatches(array, batchSize) {
    const batches = [];
    for (let i = 0; i < array.length; i += batchSize) {
      batches.push(array.slice(i, i + batchSize));
    }
    return batches;
  }

  initializeImportTracking(importId, totalCompanies) {
    this.processingStatus.set(importId, {
      total: totalCompanies,
      processed: 0,
      successful: 0,
      failed: 0,
      startTime: Date.now()
    });
  }

  updateImportProgress(importId, processed) {
    const status = this.processingStatus.get(importId);
    if (status) {
      status.processed = processed;
    }
  }

  generateImportSummary(results, importId) {
    const status = this.processingStatus.get(importId);
    const processingTime = Date.now() - status.startTime;

    return {
      import_id: importId,
      total_companies: status.total,
      successful: results.successful.length,
      failed: results.failed.length,
      success_rate: (results.successful.length / status.total) * 100,
      processing_time: processingTime,
      average_processing_time: processingTime / status.total,
      total_potential_value: results.successful.reduce((sum, r) => sum + (r.result.result.business_metrics?.potential_savings || 0), 0)
    };
  }

  async storeImportResults(importId, results, summary) {
    const { error } = await supabase
      .from('bulk_imports')
      .insert({
        import_id: importId,
        summary: summary,
        results: results,
        created_at: new Date().toISOString()
      });

    if (error) {
      this.monitoring.error('Failed to store import results', { error: error.message });
    }
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  calculatePartnershipScore(companyA, companyB) {
    // Implement partnership scoring logic
    return Math.random() * 0.5 + 0.5; // Placeholder
  }

  calculatePartnershipValue(companyA, companyB) {
    // Implement partnership value calculation
    return (companyA.business_metrics?.potential_savings || 0) + (companyB.business_metrics?.potential_savings || 0);
  }

  identifySynergyType(companyA, companyB) {
    // Implement synergy type identification
    return 'waste_to_resource';
  }

  calculatePriorityScore(company) {
    const metrics = company.business_metrics;
    return (
      (metrics?.potential_savings / 100000) * 0.4 +
      (metrics?.symbiosis_score || 0) * 0.3 +
      (metrics?.carbon_reduction_potential / 1000) * 0.3
    );
  }

  generateRecommendedApproach(company) {
    // Implement approach recommendation logic
    return 'Direct contact with value proposition';
  }
}

module.exports = BulkDataImporter; 