const axios = require('axios');
const { supabase } = require('../supabase');
const ProductionMonitoring = require('./productionMonitoring');

class RegulatoryComplianceService {
  constructor() {
    this.monitoring = ProductionMonitoring.getInstance();
    this.cache = new Map();
    this.cacheTimeout = 1800000; // 30 minutes
    
    // API configurations for different regulatory data sources
    this.apiConfigs = {
      epa: {
        baseUrl: 'https://api.epa.gov',
        apiKey: process.env.EPA_API_KEY,
        timeout: 15000
      },
      osha: {
        baseUrl: 'https://api.osha.gov',
        apiKey: process.env.OSHA_API_KEY,
        timeout: 15000
      },
      reach: {
        baseUrl: 'https://api.echa.europa.eu',
        apiKey: process.env.REACH_API_KEY,
        timeout: 20000
      },
      rohs: {
        baseUrl: 'https://api.rohs-directive.eu',
        apiKey: process.env.ROHS_API_KEY,
        timeout: 15000
      },
      iso: {
        baseUrl: 'https://api.iso.org',
        apiKey: process.env.ISO_API_KEY,
        timeout: 15000
      },
      ghs: {
        baseUrl: 'https://api.unece.org',
        apiKey: process.env.GHS_API_KEY,
        timeout: 15000
      }
    };
  }

  /**
   * Get comprehensive regulatory compliance analysis
   */
  async getComprehensiveComplianceAnalysis(materialData, companyContext) {
    const tracking = this.monitoring.trackAIRequest('regulatory-service', 'comprehensive-analysis');
    
    try {
      const cacheKey = `compliance_${materialData.id}_${JSON.stringify(companyContext)}`;
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        tracking.success();
        return cached;
      }

      const analysis = {
        global_regulations: await this.getGlobalRegulations(materialData),
        regional_regulations: await this.getRegionalRegulations(materialData, companyContext),
        industry_specific: await this.getIndustrySpecificRegulations(materialData, companyContext),
        safety_standards: await this.getSafetyStandards(materialData),
        environmental_regulations: await this.getEnvironmentalRegulations(materialData),
        transportation_regulations: await this.getTransportationRegulations(materialData),
        disposal_regulations: await this.getDisposalRegulations(materialData),
        certification_requirements: await this.getCertificationRequirements(materialData, companyContext),
        compliance_score: await this.calculateComplianceScore(materialData, companyContext),
        risk_assessment: await this.performRiskAssessment(materialData, companyContext),
        recommendations: await this.generateComplianceRecommendations(materialData, companyContext),
        monitoring_requirements: await this.getMonitoringRequirements(materialData, companyContext)
      };

      this.setCache(cacheKey, analysis);
      tracking.success();
      return analysis;
    } catch (error) {
      tracking.error('analysis_error');
      this.monitoring.error('Regulatory compliance analysis failed', {
        material: materialData.name,
        error: error.message,
        companyContext
      });
      throw error;
    }
  }

  /**
   * Get global regulations (REACH, GHS, etc.)
   */
  async getGlobalRegulations(materialData) {
    const tracking = this.monitoring.trackAIRequest('regulatory-service', 'global-regulations');
    
    try {
      const regulations = {
        reach: await this.getREACHCompliance(materialData),
        ghs: await this.getGHSCompliance(materialData),
        iso_standards: await this.getISOStandards(materialData),
        un_regulations: await this.getUNRegulations(materialData),
        international_treaties: await this.getInternationalTreaties(materialData)
      };

      tracking.success();
      return regulations;
    } catch (error) {
      tracking.error('api_error');
      throw error;
    }
  }

  /**
   * Get REACH compliance data
   */
  async getREACHCompliance(materialData) {
    const tracking = this.monitoring.trackExternalAPI('reach', 'compliance');
    
    try {
      const response = await this.makeAPICall('reach', 'GET', `/substances/${materialData.cas_number}`);
      
      const reachData = {
        registration_status: response.data.registration_status,
        registration_number: response.data.registration_number,
        registration_date: response.data.registration_date,
        registrant: response.data.registrant,
        tonnage_band: response.data.tonnage_band,
        classification: response.data.classification || [],
        labelling: response.data.labelling || [],
        restrictions: response.data.restrictions || [],
        authorisation_requirements: response.data.authorisation_requirements || [],
        svhc_status: response.data.svhc_status,
        candidate_list_status: response.data.candidate_list_status,
        authorisation_list_status: response.data.authorisation_list_status,
        restriction_list_status: response.data.restriction_list_status
      };

      tracking.success();
      return reachData;
    } catch (error) {
      tracking.error('api_error');
      return this.getDefaultREACHData();
    }
  }

  /**
   * Get GHS compliance data
   */
  async getGHSCompliance(materialData) {
    const tracking = this.monitoring.trackExternalAPI('ghs', 'compliance');
    
    try {
      const response = await this.makeAPICall('ghs', 'GET', `/classification/${materialData.cas_number}`);
      
      const ghsData = {
        hazard_classification: response.data.hazard_classification || [],
        hazard_statements: response.data.hazard_statements || [],
        precautionary_statements: response.data.precautionary_statements || [],
        signal_words: response.data.signal_words || [],
        pictograms: response.data.pictograms || [],
        safety_data_sheet_required: response.data.sds_required,
        transport_classification: response.data.transport_classification || [],
        storage_classification: response.data.storage_classification || []
      };

      tracking.success();
      return ghsData;
    } catch (error) {
      tracking.error('api_error');
      return this.getDefaultGHSData();
    }
  }

  /**
   * Get regional regulations based on company location
   */
  async getRegionalRegulations(materialData, companyContext) {
    const tracking = this.monitoring.trackAIRequest('regulatory-service', 'regional-regulations');
    
    try {
      const regions = this.getRelevantRegions(companyContext);
      const regionalData = {};

      for (const region of regions) {
        regionalData[region] = await this.getRegionSpecificRegulations(materialData, region);
      }

      tracking.success();
      return regionalData;
    } catch (error) {
      tracking.error('processing_error');
      throw error;
    }
  }

  /**
   * Get US EPA regulations
   */
  async getEPARegulations(materialData) {
    const tracking = this.monitoring.trackExternalAPI('epa', 'regulations');
    
    try {
      const response = await this.makeAPICall('epa', 'GET', `/chemicals/${materialData.cas_number}`);
      
      const epaData = {
        tsca_status: response.data.tsca_status,
        tsca_inventory_status: response.data.tsca_inventory_status,
        epcra_requirements: response.data.epcra_requirements || [],
        clean_air_act: response.data.clean_air_act || [],
        clean_water_act: response.data.clean_water_act || [],
        rcra_requirements: response.data.rcra_requirements || [],
        cercla_requirements: response.data.cercla_requirements || [],
        reporting_requirements: response.data.reporting_requirements || []
      };

      tracking.success();
      return epaData;
    } catch (error) {
      tracking.error('api_error');
      return this.getDefaultEPAData();
    }
  }

  /**
   * Get OSHA regulations
   */
  async getOSHARegulations(materialData) {
    const tracking = this.monitoring.trackExternalAPI('osha', 'regulations');
    
    try {
      const response = await this.makeAPICall('osha', 'GET', `/hazards/${materialData.cas_number}`);
      
      const oshaData = {
        permissible_exposure_limits: response.data.pels || [],
        recommended_exposure_limits: response.data.rels || [],
        short_term_exposure_limits: response.data.stels || [],
        immediately_dangerous_levels: response.data.idlhs || [],
        hazard_communication: response.data.hazcom || [],
        personal_protective_equipment: response.data.ppe || [],
        engineering_controls: response.data.engineering_controls || [],
        administrative_controls: response.data.administrative_controls || []
      };

      tracking.success();
      return oshaData;
    } catch (error) {
      tracking.error('api_error');
      return this.getDefaultOSHAData();
    }
  }

  /**
   * Get industry-specific regulations
   */
  async getIndustrySpecificRegulations(materialData, companyContext) {
    const tracking = this.monitoring.trackAIRequest('regulatory-service', 'industry-regulations');
    
    try {
      const industry = companyContext.industry || 'general';
      const regulations = {
        automotive: await this.getAutomotiveRegulations(materialData),
        aerospace: await this.getAerospaceRegulations(materialData),
        pharmaceutical: await this.getPharmaceuticalRegulations(materialData),
        food_contact: await this.getFoodContactRegulations(materialData),
        electronics: await this.getElectronicsRegulations(materialData),
        construction: await this.getConstructionRegulations(materialData),
        textiles: await this.getTextileRegulations(materialData)
      };

      tracking.success();
      return regulations[industry] || regulations.general;
    } catch (error) {
      tracking.error('processing_error');
      throw error;
    }
  }

  /**
   * Get ROHS compliance for electronics
   */
  async getElectronicsRegulations(materialData) {
    const tracking = this.monitoring.trackExternalAPI('rohs', 'compliance');
    
    try {
      const response = await this.makeAPICall('rohs', 'GET', `/materials/${materialData.cas_number}`);
      
      const rohsData = {
        restricted_substances: response.data.restricted_substances || [],
        concentration_limits: response.data.concentration_limits || {},
        exemptions: response.data.exemptions || [],
        compliance_status: response.data.compliance_status,
        testing_requirements: response.data.testing_requirements || [],
        documentation_requirements: response.data.documentation_requirements || []
      };

      tracking.success();
      return rohsData;
    } catch (error) {
      tracking.error('api_error');
      return this.getDefaultROHSData();
    }
  }

  /**
   * Calculate overall compliance score
   */
  async calculateComplianceScore(materialData, companyContext) {
    const tracking = this.monitoring.trackAIRequest('regulatory-service', 'compliance-score');
    
    try {
      const scores = {
        global_compliance: await this.calculateGlobalComplianceScore(materialData),
        regional_compliance: await this.calculateRegionalComplianceScore(materialData, companyContext),
        industry_compliance: await this.calculateIndustryComplianceScore(materialData, companyContext),
        safety_compliance: await this.calculateSafetyComplianceScore(materialData),
        environmental_compliance: await this.calculateEnvironmentalComplianceScore(materialData)
      };

      const overallScore = Object.values(scores).reduce((sum, score) => sum + score, 0) / Object.keys(scores).length;

      tracking.success();
      return {
        overall_score: Math.round(overallScore * 100) / 100,
        component_scores: scores,
        risk_level: this.determineRiskLevel(overallScore),
        compliance_status: this.determineComplianceStatus(overallScore)
      };
    } catch (error) {
      tracking.error('calculation_error');
      throw error;
    }
  }

  /**
   * Perform risk assessment
   */
  async performRiskAssessment(materialData, companyContext) {
    const tracking = this.monitoring.trackAIRequest('regulatory-service', 'risk-assessment');
    
    try {
      const risks = {
        regulatory_risks: await this.assessRegulatoryRisks(materialData, companyContext),
        compliance_risks: await this.assessComplianceRisks(materialData, companyContext),
        operational_risks: await this.assessOperationalRisks(materialData, companyContext),
        financial_risks: await this.assessFinancialRisks(materialData, companyContext),
        reputational_risks: await this.assessReputationalRisks(materialData, companyContext)
      };

      const overallRisk = this.calculateOverallRisk(risks);

      tracking.success();
      return {
        overall_risk_level: overallRisk,
        risk_components: risks,
        mitigation_strategies: await this.generateMitigationStrategies(risks),
        monitoring_requirements: this.determineMonitoringRequirements(overallRisk)
      };
    } catch (error) {
      tracking.error('assessment_error');
      throw error;
    }
  }

  /**
   * Generate compliance recommendations
   */
  async generateComplianceRecommendations(materialData, companyContext) {
    const tracking = this.monitoring.trackAIRequest('regulatory-service', 'recommendations');
    
    try {
      const recommendations = {
        immediate_actions: await this.getImmediateActions(materialData, companyContext),
        short_term_actions: await this.getShortTermActions(materialData, companyContext),
        long_term_actions: await this.getLongTermActions(materialData, companyContext),
        documentation_requirements: await this.getDocumentationRequirements(materialData, companyContext),
        training_requirements: await this.getTrainingRequirements(materialData, companyContext),
        monitoring_procedures: await this.getMonitoringProcedures(materialData, companyContext)
      };

      tracking.success();
      return recommendations;
    } catch (error) {
      tracking.error('generation_error');
      throw error;
    }
  }

  /**
   * Make API call to regulatory services
   */
  async makeAPICall(service, method, endpoint, data = null) {
    const config = this.apiConfigs[service];
    if (!config) {
      throw new Error(`Unknown regulatory service: ${service}`);
    }

    const apiConfig = {
      method,
      url: `${config.baseUrl}${endpoint}`,
      headers: {
        'Authorization': `Bearer ${config.apiKey}`,
        'Content-Type': 'application/json'
      },
      timeout: config.timeout
    };

    if (data) {
      apiConfig.data = data;
    }

    const response = await axios(apiConfig);
    return response;
  }

  /**
   * Cache management
   */
  getFromCache(key) {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }
    this.cache.delete(key);
    return null;
  }

  setCache(key, data) {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  // Helper methods for data processing and calculations
  getRelevantRegions(companyContext) {
    const regions = ['us', 'eu', 'asia'];
    if (companyContext.location) {
      const location = companyContext.location.toLowerCase();
      if (location.includes('united states') || location.includes('usa')) {
        return ['us'];
      } else if (location.includes('europe') || location.includes('eu')) {
        return ['eu'];
      } else if (location.includes('asia') || location.includes('china') || location.includes('japan')) {
        return ['asia'];
      }
    }
    return regions;
  }

  determineRiskLevel(score) {
    if (score >= 0.8) return 'low';
    if (score >= 0.6) return 'medium';
    if (score >= 0.4) return 'high';
    return 'critical';
  }

  determineComplianceStatus(score) {
    if (score >= 0.9) return 'fully_compliant';
    if (score >= 0.7) return 'mostly_compliant';
    if (score >= 0.5) return 'partially_compliant';
    return 'non_compliant';
  }

  calculateOverallRisk(risks) {
    const riskScores = {
      low: 1,
      medium: 2,
      high: 3,
      critical: 4
    };

    const totalRisk = Object.values(risks).reduce((sum, risk) => {
      return sum + (riskScores[risk.level] || 1);
    }, 0);

    const averageRisk = totalRisk / Object.keys(risks).length;

    if (averageRisk >= 3.5) return 'critical';
    if (averageRisk >= 2.5) return 'high';
    if (averageRisk >= 1.5) return 'medium';
    return 'low';
  }

  // Default data methods for when APIs are unavailable
  getDefaultREACHData() {
    return {
      registration_status: 'unknown',
      svhc_status: 'not_listed',
      candidate_list_status: 'not_listed',
      authorisation_list_status: 'not_listed',
      restriction_list_status: 'not_listed'
    };
  }

  getDefaultGHSData() {
    return {
      hazard_classification: [],
      hazard_statements: [],
      precautionary_statements: [],
      signal_words: [],
      pictograms: [],
      safety_data_sheet_required: true
    };
  }

  getDefaultEPAData() {
    return {
      tsca_status: 'unknown',
      tsca_inventory_status: 'unknown',
      epcra_requirements: [],
      clean_air_act: [],
      clean_water_act: [],
      rcra_requirements: []
    };
  }

  getDefaultOSHAData() {
    return {
      permissible_exposure_limits: [],
      recommended_exposure_limits: [],
      hazard_communication: [],
      personal_protective_equipment: []
    };
  }

  getDefaultROHSData() {
    return {
      restricted_substances: [],
      concentration_limits: {},
      compliance_status: 'unknown',
      testing_requirements: []
    };
  }
}

module.exports = RegulatoryComplianceService; 