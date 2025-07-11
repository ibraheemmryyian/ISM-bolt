const axios = require('axios');
const crypto = require('crypto');
const { supabase } = require('../supabase');

class APIFusionService {
  constructor() {
    this.apiConfigs = {
      shippo: {
        baseUrl: 'https://api.goshippo.com',
        apiKey: process.env.SHIPPO_API_KEY,
        timeout: 30000,
        retries: 3
      },
      nextGenMaterials: {
        baseUrl: 'https://api.nextgenmaterials.com',
        apiKey: process.env.NEXTGEN_MATERIALS_API_KEY,
        timeout: 15000,
        retries: 2
      },
      iWaste: {
        baseUrl: 'https://api.iwaste.com',
        apiKey: process.env.IWASTE_API_KEY,
        timeout: 20000,
        retries: 2
      },
      recycleNation: {
        baseUrl: 'https://api.recyclenation.com',
        apiKey: process.env.RECYCLENATION_API_KEY,
        timeout: 15000,
        retries: 2
      }
    };
    
    this.materialProperties = new Map();
    this.shippingStandards = new Map();
    this.conversionFactors = {
      weight: {
        metric_ton: 1,
        kg: 0.001,
        lb: 0.000453592,
        short_ton: 0.907185
      },
      volume: {
        cubic_meter: 1,
        cubic_feet: 0.0283168,
        liters: 0.001,
        gallons: 0.00378541
      }
    };
  }

  /**
   * Universal material property translator
   * Converts material descriptions into standardized shipping parameters
   */
  async translateMaterialToShippingParams(materialData) {
    try {
      const {
        material_name,
        category,
        state,
        quality_grade,
        quantity,
        unit,
        description
      } = materialData;

      // Get material properties from Next-Gen Materials API
      const materialProperties = await this.getMaterialProperties(material_name, category);
      
      // Calculate shipping parameters
      const shippingParams = this.calculateShippingParams(
        materialProperties,
        quantity,
        unit,
        state,
        quality_grade
      );

      // Determine hazard classification
      const hazardClass = await this.determineHazardClass(materialProperties, state);

      return {
        weight_kg: shippingParams.weight_kg,
        volume_cubic_meters: shippingParams.volume_cubic_meters,
        density_kg_per_m3: shippingParams.density,
        hazard_class: hazardClass,
        special_handling: shippingParams.special_handling,
        packaging_requirements: shippingParams.packaging,
        temperature_requirements: shippingParams.temperature,
        material_properties: materialProperties
      };
    } catch (error) {
      console.error('Material translation error:', error);
      return this.getFallbackShippingParams(materialData);
    }
  }

  /**
   * Get material properties from Next-Gen Materials API
   */
  async getMaterialProperties(materialName, category) {
    const cacheKey = `${materialName}_${category}`;
    
    // Check cache first
    if (this.materialProperties.has(cacheKey)) {
      return this.materialProperties.get(cacheKey);
    }

    try {
      const response = await this.makeAPICall('nextGenMaterials', 'GET', `/materials/search`, {
        params: {
          name: materialName,
          category: category
        }
      });

      const properties = {
        density: response.data.density_kg_per_m3 || 1000,
        melting_point: response.data.melting_point_celsius,
        boiling_point: response.data.boiling_point_celsius,
        chemical_formula: response.data.chemical_formula,
        hazard_classification: response.data.hazard_classification,
        recycling_code: response.data.recycling_code,
        sustainability_score: response.data.sustainability_score,
        circular_economy_potential: response.data.circular_economy_potential
      };

      // Cache the result
      this.materialProperties.set(cacheKey, properties);
      
      return properties;
    } catch (error) {
      console.error('Error fetching material properties:', error);
      return this.getDefaultMaterialProperties(materialName, category);
    }
  }

  /**
   * Calculate shipping parameters from material properties
   */
  calculateShippingParams(materialProperties, quantity, unit, state, qualityGrade) {
    // Convert quantity to metric tons
    const quantityInTons = this.convertToMetricTons(quantity, unit);
    
    // Calculate volume based on density
    const density = materialProperties.density || 1000; // kg/m³
    const volumeCubicMeters = (quantityInTons * 1000) / density;
    
    // Determine special handling requirements
    const specialHandling = this.determineSpecialHandling(materialProperties, state, qualityGrade);
    
    // Determine packaging requirements
    const packaging = this.determinePackagingRequirements(materialProperties, state, quantityInTons);
    
    // Determine temperature requirements
    const temperature = this.determineTemperatureRequirements(materialProperties, state);

    return {
      weight_kg: quantityInTons * 1000,
      volume_cubic_meters: volumeCubicMeters,
      density: density,
      special_handling: specialHandling,
      packaging: packaging,
      temperature: temperature
    };
  }

  /**
   * Convert any weight unit to metric tons
   */
  convertToMetricTons(quantity, unit) {
    const conversionFactors = {
      'metric_ton': 1,
      'ton': 1,
      'kg': 0.001,
      'pound': 0.000453592,
      'lb': 0.000453592,
      'short_ton': 0.907185,
      'long_ton': 1.01605
    };

    const factor = conversionFactors[unit.toLowerCase()] || 1;
    return quantity * factor;
  }

  /**
   * Determine hazard classification
   */
  async determineHazardClass(materialProperties, state) {
    if (materialProperties.hazard_classification) {
      return materialProperties.hazard_classification;
    }

    // Check iWaste API for hazard information
    try {
      const response = await this.makeAPICall('iWaste', 'GET', '/hazard-classification', {
        params: {
          material: materialProperties.chemical_formula,
          state: state
        }
      });
      return response.data.hazard_class;
    } catch (error) {
      return 'NON_HAZARDOUS';
    }
  }

  /**
   * Determine special handling requirements
   */
  determineSpecialHandling(materialProperties, state, qualityGrade) {
    const requirements = [];

    if (materialProperties.melting_point && materialProperties.melting_point < 50) {
      requirements.push('TEMPERATURE_CONTROLLED');
    }

    if (materialProperties.hazard_classification && materialProperties.hazard_classification !== 'NON_HAZARDOUS') {
      requirements.push('HAZARDOUS_MATERIALS');
    }

    if (qualityGrade === 'high') {
      requirements.push('FRAGILE_HANDLING');
    }

    if (state === 'liquid' || state === 'gas') {
      requirements.push('CONTAINERIZED');
    }

    return requirements;
  }

  /**
   * Determine packaging requirements
   */
  determinePackagingRequirements(materialProperties, state, quantityTons) {
    if (quantityTons > 10) {
      return 'BULK_CONTAINER';
    } else if (quantityTons > 1) {
      return 'PALLETIZED';
    } else {
      return 'STANDARD_PACKAGING';
    }
  }

  /**
   * Determine temperature requirements
   */
  determineTemperatureRequirements(materialProperties, state) {
    if (materialProperties.melting_point) {
      return {
        min_temp: materialProperties.melting_point - 10,
        max_temp: materialProperties.melting_point + 50,
        unit: 'celsius'
      };
    }
    return null;
  }

  /**
   * Universal API call handler with retry logic and error handling
   */
  async makeAPICall(apiName, method, endpoint, options = {}) {
    const config = this.apiConfigs[apiName];
    if (!config) {
      throw new Error(`Unknown API: ${apiName}`);
    }

    const requestId = crypto.randomUUID();
    const startTime = Date.now();

    for (let attempt = 1; attempt <= config.retries; attempt++) {
      try {
        const response = await axios({
          method,
          url: `${config.baseUrl}${endpoint}`,
          headers: {
            'Authorization': `Bearer ${config.apiKey}`,
            'Content-Type': 'application/json',
            'X-Request-ID': requestId,
            ...options.headers
          },
          params: options.params,
          data: options.data,
          timeout: config.timeout
        });

        // Log successful API call
        await this.logAPICall(apiName, endpoint, method, 'success', Date.now() - startTime, requestId);

        return response;
      } catch (error) {
        const duration = Date.now() - startTime;
        
        // Log failed API call
        await this.logAPICall(apiName, endpoint, method, 'error', duration, requestId, error.message);

        if (attempt === config.retries) {
          throw new Error(`API call failed after ${config.retries} attempts: ${error.message}`);
        }

        // Exponential backoff
        await this.delay(Math.pow(2, attempt) * 1000);
      }
    }
  }

  /**
   * Log API calls for monitoring and debugging
   */
  async logAPICall(apiName, endpoint, method, status, duration, requestId, errorMessage = null) {
    try {
      await supabase.from('api_call_logs').insert({
        api_name: apiName,
        endpoint: endpoint,
        method: method,
        status: status,
        duration_ms: duration,
        request_id: requestId,
        error_message: errorMessage,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to log API call:', error);
    }
  }

  /**
   * Get fallback shipping parameters when API calls fail
   */
  getFallbackShippingParams(materialData) {
    const { quantity, unit, state } = materialData;
    const quantityInTons = this.convertToMetricTons(quantity, unit);
    
    return {
      weight_kg: quantityInTons * 1000,
      volume_cubic_meters: quantityInTons * 1.5, // Rough estimate
      density_kg_per_m3: 1000,
      hazard_class: 'NON_HAZARDOUS',
      special_handling: [],
      packaging_requirements: 'STANDARD_PACKAGING',
      temperature_requirements: null
    };
  }

  /**
   * Get default material properties when API is unavailable
   */
  getDefaultMaterialProperties(materialName, category) {
    const defaults = {
      'steel': { density: 7850, melting_point: 1370 },
      'aluminum': { density: 2700, melting_point: 660 },
      'plastic': { density: 900, melting_point: 150 },
      'wood': { density: 600, melting_point: null },
      'glass': { density: 2500, melting_point: 1500 },
      'paper': { density: 800, melting_point: null }
    };

    return defaults[category.toLowerCase()] || { density: 1000, melting_point: null };
  }

  /**
   * Utility function for delays
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = new APIFusionService(); 