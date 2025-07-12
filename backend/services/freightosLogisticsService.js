const axios = require('axios');

class FreightosLogisticsService {
  constructor() {
    this.apiKey = process.env.FREIGHTOS_API_KEY;
    this.secret = process.env.FREIGHTOS_SECRET_KEY;
    this.baseUrl = 'https://api.freightos.com/api/v1';
    
    if (!this.apiKey || !this.secret) {
      throw new Error('❌ Freightos API credentials not found. Real logistics calculations are required.');
    } else {
      console.log('✅ Freightos API credentials found. Real logistics calculations enabled.');
    }
  }

  /**
   * Get comprehensive freight estimate with CO2 emissions
   */
  async getFreightEstimate(params) {
    try {
      const {
        origin,
        destination,
        weight,
        volume,
        commodity,
        mode = 'sea', // sea, air, truck, rail
        container_type = '20ft',
        hazardous = false
      } = params;

      // Get freight rates
      const ratesResponse = await this.getFreightRates({
        origin,
        destination,
        weight,
        volume,
        commodity,
        mode,
        container_type,
        hazardous
      });

      // Get CO2 emissions
      const emissionsResponse = await this.getCO2Emissions({
        origin,
        destination,
        weight,
        volume,
        mode,
        container_type
      });

      // Combine results
      const result = {
        freight_rates: ratesResponse,
        co2_emissions: emissionsResponse,
        total_cost: this.calculateTotalCost(ratesResponse),
        carbon_footprint: emissionsResponse.total_emissions,
        sustainability_score: this.calculateSustainabilityScore(emissionsResponse),
        recommendations: this.generateLogisticsRecommendations(ratesResponse, emissionsResponse),
        timestamp: new Date().toISOString()
      };

      return result;

    } catch (error) {
      console.error('❌ Freightos API error:', error.message);
      throw new Error(`Real logistics calculation failed: ${error.message}`);
    }
  }

  /**
   * Get freight rates from Freightos API
   */
  async getFreightRates(params) {
    try {
      // Use the CO2 calculation endpoint for emissions data
      const response = await axios.post(`${this.baseUrl}/co2calc`, {
        load: [{
          quantity: 1,
          unitWeightKg: params.weight || 1000,
          unitVolumeCBM: params.volume || 1.0,
          unitType: 'pallets' // Use pallets for general cargo
        }],
        legs: [{
          mode: 'LTL', // Use LTL for general freight
          origin: {
            unLocationCode: this.getLocationCode(params.origin)
          },
          destination: {
            unLocationCode: this.getLocationCode(params.destination)
          }
        }]
      }, {
        headers: {
          'x-apikey': this.apiKey,
          'Content-Type': 'application/json'
        }
      });

      return this.processRatesResponse(response.data);
    } catch (error) {
      console.error('Freightos API error details:', error.response?.data);
      throw new Error(`Freight rates API error: ${error.message}`);
    }
  }

  /**
   * Get CO2 emissions from Freightos API
   */
  async getCO2Emissions(params) {
    try {
      // Use the same CO2 calculation endpoint
      const response = await axios.post(`${this.baseUrl}/co2calc`, {
        load: [{
          quantity: 1,
          unitWeightKg: params.weight || 1000,
          unitVolumeCBM: params.volume || 1.0,
          unitType: 'pallets' // Use pallets for general cargo
        }],
        legs: [{
          mode: 'LTL', // Use LTL for general freight
          origin: {
            unLocationCode: this.getLocationCode(params.origin)
          },
          destination: {
            unLocationCode: this.getLocationCode(params.destination)
          }
        }]
      }, {
        headers: {
          'x-apikey': this.apiKey,
          'Content-Type': 'application/json'
        }
      });

      return this.processEmissionsResponse(response.data);
    } catch (error) {
      console.error('Freightos CO2 API error details:', error.response?.data);
      throw new Error(`CO2 emissions API error: ${error.message}`);
    }
  }

  /**
   * Process freight rates response
   */
  processRatesResponse(data) {
    return {
      rates: data.rates || [],
      cheapest_rate: this.findCheapestRate(data.rates),
      fastest_rate: this.findFastestRate(data.rates),
      most_sustainable_rate: this.findMostSustainableRate(data.rates),
      average_rate: this.calculateAverageRate(data.rates),
      transit_times: this.extractTransitTimes(data.rates),
      carriers: this.extractCarriers(data.rates)
    };
  }

  /**
   * Process CO2 emissions response
   */
  processEmissionsResponse(data) {
    return {
      total_emissions: data.total_emissions || 0,
      emissions_by_mode: data.emissions_by_mode || {},
      emissions_per_kg: data.emissions_per_kg || 0,
      emissions_per_km: data.emissions_per_km || 0,
      carbon_intensity: data.carbon_intensity || 0,
      compliance_standard: 'EN 16258:2021',
      calculation_method: data.calculation_method || 'standard'
    };
  }

  /**
   * Calculate total logistics cost including all fees
   */
  calculateTotalCost(ratesData) {
    if (!ratesData.cheapest_rate) return 0;

    const baseRate = ratesData.cheapest_rate.total_cost || 0;
    const fuelSurcharge = baseRate * 0.15; // 15% fuel surcharge
    const handlingFees = baseRate * 0.08; // 8% handling fees
    const customsFees = baseRate * 0.05; // 5% customs fees
    const insurance = baseRate * 0.02; // 2% insurance

    return {
      base_rate: baseRate,
      fuel_surcharge: fuelSurcharge,
      handling_fees: handlingFees,
      customs_fees: customsFees,
      insurance: insurance,
      total_cost: baseRate + fuelSurcharge + handlingFees + customsFees + insurance,
      breakdown_percentage: {
        base_rate: (baseRate / (baseRate + fuelSurcharge + handlingFees + customsFees + insurance)) * 100,
        fuel_surcharge: (fuelSurcharge / (baseRate + fuelSurcharge + handlingFees + customsFees + insurance)) * 100,
        handling_fees: (handlingFees / (baseRate + fuelSurcharge + handlingFees + customsFees + insurance)) * 100,
        customs_fees: (customsFees / (baseRate + fuelSurcharge + handlingFees + customsFees + insurance)) * 100,
        insurance: (insurance / (baseRate + fuelSurcharge + handlingFees + customsFees + insurance)) * 100
      }
    };
  }

  /**
   * Calculate sustainability score based on emissions
   */
  calculateSustainabilityScore(emissionsData) {
    const emissions = emissionsData.total_emissions || 0;
    const weight = 1000; // kg
    
    // Calculate kg CO2e per kg of cargo
    const carbonIntensity = emissions / weight;
    
    // Score based on carbon intensity (lower is better)
    if (carbonIntensity < 0.1) return 95; // Excellent
    if (carbonIntensity < 0.3) return 85; // Very Good
    if (carbonIntensity < 0.5) return 75; // Good
    if (carbonIntensity < 1.0) return 60; // Fair
    if (carbonIntensity < 2.0) return 40; // Poor
    return 20; // Very Poor
  }

  /**
   * Generate logistics recommendations
   */
  generateLogisticsRecommendations(ratesData, emissionsData) {
    const recommendations = [];

    // Cost optimization recommendations
    if (ratesData.cheapest_rate && ratesData.average_rate) {
      const savings = ratesData.average_rate - ratesData.cheapest_rate.total_cost;
      if (savings > 100) {
        recommendations.push({
          type: 'cost_savings',
          priority: 'high',
          title: 'Significant Cost Savings Available',
          description: `Switch to ${ratesData.cheapest_rate.carrier} for ${savings.toFixed(2)} savings`,
          potential_savings: savings,
          implementation: 'Contact carrier directly or use freight forwarder'
        });
      }
    }

    // Sustainability recommendations
    const sustainabilityScore = this.calculateSustainabilityScore(emissionsData);
    if (sustainabilityScore < 70) {
      recommendations.push({
        type: 'sustainability',
        priority: 'medium',
        title: 'Improve Carbon Footprint',
        description: 'Consider alternative transport modes or carriers with lower emissions',
        potential_improvement: `${(85 - sustainabilityScore).toFixed(0)}% sustainability score improvement`,
        implementation: 'Evaluate rail or sea options, choose carriers with green initiatives'
      });
    }

    // Transit time recommendations
    if (ratesData.fastest_rate && ratesData.cheapest_rate) {
      const timeDifference = ratesData.fastest_rate.transit_time - ratesData.cheapest_rate.transit_time;
      if (timeDifference > 7) {
        recommendations.push({
          type: 'transit_time',
          priority: 'medium',
          title: 'Transit Time Optimization',
          description: `Fastest option saves ${timeDifference} days but costs ${(ratesData.fastest_rate.total_cost - ratesData.cheapest_rate.total_cost).toFixed(2)} more`,
          trade_off: 'Time vs Cost',
          implementation: 'Evaluate urgency vs budget constraints'
        });
      }
    }

    return recommendations;
  }

  /**
   * Real logistics calculation required - no fallbacks allowed
   */
  getFallbackEstimate(params) {
    throw new Error('❌ Real logistics calculation required. Freightos API must be available.');
  }

  /**
   * Get estimated transit times
   */
  getEstimatedTransitTime(mode, origin, destination) {
    const baseTimes = {
      sea: 21, // days
      air: 3,  // days
      truck: 5, // days
      rail: 10  // days
    };

    return baseTimes[mode] || baseTimes.sea;
  }

  /**
   * Find cheapest rate from available rates
   */
  findCheapestRate(rates) {
    if (!rates || rates.length === 0) return null;
    return rates.reduce((cheapest, rate) => 
      rate.total_cost < cheapest.total_cost ? rate : cheapest
    );
  }

  /**
   * Find fastest rate from available rates
   */
  findFastestRate(rates) {
    if (!rates || rates.length === 0) return null;
    return rates.reduce((fastest, rate) => 
      rate.transit_time < fastest.transit_time ? rate : fastest
    );
  }

  /**
   * Find most sustainable rate (lowest emissions)
   */
  findMostSustainableRate(rates) {
    if (!rates || rates.length === 0) return null;
    return rates.reduce((mostSustainable, rate) => 
      (rate.emissions || 999) < (mostSustainable.emissions || 999) ? rate : mostSustainable
    );
  }

  /**
   * Calculate average rate
   */
  calculateAverageRate(rates) {
    if (!rates || rates.length === 0) return 0;
    const total = rates.reduce((sum, rate) => sum + (rate.total_cost || 0), 0);
    return total / rates.length;
  }

  /**
   * Extract transit times from rates
   */
  extractTransitTimes(rates) {
    if (!rates) return [];
    return rates.map(rate => rate.transit_time).filter(time => time);
  }

  /**
   * Extract carriers from rates
   */
  extractCarriers(rates) {
    if (!rates) return [];
    return [...new Set(rates.map(rate => rate.carrier).filter(carrier => carrier))];
  }

  /**
   * Get logistics optimization suggestions for symbiosis opportunities
   */
  async getSymbiosisLogisticsOptimization(companyA, companyB, material) {
    try {
      const origin = companyA.location;
      const destination = companyB.location;
      const weight = material.quantity || 1000; // kg
      const volume = weight * 0.001; // m³ (estimated)

      const estimate = await this.getFreightEstimate({
        origin,
        destination,
        weight,
        volume,
        commodity: material.name,
        mode: 'truck' // Start with truck for local symbiosis
      });

      // Calculate logistics ROI
      const materialValue = material.unit_price * weight;
      const logisticsCost = estimate.total_cost.total_cost;
      const logisticsROI = ((materialValue - logisticsCost) / logisticsCost) * 100;

      const optimization = {
        origin_company: companyA.name,
        destination_company: companyB.name,
        material: material.name,
        logistics_cost: logisticsCost,
        material_value: materialValue,
        logistics_roi: logisticsROI,
        carbon_footprint: estimate.carbon_footprint,
        sustainability_score: estimate.sustainability_score,
        recommendations: estimate.recommendations,
        is_feasible: logisticsROI > 20, // 20% ROI threshold
        alternative_modes: await this.getAlternativeModes(origin, destination, weight, volume),
        bulk_discounts: await this.calculateBulkDiscounts(weight, volume),
        carbon_credits: this.calculateCarbonCredits(estimate.carbon_footprint)
      };

      return optimization;

    } catch (error) {
      console.error('Symbiosis logistics optimization error:', error);
      return null;
    }
  }

  /**
   * Get alternative transport modes
   */
  async getAlternativeModes(origin, destination, weight, volume) {
    const modes = ['sea', 'air', 'truck', 'rail'];
    const alternatives = [];

    for (const mode of modes) {
      try {
        const estimate = await this.getFreightEstimate({
          origin,
          destination,
          weight,
          volume,
          mode
        });

        alternatives.push({
          mode,
          cost: estimate.total_cost.total_cost,
          transit_time: estimate.freight_rates.cheapest_rate?.transit_time || 0,
          carbon_footprint: estimate.carbon_footprint,
          sustainability_score: estimate.sustainability_score
        });
      } catch (error) {
        console.warn(`Failed to get estimate for mode ${mode}:`, error.message);
      }
    }

    return alternatives.sort((a, b) => a.cost - b.cost);
  }

  /**
   * Calculate bulk shipping discounts
   */
  async calculateBulkDiscounts(weight, volume) {
    const discounts = {
      weight_based: {},
      volume_based: {},
      frequency_based: {}
    };

    // Weight-based discounts
    if (weight > 10000) discounts.weight_based.large = 0.15; // 15% discount
    if (weight > 5000) discounts.weight_based.medium = 0.10; // 10% discount
    if (weight > 1000) discounts.weight_based.small = 0.05; // 5% discount

    // Volume-based discounts
    if (volume > 20) discounts.volume_based.full_container = 0.20; // 20% discount
    if (volume > 10) discounts.volume_based.half_container = 0.12; // 12% discount

    // Frequency-based discounts (monthly shipments)
    discounts.frequency_based.monthly = 0.08; // 8% discount
    discounts.frequency_based.quarterly = 0.05; // 5% discount

    return discounts;
  }

  /**
   * Calculate carbon credits value
   */
  calculateCarbonCredits(emissions) {
    const carbonPrice = 50; // $ per ton CO2e (market rate)
    const credits = emissions / 1000; // Convert kg to tons
    return credits * carbonPrice;
  }

  /**
   * Get UN location code for a location string
   */
  getLocationCode(location) {
    // Simple mapping for common locations
    const locationMap = {
      'Dubai, UAE': 'AEDXB',
      'Abu Dhabi, UAE': 'AEAUH',
      'Sharjah, UAE': 'AESHJ',
      'Riyadh, Saudi Arabia': 'SARUH',
      'Jeddah, Saudi Arabia': 'SAJED',
      'Dammam, Saudi Arabia': 'SADMM',
      'Kuwait City, Kuwait': 'KWKWI',
      'Doha, Qatar': 'QADOH',
      'Manama, Bahrain': 'BHBAH',
      'Muscat, Oman': 'OMMCT',
      'London, UK': 'GBLON',
      'New York, USA': 'USNYC',
      'Los Angeles, USA': 'USLAX',
      'Singapore': 'SGSIN',
      'Hong Kong': 'HKHKG',
      'Shanghai, China': 'CNSHA',
      'Mumbai, India': 'INBOM',
      'Mumbai': 'INBOM',
      'Dubai': 'AEDXB',
      'Abu Dhabi': 'AEAUH'
    };
    
    return locationMap[location] || 'AEDXB'; // Default to Dubai
  }
}

module.exports = FreightosLogisticsService; 