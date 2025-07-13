const axios = require('axios');
const crypto = require('crypto');

class FreightosService {
  constructor() {
    // Ensure environment variables are loaded
    if (typeof require !== 'undefined' && require.main !== module) {
      require('dotenv').config();
    }
    
    this.apiKey = process.env.FREIGHTOS_API_KEY;
    this.secretKey = process.env.FREIGHTOS_SECRET_KEY;
    this.baseUrl = 'https://api.freightos.com';
    this.isConfigured = !!(this.apiKey && this.secretKey);
    
    // Only log once during initialization
    if (this.isConfigured) {
      console.log('✅ Freightos API credentials found. Real logistics calculations enabled.');
    } else {
      console.warn('⚠️ Freightos API credentials not found. Logistics calculations will use estimates.');
      console.warn('   Set FREIGHTOS_API_KEY and FREIGHTOS_SECRET_KEY in your .env file');
    }
  }

  /**
   * Get real-time freight rates and CO2 emissions (using CO2 endpoint)
   * Note: /api/v1/rates is not supported; /api/v1/co2calc returns both rates and emissions.
   */
  async getShippingRates(shipmentData) {
    if (!this.isConfigured) {
      throw new Error('❌ Freightos API credentials not found. Real logistics calculations are required.');
    }

    try {
      // Use the CO2 calculation endpoint for both rates and emissions
      const response = await this.makeAPICall('POST', '/api/v1/co2calc', shipmentData);
      return response;
    } catch (error) {
      console.error('❌ Freightos API error:', error.message);
      throw new Error(`Real logistics calculation failed: ${error.message}`);
    }
  }

  /**
   * Get CO2 emissions for different transport modes (using CO2 endpoint)
   */
  async getCO2Emissions(shipmentData) {
    if (!this.isConfigured) {
      throw new Error('❌ Freightos API credentials not found. Real CO2 calculations are required.');
    }

    try {
      // Use the CO2 calculation endpoint
      const response = await this.makeAPICall('POST', '/api/v1/co2calc', shipmentData);
      return response;
    } catch (error) {
      console.error('❌ Freightos CO2 API error:', error.message);
      throw new Error(`Real CO2 calculation failed: ${error.message}`);
    }
  }

  /**
   * Get global logistics network data
   */
  async getLogisticsNetwork(region = 'gulf') {
    try {
      if (!this.isConfigured) {
        return this.getFallbackNetwork(region);
      }

      const response = await this.makeFreightosRequest('/network/coverage', { region });
      
      return {
        success: true,
        network: {
          ports: response.ports || [],
          airports: response.airports || [],
          rail_stations: response.rail_stations || [],
          road_routes: response.road_routes || []
        },
        coverage_score: response.coverage_score || 0,
        connectivity_index: response.connectivity_index || 0
      };

    } catch (error) {
      console.error('❌ Network data error:', error.message);
      return this.getFallbackNetwork(region);
    }
  }

  /**
   * Make authenticated request to Freightos API
   */
  async makeFreightosRequest(endpoint, data) {
    const timestamp = Math.floor(Date.now() / 1000);
    const signature = this.generateSignature(endpoint, data, timestamp);

    const response = await axios({
      method: 'POST',
      url: `${this.baseUrl}${endpoint}`,
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey,
        'X-Signature': signature,
        'X-Timestamp': timestamp
      },
      data: data,
      timeout: 30000
    });

    return response.data;
  }

  /**
   * Generate API signature for authentication
   */
  generateSignature(endpoint, data, timestamp) {
    const message = `${endpoint}${JSON.stringify(data)}${timestamp}`;
    return crypto.createHmac('sha256', this.secretKey).update(message).digest('hex');
  }

  /**
   * Process Freightos rates response
   */
  processFreightosRates(rates) {
    return rates.map(rate => ({
      carrier: rate.carrier_name,
      service: rate.service_name,
      price_usd: rate.total_price,
      currency: rate.currency || 'USD',
      transit_days: rate.transit_days,
      delivery_date: rate.delivery_date,
      pickup_date: rate.pickup_date,
      transport_mode: rate.transport_mode,
      co2_kg: rate.co2_emissions || 0,
      reliability_score: rate.reliability_score || 0.8
    })).sort((a, b) => a.price_usd - b.price_usd);
  }

  /**
   * Calculate CO2 emissions from rates
   */
  calculateCO2Emissions(rates) {
    const totalCO2 = rates.reduce((sum, rate) => sum + (rate.co2_emissions || 0), 0);
    const avgCO2 = rates.length > 0 ? totalCO2 / rates.length : 0;
    
    return {
      total_kg: totalCO2,
      average_kg: avgCO2,
      lowest_kg: Math.min(...rates.map(r => r.co2_emissions || 0)),
      highest_kg: Math.max(...rates.map(r => r.co2_emissions || 0))
    };
  }

  /**
   * Extract unique carriers from rates
   */
  extractCarriers(rates) {
    return [...new Set(rates.map(rate => rate.carrier_name))];
  }

  /**
   * Calculate total emissions across modes
   */
  calculateTotalEmissions(response) {
    return Object.values(response).reduce((sum, mode) => {
      return sum + (mode?.co2_kg || 0);
    }, 0);
  }

  /**
   * Get recommended transport mode based on emissions
   */
  getRecommendedMode(response) {
    const modes = Object.entries(response).map(([mode, data]) => ({
      mode,
      co2: data?.co2_kg || 0
    }));
    
    return modes.reduce((lowest, current) => 
      current.co2 < lowest.co2 ? current : lowest
    ).mode;
  }

  /**
   * Make API call to Freightos
   */
  async makeAPICall(method, endpoint, data = null) {
    if (!this.isConfigured) {
      throw new Error('❌ Freightos API credentials not found. Real logistics calculations are required.');
    }

    try {
      const config = {
        method,
        url: `${this.baseUrl}${endpoint}`,
        headers: {
          'x-apikey': this.apiKey,
          'Content-Type': 'application/json',
          'User-Agent': 'ISM-AI-Platform/1.0'
        },
        timeout: 30000 // 30 second timeout
      };

      if (data) {
        config.data = data;
      }

      const response = await axios(config);
      return response.data;

    } catch (error) {
      console.error(`❌ Freightos API error (${method} ${endpoint}):`, error.message);
      throw new Error(`Real logistics calculation failed: ${error.message}`);
    }
  }

  /**
   * Fallback network data
   */
  getFallbackNetwork(region) {
    return {
      success: false,
      network: {
        ports: ['Jebel Ali', 'Jeddah', 'Doha', 'Kuwait'],
        airports: ['DXB', 'RUH', 'DOH', 'KWI'],
        rail_stations: ['Gulf Railway Network'],
        road_routes: ['GCC Highway Network']
      },
      coverage_score: 0.8,
      connectivity_index: 0.7
    };
  }

  /**
   * Calculate distance between two points
   */
  calculateDistance(origin, destination) {
    // Simple distance calculation (can be enhanced with real geocoding)
    const lat1 = this.getLatitude(origin.city);
    const lon1 = this.getLongitude(origin.city);
    const lat2 = this.getLatitude(destination.city);
    const lon2 = this.getLongitude(destination.city);
    
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    
    return R * c;
  }

  /**
   * Get latitude for major Gulf cities
   */
  getLatitude(city) {
    const cities = {
      'Dubai': 25.2048,
      'Abu Dhabi': 24.4539,
      'Riyadh': 24.7136,
      'Jeddah': 21.4858,
      'Doha': 25.2854,
      'Kuwait City': 29.3759,
      'Muscat': 23.5880,
      'Manama': 26.2285
    };
    return cities[city] || 25.2048; // Default to Dubai
  }

  /**
   * Get longitude for major Gulf cities
   */
  getLongitude(city) {
    const cities = {
      'Dubai': 55.2708,
      'Abu Dhabi': 54.3773,
      'Riyadh': 46.6753,
      'Jeddah': 39.1925,
      'Doha': 51.5310,
      'Kuwait City': 47.9774,
      'Muscat': 58.3829,
      'Manama': 50.5860
    };
    return cities[city] || 55.2708; // Default to Dubai
  }
}

// Singleton pattern to ensure only one instance
let freightosServiceInstance = null;

function getFreightosService() {
  if (!freightosServiceInstance) {
    freightosServiceInstance = new FreightosService();
  }
  return freightosServiceInstance;
}

module.exports = getFreightosService(); 