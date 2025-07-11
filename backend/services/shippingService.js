const axios = require('axios');
const apiFusionService = require('./apiFusionService');
const { supabase } = require('../supabase');

class ShippingService {
  constructor() {
    this.shippoApiKey = process.env.SHIPPO_API_KEY;
    this.baseUrl = 'https://api.goshippo.com';
    this.isTestMode = this.shippoApiKey && this.shippoApiKey.startsWith('shippo_test_');
  }

  /**
   * Get shipping rates for material exchange
   * Handles bulk shipments measured in metric tons
   */
  async getShippingRates(materialData, fromAddress, toAddress) {
    try {
      console.log('🚢 Getting shipping rates for material:', materialData.material_name);
      
      // Translate material to shipping parameters
      const shippingParams = await apiFusionService.translateMaterialToShippingParams(materialData);
      
      // Prepare shipment data
      const shipmentData = {
        address_from: this.formatAddress(fromAddress),
        address_to: this.formatAddress(toAddress),
        parcels: [{
          length: this.calculateParcelDimensions(shippingParams.volume_cubic_meters).length,
          width: this.calculateParcelDimensions(shippingParams.volume_cubic_meters).width,
          height: this.calculateParcelDimensions(shippingParams.volume_cubic_meters).height,
          distance_unit: 'cm',
          weight: shippingParams.weight_kg,
          mass_unit: 'kg'
        }],
        async: false
      };

      // Add special handling requirements
      if (shippingParams.special_handling.length > 0) {
        shipmentData.extra = {
          special_handling: shippingParams.special_handling,
          hazard_class: shippingParams.hazard_class,
          packaging_requirements: shippingParams.packaging_requirements
        };
      }

      // Make API call to Shippo
      const response = await this.makeShippoRequest('/shipments/', shipmentData);
      
      // Process and format rates
      const rates = this.processShippingRates(response.rates, shippingParams);
      
      // Log the shipping request
      await this.logShippingRequest(materialData, fromAddress, toAddress, rates);
      
      return {
        success: true,
        rates: rates,
        shipping_params: shippingParams,
        test_mode: this.isTestMode
      };
      
    } catch (error) {
      console.error('❌ Shipping rate calculation error:', error);
      return {
        success: false,
        error: error.message,
        fallback_rates: this.getFallbackRates(materialData, fromAddress, toAddress)
      };
    }
  }

  /**
   * Create shipping label for material exchange
   */
  async createShippingLabel(materialData, fromAddress, toAddress, rateId) {
    try {
      console.log('🏷️ Creating shipping label for material:', materialData.material_name);
      
      // Get shipping parameters
      const shippingParams = await apiFusionService.translateMaterialToShippingParams(materialData);
      
      // Prepare label data
      const labelData = {
        rate: rateId,
        label_file_type: 'PDF',
        async: false
      };

      // Add customs information for international shipments
      if (this.isInternationalShipment(fromAddress, toAddress)) {
        labelData.metadata = `Material: ${materialData.material_name}, Weight: ${shippingParams.weight_kg}kg`;
      }

      // Make API call to Shippo
      const response = await this.makeShippoRequest('/transactions/', labelData);
      
      // Log the label creation
      await this.logLabelCreation(materialData, fromAddress, toAddress, response);
      
      return {
        success: true,
        label_url: response.label_url,
        tracking_number: response.tracking_number,
        tracking_url: response.tracking_url_provider,
        estimated_delivery: response.eta,
        shipping_cost: response.rate.amount,
        test_mode: this.isTestMode
      };
      
    } catch (error) {
      console.error('❌ Label creation error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Track shipment status
   */
  async trackShipment(trackingNumber, carrier) {
    try {
      console.log('📦 Tracking shipment:', trackingNumber);
      
      const response = await this.makeShippoRequest(`/tracks/${carrier}/${trackingNumber}/`);
      
      return {
        success: true,
        tracking_number: trackingNumber,
        status: response.tracking_status.status,
        location: response.tracking_status.location,
        eta: response.tracking_status.eta,
        history: response.tracking_history,
        test_mode: this.isTestMode
      };
      
    } catch (error) {
      console.error('❌ Shipment tracking error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Validate address for shipping
   */
  async validateAddress(address) {
    try {
      const formattedAddress = this.formatAddress(address);
      
      const response = await this.makeShippoRequest('/addresses/', formattedAddress);
      
      return {
        success: true,
        validated_address: response,
        is_valid: true
      };
      
    } catch (error) {
      console.error('❌ Address validation error:', error);
      return {
        success: false,
        error: error.message,
        is_valid: false
      };
    }
  }

  /**
   * Calculate parcel dimensions from volume
   */
  calculateParcelDimensions(volumeCubicMeters) {
    // Convert to cubic centimeters
    const volumeCm3 = volumeCubicMeters * 1000000;
    
    // Calculate cube root for approximate dimensions
    const sideLength = Math.cbrt(volumeCm3);
    
    return {
      length: Math.ceil(sideLength),
      width: Math.ceil(sideLength),
      height: Math.ceil(sideLength)
    };
  }

  /**
   * Format address for Shippo API
   */
  formatAddress(address) {
    return {
      name: address.name || 'Company Name',
      company: address.company || '',
      street1: address.street1 || address.address || '',
      street2: address.street2 || '',
      city: address.city || '',
      state: address.state || address.province || '',
      zip: address.zip || address.postal_code || '',
      country: address.country || 'US',
      phone: address.phone || '',
      email: address.email || ''
    };
  }

  /**
   * Check if shipment is international
   */
  isInternationalShipment(fromAddress, toAddress) {
    return fromAddress.country !== toAddress.country;
  }

  /**
   * Process shipping rates from Shippo response
   */
  processShippingRates(rates, shippingParams) {
    return rates.map(rate => ({
      rate_id: rate.object_id,
      service: rate.servicelevel.name,
      carrier: rate.provider,
      delivery_days: rate.estimated_days,
      price: parseFloat(rate.amount),
      currency: rate.currency,
      special_handling: shippingParams.special_handling,
      packaging_requirements: shippingParams.packaging_requirements,
      temperature_requirements: shippingParams.temperature_requirements
    })).sort((a, b) => a.price - b.price);
  }

  /**
   * Make request to Shippo API with error handling
   */
  async makeShippoRequest(endpoint, data) {
    try {
      const response = await axios({
        method: 'POST',
        url: `${this.baseUrl}${endpoint}`,
        headers: {
          'Authorization': `ShippoToken ${this.shippoApiKey}`,
          'Content-Type': 'application/json'
        },
        data: data,
        timeout: 30000
      });

      return response.data;
    } catch (error) {
      console.error('Shippo API error:', error.response?.data || error.message);
      throw new Error(`Shippo API error: ${error.response?.data?.detail || error.message}`);
    }
  }

  /**
   * Get fallback rates when API is unavailable
   */
  getFallbackRates(materialData, fromAddress, toAddress) {
    const distance = this.calculateDistance(fromAddress, toAddress);
    const basePrice = distance * 0.5; // $0.50 per km
    
    return [
      {
        rate_id: 'fallback_1',
        service: 'Standard Ground',
        carrier: 'Fallback Carrier',
        delivery_days: Math.ceil(distance / 500), // 500 km per day
        price: basePrice,
        currency: 'USD',
        special_handling: [],
        packaging_requirements: 'STANDARD_PACKAGING'
      },
      {
        rate_id: 'fallback_2',
        service: 'Express',
        carrier: 'Fallback Carrier',
        delivery_days: Math.ceil(distance / 1000), // 1000 km per day
        price: basePrice * 2,
        currency: 'USD',
        special_handling: [],
        packaging_requirements: 'STANDARD_PACKAGING'
      }
    ];
  }

  /**
   * Calculate distance between addresses (simplified)
   */
  calculateDistance(fromAddress, toAddress) {
    // Simplified distance calculation
    // In production, use a proper geocoding service
    return 100; // Default 100 km
  }

  /**
   * Log shipping request for analytics
   */
  async logShippingRequest(materialData, fromAddress, toAddress, rates) {
    try {
      await supabase.from('shipping_requests').insert({
        material_name: materialData.material_name,
        material_type: materialData.type,
        quantity: materialData.quantity,
        unit: materialData.unit,
        from_location: `${fromAddress.city}, ${fromAddress.state}`,
        to_location: `${toAddress.city}, ${toAddress.state}`,
        rates_count: rates.length,
        lowest_price: rates.length > 0 ? rates[0].price : 0,
        test_mode: this.isTestMode,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to log shipping request:', error);
    }
  }

  /**
   * Log label creation for analytics
   */
  async logLabelCreation(materialData, fromAddress, toAddress, labelResponse) {
    try {
      await supabase.from('shipping_labels').insert({
        material_name: materialData.material_name,
        tracking_number: labelResponse.tracking_number,
        from_location: `${fromAddress.city}, ${fromAddress.state}`,
        to_location: `${toAddress.city}, ${toAddress.state}`,
        shipping_cost: labelResponse.rate.amount,
        carrier: labelResponse.rate.provider,
        service: labelResponse.rate.servicelevel.name,
        test_mode: this.isTestMode,
        created_at: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to log label creation:', error);
    }
  }

  /**
   * Get shipping statistics
   */
  async getShippingStats(companyId) {
    try {
      const { data: requests } = await supabase
        .from('shipping_requests')
        .select('*')
        .eq('company_id', companyId);

      const { data: labels } = await supabase
        .from('shipping_labels')
        .select('*')
        .eq('company_id', companyId);

      return {
        total_requests: requests?.length || 0,
        total_shipments: labels?.length || 0,
        total_spent: labels?.reduce((sum, label) => sum + parseFloat(label.shipping_cost), 0) || 0,
        average_cost: labels?.length > 0 ? 
          labels.reduce((sum, label) => sum + parseFloat(label.shipping_cost), 0) / labels.length : 0
      };
    } catch (error) {
      console.error('Error getting shipping stats:', error);
      return {
        total_requests: 0,
        total_shipments: 0,
        total_spent: 0,
        average_cost: 0
      };
    }
  }
}

module.exports = new ShippingService(); 