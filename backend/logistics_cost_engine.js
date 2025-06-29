/**
 * Advanced Logistics & Cost Engine for Industrial Symbiosis
 * Handles multi-modal transportation, cost optimization, and carbon impact analysis
 */

const TransportMode = {
  TRUCK: 'truck',
  TRAIN: 'train',
  SHIP: 'ship',
  AIR: 'air',
  MULTIMODAL: 'multimodal'
};

class Location {
  constructor(name, latitude, longitude, country, city, postal_code = '') {
    this.name = name;
    this.latitude = latitude;
    this.longitude = longitude;
    this.country = country;
    this.city = city;
    this.postal_code = postal_code;
  }
}

class RouteSegment {
  constructor(mode, origin, destination, distance_km, duration_hours, cost_euro, carbon_kg, capacity_ton, reliability_score) {
    this.mode = mode;
    this.origin = origin;
    this.destination = destination;
    this.distance_km = distance_km;
    this.duration_hours = duration_hours;
    this.cost_euro = cost_euro;
    this.carbon_kg = carbon_kg;
    this.capacity_ton = capacity_ton;
    this.reliability_score = reliability_score;
  }
}

class Route {
  constructor(segments, total_distance, total_duration, total_cost, total_carbon, reliability_score, route_id) {
    this.segments = segments;
    this.total_distance = total_distance;
    this.total_duration = total_duration;
    this.total_cost = total_cost;
    this.total_carbon = total_carbon;
    this.reliability_score = reliability_score;
    this.route_id = route_id;
  }
}

class CostBreakdown {
  constructor(transport, customs, insurance, storage, carbon_tax, handling, total) {
    this.transport = transport;
    this.customs = customs;
    this.insurance = insurance;
    this.storage = storage;
    this.carbon_tax = carbon_tax;
    this.handling = handling;
    this.total = total;
  }
}

class LogisticsCostEngine {
  constructor() {
    this.api_keys = {
      'google_maps': process.env.GOOGLE_MAPS_API_KEY || '',
      'openweather': process.env.OPENWEATHER_API_KEY || '',
      'freightos': process.env.FREIGHTOS_API_KEY || '',
      'carbon_interface': process.env.CARBON_INTERFACE_API_KEY || ''
    };
    
    // Cost parameters (EUR)
    this.cost_rates = {
      'truck': {
        'per_km': 0.15,
        'per_ton_km': 0.08,
        'base_cost': 50,
        'carbon_per_km': 0.15  // kg CO2 per km
      },
      'train': {
        'per_km': 0.08,
        'per_ton_km': 0.04,
        'base_cost': 200,
        'carbon_per_km': 0.04
      },
      'ship': {
        'per_km': 0.02,
        'per_ton_km': 0.01,
        'base_cost': 500,
        'carbon_per_km': 0.03
      },
      'air': {
        'per_km': 0.50,
        'per_ton_km': 0.25,
        'base_cost': 1000,
        'carbon_per_km': 0.50
      }
    };
    
    // Customs and insurance rates
    this.customs_rate = 0.05;  // 5% of cargo value
    this.insurance_rate = 0.02;  // 2% of cargo value
    this.storage_rate = 50;  // EUR per day per ton
    this.carbon_tax_rate = 50;  // EUR per ton CO2
  }
  
  calculateDistance(origin, destination) {
    /** Calculate distance between two locations using Haversine formula */
    const lat1 = this.toRadians(origin.latitude);
    const lon1 = this.toRadians(origin.longitude);
    const lat2 = this.toRadians(destination.latitude);
    const lon2 = this.toRadians(destination.longitude);
    
    const dlat = lat2 - lat1;
    const dlon = lon2 - lon1;
    
    const a = Math.sin(dlat/2)**2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon/2)**2;
    const c = 2 * Math.asin(Math.sqrt(a));
    
    return 6371 * c;  // Earth radius in km
  }
  
  toRadians(degrees) {
    return degrees * (Math.PI / 180);
  }
  
  getRoutePlanning(origin, destination, cargo_weight, cargo_value, urgency = 'normal') {
    /**
     * Generate multi-modal route options with cost and carbon analysis
     */
    const routes = [];
    
    // Calculate base distance
    const direct_distance = this.calculateDistance(origin, destination);
    
    // Generate different route options
    const route_options = this.generateRouteOptions(origin, destination, direct_distance);
    
    for (let i = 0; i < route_options.length; i++) {
      const route = this.calculateRouteDetails(route_options[i], cargo_weight, cargo_value, urgency);
      if (route) {
        routes.push(route);
      }
    }
    
    // Sort by total cost
    routes.sort((a, b) => a.total_cost - b.total_cost);
    
    return routes;
  }
  
  generateRouteOptions(origin, destination, direct_distance) {
    /** Generate different route combinations */
    const options = [];
    
    // Option 1: Direct truck
    if (direct_distance <= 800) {  // Suitable for truck
      options.push([{
        mode: TransportMode.TRUCK,
        origin: origin,
        destination: destination,
        distance: direct_distance
      }]);
    }
    
    // Option 2: Train (if distance > 300km)
    if (direct_distance > 300) {
      options.push([{
        mode: TransportMode.TRAIN,
        origin: origin,
        destination: destination,
        distance: direct_distance
      }]);
    }
    
    // Option 3: Ship (if international or coastal)
    if (this.isInternational(origin, destination) || this.isCoastal(origin) || this.isCoastal(destination)) {
      options.push([{
        mode: TransportMode.SHIP,
        origin: origin,
        destination: destination,
        distance: direct_distance * 1.2  // Sea routes are longer
      }]);
    }
    
    // Option 4: Air (for urgent shipments)
    options.push([{
      mode: TransportMode.AIR,
      origin: origin,
      destination: destination,
      distance: direct_distance
    }]);
    
    // Option 5: Multimodal (truck + train + truck)
    if (direct_distance > 500) {
      // Find intermediate points for multimodal
      const intermediate = this.findIntermediatePoint(origin, destination);
      if (intermediate) {
        options.push([
          {
            mode: TransportMode.TRUCK,
            origin: origin,
            destination: intermediate,
            distance: this.calculateDistance(origin, intermediate)
          },
          {
            mode: TransportMode.TRAIN,
            origin: intermediate,
            destination: intermediate,  // Same point for transfer
            distance: 0  // Transfer distance
          },
          {
            mode: TransportMode.TRUCK,
            origin: intermediate,
            destination: destination,
            distance: this.calculateDistance(intermediate, destination)
          }
        ]);
      }
    }
    
    return options;
  }
  
  calculateRouteDetails(route_option, cargo_weight, cargo_value, urgency) {
    /** Calculate detailed costs and carbon impact for a route */
    const segments = [];
    let total_distance = 0;
    let total_duration = 0;
    let total_cost = 0;
    let total_carbon = 0;
    let reliability_score = 1.0;
    
    for (let i = 0; i < route_option.length; i++) {
      const segment = route_option[i];
      
      // Calculate segment details
      const segment_details = this.calculateSegmentCost(
        segment.mode, segment.distance, cargo_weight, cargo_value
      );
      
      // Create route segment
      const route_segment = new RouteSegment(
        segment.mode,
        segment.origin,
        segment.destination,
        segment.distance,
        segment_details.duration,
        segment_details.cost,
        segment_details.carbon,
        segment_details.capacity,
        segment_details.reliability
      );
      
      segments.push(route_segment);
      total_distance += segment.distance;
      total_duration += segment_details.duration;
      total_cost += segment_details.cost;
      total_carbon += segment_details.carbon;
      reliability_score *= segment_details.reliability;
    }
    
    // Add additional costs
    const additional_costs = this.calculateAdditionalCosts(cargo_weight, cargo_value, total_distance);
    total_cost += additional_costs.total;
    
    // Apply urgency multiplier
    if (urgency === 'urgent') {
      total_cost *= 1.5;
      total_duration *= 0.7;
    } else if (urgency === 'express') {
      total_cost *= 2.0;
      total_duration *= 0.5;
    }
    
    return new Route(
      segments,
      total_distance,
      total_duration,
      total_cost,
      total_carbon,
      reliability_score,
      `route_${new Date().toISOString().replace(/[:.]/g, '')}_${segments.length}`
    );
  }
  
  calculateSegmentCost(mode, distance, cargo_weight, cargo_value) {
    /** Calculate cost breakdown for a single segment */
    const rates = this.cost_rates[mode];
    
    // Base transport cost
    const transport_cost = rates.base_cost + (distance * rates.per_km) + (distance * cargo_weight * rates.per_ton_km);
    
    // Carbon impact
    const carbon_kg = distance * rates.carbon_per_km * cargo_weight;
    
    // Duration estimation
    const speeds = {
      'truck': 60,
      'train': 80,
      'ship': 25,
      'air': 800
    };
    const duration = distance / speeds[mode];
    
    // Capacity and reliability
    const capacity = {
      'truck': 25,
      'train': 1000,
      'ship': 5000,
      'air': 100
    };
    
    const reliability = {
      'truck': 0.95,
      'train': 0.98,
      'ship': 0.90,
      'air': 0.99
    };
    
    return {
      cost: transport_cost,
      carbon: carbon_kg,
      duration: duration,
      capacity: capacity[mode],
      reliability: reliability[mode]
    };
  }
  
  calculateAdditionalCosts(cargo_weight, cargo_value, distance) {
    /** Calculate additional costs (customs, insurance, storage, etc.) */
    const customs = cargo_value * this.customs_rate;
    const insurance = cargo_value * this.insurance_rate;
    const storage = cargo_weight * this.storage_rate * (distance / 500);  // Days based on distance
    const carbon_tax = 0;  // Will be calculated based on actual carbon
    const handling = cargo_weight * 10;  // EUR per ton
    
    const total = customs + insurance + storage + carbon_tax + handling;
    
    return new CostBreakdown(
      0,  // transport already calculated
      customs,
      insurance,
      storage,
      carbon_tax,
      handling,
      total
    );
  }
  
  isInternational(origin, destination) {
    /** Check if route is international */
    return origin.country !== destination.country;
  }
  
  isCoastal(location) {
    /** Check if location is coastal (simplified) */
    const coastal_cities = ['rotterdam', 'hamburg', 'antwerp', 'barcelona', 'genoa'];
    return coastal_cities.includes(location.city.toLowerCase());
  }
  
  findIntermediatePoint(origin, destination) {
    /** Find intermediate point for multimodal routes */
    // Major logistics hubs
    const hubs = [
      new Location("Rotterdam", 51.9225, 4.4792, "Netherlands", "Rotterdam"),
      new Location("Hamburg", 53.5511, 9.9937, "Germany", "Hamburg"),
      new Location("Antwerp", 51.2194, 4.4025, "Belgium", "Antwerp"),
      new Location("Barcelona", 41.3851, 2.1734, "Spain", "Barcelona"),
      new Location("Milan", 45.4642, 9.1900, "Italy", "Milan")
    ];
    
    // Find closest hub to midpoint
    const mid_lat = (origin.latitude + destination.latitude) / 2;
    const mid_lon = (origin.longitude + destination.longitude) / 2;
    
    let closest_hub = null;
    let min_distance = Infinity;
    
    for (const hub of hubs) {
      const distance = this.calculateDistance(
        new Location("mid", mid_lat, mid_lon, "", ""),
        hub
      );
      if (distance < min_distance) {
        min_distance = distance;
        closest_hub = hub;
      }
    }
    
    return closest_hub;
  }
  
  getRealTimeRates(origin, destination, cargo_weight, mode) {
    /** Get real-time rates from external APIs (placeholder) */
    // This would integrate with real logistics APIs
    // For now, return estimated rates
    const base_rate = this.cost_rates[mode].per_km;
    
    // Simulate real-time pricing variations
    const variation = 0.8 + Math.random() * 0.4;  // 0.8 to 1.2
    
    return {
      rate_per_km: base_rate * variation,
      availability: true,
      transit_time_days: this.estimateTransitTime(mode, origin, destination),
      last_updated: new Date().toISOString()
    };
  }
  
  estimateTransitTime(mode, origin, destination) {
    /** Estimate transit time in days */
    const distance = this.calculateDistance(origin, destination);
    
    const speeds = {
      'truck': 60,  // km/h
      'train': 80,
      'ship': 25,
      'air': 800
    };
    
    const hours = distance / speeds[mode];
    const days = Math.ceil(hours / 24);
    
    // Add buffer days
    const buffer_days = {
      'truck': 1,
      'train': 2,
      'ship': 3,
      'air': 1
    };
    
    return days + buffer_days[mode];
  }
  
  calculateCarbonImpact(route, cargo_weight) {
    /** Calculate detailed carbon impact */
    const total_carbon = route.total_carbon;
    
    // Carbon tax calculation
    const carbon_tax = total_carbon * (this.carbon_tax_rate / 1000);  // Convert to EUR
    
    // Carbon intensity per ton-km
    const carbon_intensity = total_carbon / (cargo_weight * route.total_distance);
    
    return {
      total_carbon_kg: total_carbon,
      carbon_tax_eur: carbon_tax,
      carbon_intensity_kg_per_ton_km: carbon_intensity,
      carbon_savings_vs_truck: this.calculateCarbonSavings(route),
      carbon_equivalent: {
        trees_needed: total_carbon / 22,  // kg CO2 per tree per year
        car_km_equivalent: total_carbon / 0.2,  // kg CO2 per km
        flight_km_equivalent: total_carbon / 0.25  // kg CO2 per km
      }
    };
  }
  
  calculateCarbonSavings(route) {
    /** Calculate carbon savings compared to truck-only route */
    const truck_carbon = route.total_distance * this.cost_rates.truck.carbon_per_km;
    return truck_carbon - route.total_carbon;
  }
  
  optimizeRoute(routes, optimization_criteria = 'cost') {
    /** Optimize route based on different criteria */
    if (optimization_criteria === 'cost') {
      return routes.reduce((min, curr) => curr.total_cost < min.total_cost ? curr : min);
    } else if (optimization_criteria === 'carbon') {
      return routes.reduce((min, curr) => curr.total_carbon < min.total_carbon ? curr : min);
    } else if (optimization_criteria === 'time') {
      return routes.reduce((min, curr) => curr.total_duration < min.total_duration ? curr : min);
    } else if (optimization_criteria === 'reliability') {
      return routes.reduce((max, curr) => curr.reliability_score > max.reliability_score ? curr : max);
    } else {
      // Multi-criteria optimization
      return this.multiCriteriaOptimization(routes);
    }
  }
  
  multiCriteriaOptimization(routes) {
    /** Multi-criteria optimization using weighted scoring */
    let best_route = null;
    let best_score = -1;
    
    for (const route of routes) {
      // Normalize scores (0-1)
      const max_cost = Math.max(...routes.map(r => r.total_cost));
      const max_carbon = Math.max(...routes.map(r => r.total_carbon));
      const max_duration = Math.max(...routes.map(r => r.total_duration));
      
      const cost_score = 1 - (route.total_cost / max_cost);
      const carbon_score = 1 - (route.total_carbon / max_carbon);
      const time_score = 1 - (route.total_duration / max_duration);
      const reliability_score = route.reliability_score;
      
      // Weighted combination
      const total_score = (
        cost_score * 0.4 +
        carbon_score * 0.3 +
        time_score * 0.2 +
        reliability_score * 0.1
      );
      
      if (total_score > best_score) {
        best_score = total_score;
        best_route = route;
      }
    }
    
    return best_route;
  }
  
  generateRouteReport(route, cargo_weight, cargo_value) {
    /** Generate comprehensive route report */
    const carbon_impact = this.calculateCarbonImpact(route, cargo_weight);
    
    return {
      route_id: route.route_id,
      summary: {
        total_distance_km: route.total_distance,
        total_duration_hours: route.total_duration,
        total_cost_eur: route.total_cost,
        total_carbon_kg: route.total_carbon,
        reliability_score: route.reliability_score
      },
      segments: route.segments.map(segment => ({
        mode: segment.mode,
        origin: segment.origin.name,
        destination: segment.destination.name,
        distance_km: segment.distance_km,
        duration_hours: segment.duration_hours,
        cost_eur: segment.cost_euro,
        carbon_kg: segment.carbon_kg
      })),
      carbon_impact: carbon_impact,
      cost_breakdown: {
        transport: route.segments.reduce((sum, s) => sum + s.cost_euro, 0),
        carbon_tax: carbon_impact.carbon_tax_eur,
        total: route.total_cost
      },
      recommendations: this.generateRecommendations(route, carbon_impact),
      generated_at: new Date().toISOString()
    };
  }
  
  generateRecommendations(route, carbon_impact) {
    /** Generate recommendations for route optimization */
    const recommendations = [];
    
    if (carbon_impact.carbon_savings_vs_truck > 0) {
      recommendations.push(
        `Carbon savings: ${carbon_impact.carbon_savings_vs_truck.toFixed(1)} kg CO2 vs truck-only route`
      );
    }
    
    if (route.reliability_score < 0.95) {
      recommendations.push("Consider alternative routes for better reliability");
    }
    
    if (route.total_duration > 48) {
      recommendations.push("Route may be suitable for non-urgent shipments");
    }
    
    if (carbon_impact.carbon_intensity_kg_per_ton_km > 0.1) {
      recommendations.push("Consider more carbon-efficient transport modes");
    }
    
    return recommendations;
  }
}

module.exports = {
  LogisticsCostEngine,
  Location,
  RouteSegment,
  Route,
  CostBreakdown,
  TransportMode
}; 
