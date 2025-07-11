const axios = require('axios');
const { supabase } = require('../supabase');

class MaterialsService {
  constructor() {
    this.apiKey = process.env.NEXT_GEN_MATERIALS_API_KEY;
    this.baseUrl = 'https://api.next-gen-materials.com/v1'; // Replace with actual API endpoint
  }

  /**
   * Get comprehensive material data from Next-Gen Materials API
   */
  async getMaterialData(materialName) {
    try {
      const response = await axios.get(`${this.baseUrl}/materials/search`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        params: {
          query: materialName,
          include_properties: true,
          include_sustainability: true,
          include_alternatives: true
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error fetching material data:', error);
      throw error;
    }
  }

  /**
   * Get material properties and sustainability metrics
   */
  async getMaterialProperties(materialId) {
    try {
      const response = await axios.get(`${this.baseUrl}/materials/${materialId}/properties`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error fetching material properties:', error);
      throw error;
    }
  }

  /**
   * Find sustainable alternatives for a material
   */
  async findSustainableAlternatives(materialId) {
    try {
      const response = await axios.get(`${this.baseUrl}/materials/${materialId}/alternatives`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        params: {
          sustainability_focus: true,
          include_comparison: true
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error finding alternatives:', error);
      throw error;
    }
  }

  /**
   * Calculate environmental impact for material exchanges
   */
  async calculateEnvironmentalImpact(materialId, quantity, unit = 'kg') {
    try {
      const response = await axios.post(`${this.baseUrl}/impact/calculate`, {
        material_id: materialId,
        quantity: quantity,
        unit: unit
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error calculating environmental impact:', error);
      throw error;
    }
  }

  /**
   * Get circular economy opportunities for materials
   */
  async getCircularEconomyOpportunities(materialId) {
    try {
      const response = await axios.get(`${this.baseUrl}/materials/${materialId}/circular-opportunities`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error fetching circular economy opportunities:', error);
      throw error;
    }
  }

  /**
   * Enhanced material matching using scientific data
   */
  async findScientificMatches(companyProfile, wasteStreams, resourceNeeds) {
    try {
      const response = await axios.post(`${this.baseUrl}/matching/scientific`, {
        company_profile: companyProfile,
        waste_streams: wasteStreams,
        resource_needs: resourceNeeds,
        include_sustainability: true,
        include_circular_economy: true
      }, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error finding scientific matches:', error);
      throw error;
    }
  }

  /**
   * Get supply chain insights for materials
   */
  async getSupplyChainInsights(materialId, location) {
    try {
      const response = await axios.get(`${this.baseUrl}/materials/${materialId}/supply-chain`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json'
        },
        params: {
          location: location,
          include_availability: true,
          include_cost_trends: true
        }
      });

      return response.data;
    } catch (error) {
      console.error('Error fetching supply chain insights:', error);
      throw error;
    }
  }

  /**
   * Enhanced AI portfolio generation with scientific data
   */
  async generateScientificPortfolio(companyData) {
    try {
      // Get material data for company's waste streams and needs
      const materialPromises = [];
      
      if (companyData.waste_streams) {
        companyData.waste_streams.forEach(waste => {
          materialPromises.push(this.getMaterialData(waste.name));
        });
      }

      if (companyData.resource_needs) {
        companyData.resource_needs.forEach(need => {
          materialPromises.push(this.getMaterialData(need.name));
        });
      }

      const materialData = await Promise.all(materialPromises);

      // Calculate environmental impact
      const impactPromises = materialData.map(material => 
        this.calculateEnvironmentalImpact(material.id, 1000, 'kg')
      );
      const impactData = await Promise.all(impactPromises);

      // Find circular economy opportunities
      const circularPromises = materialData.map(material =>
        this.getCircularEconomyOpportunities(material.id)
      );
      const circularData = await Promise.all(circularPromises);

      // Generate comprehensive portfolio
      const portfolio = {
        company: companyData,
        materials: materialData,
        environmental_impact: impactData,
        circular_opportunities: circularData,
        sustainability_score: this.calculateSustainabilityScore(materialData),
        symbiosis_potential: this.calculateSymbiosisPotential(materialData, circularData),
        recommendations: this.generateRecommendations(materialData, impactData, circularData)
      };

      return portfolio;
    } catch (error) {
      console.error('Error generating scientific portfolio:', error);
      throw error;
    }
  }

  /**
   * Calculate sustainability score based on material properties
   */
  calculateSustainabilityScore(materials) {
    if (!materials || materials.length === 0) return 0;

    const scores = materials.map(material => {
      let score = 0;
      
      // Recyclability score
      if (material.recyclability) {
        score += material.recyclability * 25;
      }
      
      // Carbon footprint score
      if (material.carbon_footprint) {
        const carbonScore = Math.max(0, 100 - (material.carbon_footprint * 10));
        score += carbonScore * 0.25;
      }
      
      // Renewable content score
      if (material.renewable_content) {
        score += material.renewable_content * 25;
      }
      
      // Biodegradability score
      if (material.biodegradability) {
        score += material.biodegradability * 25;
      }

      return Math.min(100, score);
    });

    return scores.reduce((sum, score) => sum + score, 0) / scores.length;
  }

  /**
   * Calculate symbiosis potential based on material compatibility
   */
  calculateSymbiosisPotential(materials, circularData) {
    if (!materials || materials.length === 0) return 0;

    let potential = 0;
    let count = 0;

    materials.forEach((material, index) => {
      if (circularData[index] && circularData[index].opportunities) {
        potential += circularData[index].opportunities.length * 10;
        count++;
      }
    });

    return count > 0 ? Math.min(100, potential / count) : 0;
  }

  /**
   * Generate recommendations based on scientific data
   */
  generateRecommendations(materials, impactData, circularData) {
    const recommendations = [];

    // High impact materials recommendations
    impactData.forEach((impact, index) => {
      if (impact.carbon_savings > 1000) {
        recommendations.push({
          type: 'high_impact',
          title: `High Carbon Reduction Opportunity`,
          description: `${materials[index].name} can reduce carbon emissions by ${impact.carbon_savings} kg CO2e per ton`,
          priority: 'high',
          potential_savings: impact.carbon_savings * 0.05 // $50 per ton CO2
        });
      }
    });

    // Circular economy recommendations
    circularData.forEach((circular, index) => {
      if (circular.opportunities && circular.opportunities.length > 0) {
        recommendations.push({
          type: 'circular_economy',
          title: `Circular Economy Opportunity`,
          description: `${materials[index].name} has ${circular.opportunities.length} circular economy applications`,
          priority: 'medium',
          opportunities: circular.opportunities
        });
      }
    });

    return recommendations;
  }

  /**
   * Save enhanced material data to database
   */
  async saveEnhancedMaterialData(materialId, scientificData) {
    try {
      const { error } = await supabase
        .from('materials')
        .update({
          scientific_properties: scientificData.properties,
          sustainability_metrics: scientificData.sustainability,
          environmental_impact: scientificData.impact,
          circular_opportunities: scientificData.circular_opportunities,
          updated_at: new Date().toISOString()
        })
        .eq('id', materialId);

      if (error) throw error;
      return true;
    } catch (error) {
      console.error('Error saving enhanced material data:', error);
      throw error;
    }
  }
}

module.exports = new MaterialsService(); 