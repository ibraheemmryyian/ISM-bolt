// AI service implementation without React dependencies
import { supabase } from '../lib/supabase';

interface AIBackendInterface {
  generateRecommendations: (userId: string, userProfile: any) => Promise<any[]>;
  findMaterialMatches: (materialId: string) => Promise<any[]>;
  calculateCompatibilityScore: (company1: any, company2: any) => Promise<number>;
  predictDemand: (materialType: string, location: string) => Promise<any>;
  optimizeLogistics: (materials: any[], location: string) => Promise<any>;
}

// This would be your AI backend interface
export class AIService implements AIBackendInterface {
  private apiEndpoint: string;

  constructor() {
    // In production, this would be your AI service endpoint
    this.apiEndpoint = '/api/ai'; // This would be your actual AI backend
  }

  async generateRecommendations(userId: string, userProfile: any): Promise<any[]> {
    try {
      // Call revolutionary AI backend for recommendations
      const response = await fetch('/api/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          buyer: {
            id: userId,
            industry: userProfile.industry,
            waste_type: userProfile.materials_of_interest,
            quantity: 1000, // Default quantity
            location: userProfile.location
          },
          seller: {
            // In MVP, use mock seller data from UAE seed
            material_needed: userProfile.materials_of_interest,
            capabilities: userProfile.capabilities || [],
            location: 'UAE'
          }
        })
      });

      if (!response.ok) throw new Error('AI matching failed');
      
      const matchResult = await response.json();
      
      // Format as recommendation
      const recommendation = {
        id: `rec_${Date.now()}`,
        type: 'match',
        title: 'Industrial Symbiosis Match',
        description: `Found ${matchResult.match_quality} match for your materials`,
        confidence: matchResult.revolutionary_score * 100,
        action: 'View Match',
        metadata: {
          ...matchResult,
          trust_status: matchResult.trust_status
        }
      };

      // Store recommendation in database
      const { error } = await supabase
        .from('ai_recommendations')
        .insert({
          company_id: userId,
          type: recommendation.type,
          title: recommendation.title,
          description: recommendation.description,
          confidence: recommendation.confidence,
          action_url: `/match/${recommendation.id}`,
          status: 'pending'
        });

      if (error) throw error;

      return [recommendation];
    } catch (error) {
      console.error('Error generating AI recommendations:', error);
      return [];
    }
  }

  async findMaterialMatches(materialId: string): Promise<any[]> {
    try {
      // Get the material details
      const { data: material, error } = await supabase
        .from('materials')
        .select('*')
        .eq('id', materialId)
        .maybeSingle();

      if (error) throw error;

      // Call revolutionary AI backend for matches
      const response = await fetch('/api/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          buyer: {
            id: 'buyer_mock',
            industry: 'Manufacturing',
            waste_type: material.material_name,
            quantity: material.quantity,
            location: material.location
          },
          seller: {
            id: material.company_id,
            material_needed: material.material_name,
            capabilities: [],
            location: material.location
          }
        })
      });

      if (!response.ok) throw new Error('AI matching failed');
      
      const matchResult = await response.json();

      // For MVP, return the match result as a single match
      return [{
        ...material,
        match_score: matchResult.revolutionary_score,
        sustainability_score: matchResult.sustainability_score,
        trust_status: matchResult.trust_status
      }];
    } catch (error) {
      console.error('Error finding material matches:', error);
      return [];
    }
  }

  async calculateCompatibilityScore(company1: any, company2: any): Promise<number> {
    // Simplified compatibility calculation
    let score = 50; // Base score

    // Location proximity
    if (company1.location && company2.location) {
      score += 20;
    }

    // Material interest overlap
    if (company1.materials_of_interest && company2.materials_of_interest) {
      const interests1 = company1.materials_of_interest.toLowerCase().split(',');
      const interests2 = company2.materials_of_interest.toLowerCase().split(',');
      const overlap = interests1.some((interest: string) => 
        interests2.some((interest2: string) => interest2.includes(interest.trim()))
      );
      if (overlap) score += 25;
    }

    // Organization type compatibility
    type OrganizationType = 'manufacturing' | 'research' | 'recycling' | 'waste_management' | 'consulting' | 'investment';
    const compatibleTypes: Record<OrganizationType, OrganizationType[]> = {
      'manufacturing': ['recycling', 'waste_management'],
      'research': ['manufacturing', 'consulting'],
      'recycling': ['manufacturing', 'waste_management'],
      'waste_management': ['recycling', 'manufacturing'],
      'consulting': ['manufacturing', 'research'],
      'investment': ['manufacturing', 'research', 'recycling']
    };

    if (company1.organization_type && company2.organization_type) {
      const orgType1 = company1.organization_type as OrganizationType;
      const orgType2 = company2.organization_type as OrganizationType;
      
      if (compatibleTypes[orgType1]?.includes(orgType2)) {
        score += 15;
      }
    }

    return Math.min(score, 100);
  }

  async predictDemand(materialType: string, location: string): Promise<any> {
    // Mock demand prediction
    return {
      material_type: materialType,
      location: location,
      predicted_demand: Math.floor(Math.random() * 1000) + 100,
      confidence: Math.floor(Math.random() * 30) + 70,
      trend: Math.random() > 0.5 ? 'increasing' : 'stable',
      factors: ['seasonal_variation', 'local_industry', 'regulations']
    };
  }

  async optimizeLogistics(materials: any[], location: string): Promise<any> {
    // Mock logistics optimization
    return {
      optimized_routes: materials.length,
      estimated_cost_savings: Math.floor(Math.random() * 5000) + 1000,
      carbon_reduction: Math.floor(Math.random() * 500) + 100,
      recommendations: [
        'Consolidate shipments from nearby locations',
        'Use electric vehicles for short distances',
        'Schedule pickups during off-peak hours'
      ]
    };
  }

  private calculateSimpleMatchScore(material1: any, material2: any): number {
    let score = 60; // Base score

    // Material name similarity (simple keyword matching)
    const name1 = material1.material_name.toLowerCase();
    const name2 = material2.material_name.toLowerCase();
    
    if (name1.includes(name2) || name2.includes(name1)) {
      score += 30;
    }

    // Quantity compatibility
    const ratio = Math.min(material1.quantity, material2.quantity) / 
                  Math.max(material1.quantity, material2.quantity);
    score += ratio * 10;

    return Math.min(score, 100);
  }

  private getCompatibilityFactors(material1: any, material2: any): string[] {
    const factors = [];
    
    if (material1.material_name.toLowerCase().includes(material2.material_name.toLowerCase())) {
      factors.push('material_type_match');
    }
    
    if (material1.unit === material2.unit) {
      factors.push('unit_compatibility');
    }
    
    factors.push('location_proximity');
    factors.push('company_compatibility');
    
    return factors;
  }
}

// Export singleton instance
export const aiService = new AIService();
