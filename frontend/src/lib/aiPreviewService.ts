import { supabase } from './supabase';

export interface AIPreviewData {
  company_profile: {
    id: string;
    name: string;
    industry: string;
    location: string;
    employee_count: number;
    products: string;
    main_materials: string;
    production_volume: string;
    process_description: string;
  };
  materials: Array<{
    id?: string;
    name: string;
    description: string;
    category: string;
    quantity: string;
    frequency: string;
    notes: string;
    potential_value?: string;
    quality_grade?: string;
    potential_uses?: string[];
    symbiosis_opportunities?: string[];
    ai_generated: boolean;
  }>;
  requirements: Array<{
    id?: string;
    name: string;
    description: string;
    category: string;
    quantity: string;
    frequency: string;
    notes: string;
    current_cost?: string;
    priority?: string;
    potential_sources?: string[];
    symbiosis_opportunities?: string[];
    ai_generated: boolean;
  }>;
  potential_matches: Array<{
    id?: string;
    company_id: string;
    partner_company_id: string;
    company_name: string;
    partner_company_name: string;
    industry: string;
    match_reason: string;
    match_score: number;
    materials_involved: string[];
    potential_savings?: number;
    carbon_reduction?: number;
    status: 'pending' | 'accepted' | 'rejected';
    ai_generated: boolean;
  }>;
  ai_insights: {
    symbiosis_score: string;
    estimated_savings: string;
    carbon_reduction: string;
    top_opportunities: string[];
    recommended_partners: string[];
    implementation_roadmap: string[];
  };
  generation_status: 'generating' | 'completed' | 'error';
  generation_progress: number;
}

class AIPreviewService {
  private baseUrl = import.meta.env.VITE_AI_PREVIEW_URL;

  /**
   * Generate AI preview data for a company
   */
  async generateAIPreview(companyProfile: any): Promise<AIPreviewData> {
    try {
      console.log('Calling AI pipeline with company profile:', companyProfile);
      
      const response = await fetch(`${this.baseUrl}/ai-pipeline`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          company_name: companyProfile.name,
          industry: companyProfile.industry,
          location: companyProfile.location,
          employee_count: companyProfile.employee_count,
          products: companyProfile.products,
          main_materials: companyProfile.main_materials,
          production_volume: companyProfile.production_volume,
          process_description: companyProfile.process_description
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }

      const result = await response.json();
      console.log('AI pipeline response:', result);
      
      // Transform the backend response to match our frontend interface
      return this.transformBackendResponse(result);
    } catch (error) {
      console.error('Error generating AI preview:', error);
      throw error;
    }
  }

  /**
   * Get existing AI preview data for a company
   */
  async getAIPreview(companyId: string): Promise<AIPreviewData | null> {
    try {
      // First, get the company profile
      const { data: company, error: companyError } = await supabase
        .from('companies')
        .select('*')
        .eq('id', companyId)
        .single();

      if (companyError || !company) {
        throw new Error('Company not found');
      }

      // Get materials from database
      const { data: materials, error: materialsError } = await supabase
        .from('materials')
        .select('*')
        .eq('company_id', companyId)
        .eq('ai_generated', true);

      if (materialsError) {
        console.error('Error fetching materials:', materialsError);
      }

      // Get requirements from database
      const { data: requirements, error: requirementsError } = await supabase
        .from('requirements')
        .select('*')
        .eq('company_id', companyId)
        .eq('ai_generated', true);

      if (requirementsError) {
        console.error('Error fetching requirements:', requirementsError);
      }

      // Get matches from database
      const { data: matches, error: matchesError } = await supabase
        .from('matches')
        .select('*')
        .eq('company_id', companyId)
        .eq('ai_generated', true);

      if (matchesError) {
        console.error('Error fetching matches:', matchesError);
      }

      // Get AI insights from database
      const { data: insights, error: insightsError } = await supabase
        .from('ai_insights')
        .select('*')
        .eq('company_id', companyId)
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (insightsError) {
        console.error('Error fetching AI insights:', insightsError);
      }

      // Transform database data to match our interface
      return this.transformDatabaseData(company, materials || [], requirements || [], matches || [], insights);
    } catch (error) {
      console.error('Error getting AI preview:', error);
      return null;
    }
  }

  /**
   * Save AI-generated content to database
   */
  async saveAIContent(companyId: string, previewData: AIPreviewData): Promise<boolean> {
    try {
      // Save materials
      if (previewData.materials.length > 0) {
        const { error: materialsError } = await supabase
          .from('materials')
          .upsert(
            previewData.materials.map(material => ({
              ...material,
              company_id: companyId,
              ai_generated: true
            }))
          );

        if (materialsError) {
          console.error('Error saving materials:', materialsError);
        }
      }

      // Save requirements
      if (previewData.requirements.length > 0) {
        const { error: requirementsError } = await supabase
          .from('requirements')
          .upsert(
            previewData.requirements.map(requirement => ({
              ...requirement,
              company_id: companyId,
              ai_generated: true
            }))
          );

        if (requirementsError) {
          console.error('Error saving requirements:', requirementsError);
        }
      }

      // Save matches
      if (previewData.potential_matches.length > 0) {
        const { error: matchesError } = await supabase
          .from('matches')
          .upsert(
            previewData.potential_matches.map(match => ({
              ...match,
              company_id: companyId,
              ai_generated: true
            }))
          );

        if (matchesError) {
          console.error('Error saving matches:', matchesError);
        }
      }

      // Save AI insights
      if (previewData.ai_insights) {
        const { error: insightsError } = await supabase
          .from('ai_insights')
          .insert({
            company_id: companyId,
            insight_type: 'opportunity',
            title: 'AI-Generated Symbiosis Insights',
            description: `Your company has a symbiosis score of ${previewData.ai_insights.symbiosis_score} with potential savings of ${previewData.ai_insights.estimated_savings} and carbon reduction of ${previewData.ai_insights.carbon_reduction}.`,
            confidence: 90,
            impact: 'high',
            action_required: true,
            metadata: {
              symbiosis_score: previewData.ai_insights.symbiosis_score,
              estimated_savings: previewData.ai_insights.estimated_savings,
              carbon_reduction: previewData.ai_insights.carbon_reduction,
              top_opportunities: previewData.ai_insights.top_opportunities,
              recommended_partners: previewData.ai_insights.recommended_partners,
              implementation_roadmap: previewData.ai_insights.implementation_roadmap
            }
          });

        if (insightsError) {
          console.error('Error saving AI insights:', insightsError);
        }
      }

      return true;
    } catch (error) {
      console.error('Error saving AI content:', error);
      return false;
    }
  }

  /**
   * Regenerate AI content
   */
  async regenerateAIContent(companyId: string): Promise<AIPreviewData> {
    try {
      // Get company profile
      const { data: company, error: companyError } = await supabase
        .from('companies')
        .select('*')
        .eq('id', companyId)
        .single();

      if (companyError || !company) {
        throw new Error('Company not found');
      }

      // Call the AI pipeline to regenerate
      return await this.generateAIPreview(company);
    } catch (error) {
      console.error('Error regenerating AI content:', error);
      throw error;
    }
  }

  /**
   * Transform backend response to frontend interface
   */
  private transformBackendResponse(result: any): AIPreviewData {
    console.log('Transforming backend response:', result);
    
    // Extract materials from the portfolio
    const materials = result.portfolio?.materials || [];
    const requirements = result.portfolio?.requirements || [];
    const matches = result.total_matches || [];
    
    // Calculate insights from the data
    const totalSavings = matches.reduce((sum: number, match: any) => sum + (match.potential_savings || 0), 0);
    const totalCarbonReduction = matches.reduce((sum: number, match: any) => sum + (match.carbon_reduction || 0), 0);
    
    // Generate top opportunities from materials and matches
    const topOpportunities = materials
      .slice(0, 4)
      .map((material: any) => `${material.name} to potential partners`);
    
    // Generate recommended partners from matches
    const recommendedPartners = matches
      .slice(0, 4)
      .map((match: any) => match.partner_company_name || 'Potential Partner');
    
    // Generate implementation roadmap
    const implementationRoadmap = [
      "Review and approve AI-generated materials",
      "Select preferred partner matches",
      "Contact potential partners",
      "Establish supply agreements"
    ];
    
    return {
      company_profile: {
        id: result.company_profile?.id || "unknown",
        name: result.company_profile?.name || "Unknown Company",
        industry: result.company_profile?.industry || "Unknown",
        location: result.company_profile?.location || "Unknown",
        employee_count: result.company_profile?.employee_count || 0,
        products: result.company_profile?.products || "Unknown",
        main_materials: result.company_profile?.main_materials || "Unknown",
        production_volume: result.company_profile?.production_volume || "Unknown",
        process_description: result.company_profile?.process_description || "Unknown"
      },
      materials: materials.map((material: any) => ({
        id: material.id,
        name: material.name,
        description: material.description,
        category: material.category,
        quantity: material.quantity,
        frequency: material.frequency,
        notes: material.notes,
        potential_value: material.potential_value,
        quality_grade: material.quality_grade,
        potential_uses: material.potential_uses || [],
        symbiosis_opportunities: material.symbiosis_opportunities || [],
        ai_generated: true
      })),
      requirements: requirements.map((requirement: any) => ({
        id: requirement.id,
        name: requirement.name,
        description: requirement.description,
        category: requirement.category,
        quantity: requirement.quantity,
        frequency: requirement.frequency,
        notes: requirement.notes,
        current_cost: requirement.current_cost,
        priority: requirement.priority,
        potential_sources: requirement.potential_sources || [],
        symbiosis_opportunities: requirement.symbiosis_opportunities || [],
        ai_generated: true
      })),
      potential_matches: matches.map((match: any) => ({
        id: match.id,
        company_id: match.company_id,
        partner_company_id: match.partner_company_id,
        company_name: match.company_name || "Your Company",
        partner_company_name: match.partner_company_name || "Potential Partner",
        industry: match.industry || "Unknown",
        match_reason: match.match_reason,
        match_score: match.match_score,
        materials_involved: match.materials_involved || [],
        potential_savings: match.potential_savings,
        carbon_reduction: match.carbon_reduction,
        status: match.status || 'pending',
        ai_generated: true
      })),
      ai_insights: {
        symbiosis_score: `${Math.round((matches.length / 10) * 100)}%`,
        estimated_savings: `$${(totalSavings * 12).toLocaleString()}/year`,
        carbon_reduction: `${totalCarbonReduction} tons CO2/year`,
        top_opportunities: topOpportunities,
        recommended_partners: recommendedPartners,
        implementation_roadmap: implementationRoadmap
      },
      generation_status: 'completed',
      generation_progress: 100
    };
  }

  /**
   * Transform database data to frontend interface
   */
  private transformDatabaseData(
    company: any,
    materials: any[],
    requirements: any[],
    matches: any[],
    insights: any
  ): AIPreviewData {
    return {
      company_profile: {
        id: company.id,
        name: company.name,
        industry: company.industry,
        location: company.location,
        employee_count: company.employee_count,
        products: company.products,
        main_materials: company.main_materials,
        production_volume: company.production_volume,
        process_description: company.process_description
      },
      materials: materials.map(material => ({
        id: material.id,
        name: material.name,
        description: material.description,
        category: material.category,
        quantity: material.quantity,
        frequency: material.frequency,
        notes: material.notes,
        potential_value: material.potential_value,
        quality_grade: material.quality_grade,
        potential_uses: material.potential_uses,
        symbiosis_opportunities: material.symbiosis_opportunities,
        ai_generated: material.ai_generated
      })),
      requirements: requirements.map(requirement => ({
        id: requirement.id,
        name: requirement.name,
        description: requirement.description,
        category: requirement.category,
        quantity: requirement.quantity,
        frequency: requirement.frequency,
        notes: requirement.notes,
        current_cost: requirement.current_cost,
        priority: requirement.priority,
        potential_sources: requirement.potential_sources,
        symbiosis_opportunities: requirement.symbiosis_opportunities,
        ai_generated: requirement.ai_generated
      })),
      potential_matches: matches.map(match => ({
        id: match.id,
        company_id: match.company_id,
        partner_company_id: match.partner_company_id,
        company_name: "Your Company", // This would need to be fetched
        partner_company_name: "Partner Company", // This would need to be fetched
        industry: "Unknown", // This would need to be fetched
        match_reason: match.match_reason,
        match_score: match.match_score,
        materials_involved: match.materials_involved,
        potential_savings: match.potential_savings,
        carbon_reduction: match.carbon_reduction,
        status: match.status,
        ai_generated: match.ai_generated
      })),
      ai_insights: insights ? {
        symbiosis_score: insights.metadata?.symbiosis_score || "0%",
        estimated_savings: insights.metadata?.estimated_savings || "$0/year",
        carbon_reduction: insights.metadata?.carbon_reduction || "0 tons CO2/year",
        top_opportunities: insights.metadata?.top_opportunities || [],
        recommended_partners: insights.metadata?.recommended_partners || [],
        implementation_roadmap: insights.metadata?.implementation_roadmap || []
      } : {
        symbiosis_score: "0%",
        estimated_savings: "$0/year",
        carbon_reduction: "0 tons CO2/year",
        top_opportunities: [],
        recommended_partners: [],
        implementation_roadmap: []
      },
      generation_status: 'completed',
      generation_progress: 100
    };
  }
}

export const aiPreviewService = new AIPreviewService(); 