import { supabase } from './supabase';

export interface CompanyProfile {
  id?: string;
  name: string;
  industry: string;
  location: string;
  employee_count: number;
  products: string;
  main_materials: string;
  production_volume: string;
  process_description: string;
  sustainability_goals: string[];
  current_waste_management: string;
  onboarding_completed: boolean;
}

export interface PortfolioData {
  achievements: {
    total_savings: number;
    carbon_reduction: number;
    partnerships: number;
    waste_diverted: number;
    matches: number;
    sustainability_score: number;
    efficiency_improvement: number;
  };
  recommendations: any[];
  recent_activity: any[];
  next_milestones: string[];
  industry_rank: number;
}

export class MigrationService {
  static async migrateCompanyData(): Promise<CompanyProfile | null> {
    try {
      console.log('Starting company data migration...');
      
      // Get localStorage data
      const companyProfile = JSON.parse(localStorage.getItem('symbioflows-company-profile') || '{}');
      const portfolio = JSON.parse(localStorage.getItem('symbioflows-portfolio') || '{}');
      
      if (!companyProfile.name) {
        console.log('No company profile found in localStorage');
        return null;
      }

      console.log('Found company profile:', companyProfile.name);

      // Check if company already exists in DB
      const { data: existingCompany, error: checkError } = await supabase
        .from('companies')
        .select('id, name')
        .eq('name', companyProfile.name)
        .maybeSingle();

      if (checkError && checkError.code !== 'PGRST116') {
        console.error('Error checking existing company:', checkError);
        throw checkError;
      }

      if (existingCompany) {
        console.log('Company already exists in DB:', existingCompany.id);
        // Fetch full company data
        const { data: fullCompany, error: fetchError } = await supabase
          .from('companies')
          .select('*')
          .eq('id', existingCompany.id)
          .maybeSingle();
        
        if (fetchError) {
          console.error('Error fetching company data:', fetchError);
          throw fetchError;
        }
        
        return fullCompany;
      }

      // Transform and insert company data
      const companyData = {
        name: companyProfile.name,
        industry: companyProfile.industry || 'Unknown',
        location: companyProfile.location || 'Unknown',
        employee_count: typeof companyProfile.employee_count === 'string' 
          ? parseInt(companyProfile.employee_count) 
          : companyProfile.employee_count || 0,
        products: companyProfile.products || '',
        main_materials: companyProfile.main_materials || '',
        production_volume: companyProfile.production_volume || '',
        process_description: companyProfile.process_description || '',
        sustainability_goals: companyProfile.sustainability_goals || [],
        current_waste_management: companyProfile.current_waste_management || 'Unknown',
        onboarding_completed: true
      };

      const { data: newCompany, error: insertError } = await supabase
        .from('companies')
        .insert(companyData)
        .select()
        .maybeSingle();

      if (insertError) {
        console.error('Company insertion error:', insertError);
        throw insertError;
      }

      console.log('Company migrated successfully:', newCompany.id);

      // Store portfolio data as AI insights
      if (portfolio.achievements) {
        const { error: insightsError } = await supabase
          .from('ai_insights')
          .insert({
            company_id: newCompany.id,
            insight_type: 'opportunity',
            title: 'Welcome to Industrial Symbiosis',
            description: `Your company has a sustainability score of ${portfolio.achievements.sustainability_score || 65}%. Start your symbiosis journey to improve efficiency and reduce costs.`,
            confidence: 85,
            impact: 'high',
            action_required: true,
            metadata: {
              estimated_savings: `$${portfolio.achievements.total_savings?.toLocaleString() || 0}/year`,
              carbon_reduction: `${portfolio.achievements.carbon_reduction || 0} tons CO2/year`,
              top_opportunities: portfolio.next_milestones || ['Complete first material exchange'],
              recommended_partners: ['Local manufacturers', 'Recycling facilities'],
              implementation_roadmap: ['Assess current waste streams', 'Identify potential partners']
            }
          });

        if (insightsError) {
          console.error('AI insights insertion error:', insightsError);
        }
      }

      // Clear localStorage after successful migration
      localStorage.removeItem('symbioflows-company-profile');
      localStorage.removeItem('symbioflows-portfolio');

      console.log('Migration completed successfully');
      return newCompany;

    } catch (error) {
      console.error('Migration failed:', error);
      throw error;
    }
  }

  static async getCompanyFromDB(): Promise<CompanyProfile | null> {
    try {
      // Try to get company from localStorage first (for migration)
      const localCompany = localStorage.getItem('symbioflows-company-profile');
      if (localCompany) {
        const migrated = await this.migrateCompanyData();
        if (migrated) return migrated;
      }

      // Get from database
      const { data: company, error } = await supabase
        .from('companies')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(1)
        .maybeSingle();

      if (error || !company) {
        return null;
      }

      return company;
    } catch (error) {
      console.error('Error getting company from DB:', error);
      return null;
    }
  }

  static async getPortfolioFromDB(companyId: string): Promise<PortfolioData | null> {
    try {
      // Get AI insights
      const { data: insights } = await supabase
        .from('ai_insights')
        .select('*')
        .eq('company_id', companyId)
        .order('created_at', { ascending: false })
        .limit(1)
        .maybeSingle();

      // Get materials and requirements
      const { data: materials } = await supabase
        .from('materials')
        .select('*')
        .eq('company_id', companyId);

      const { data: requirements } = await supabase
        .from('requirements')
        .select('*')
        .eq('company_id', companyId);

      // Get matches
      const { data: matches } = await supabase
        .from('matches')
        .select('*')
        .eq('company_id', companyId);

      // Construct portfolio data
      const portfolio: PortfolioData = {
        achievements: {
          total_savings: insights?.metadata?.estimated_savings ? 
            parseInt(insights.metadata.estimated_savings.replace(/[^0-9]/g, '')) : 0,
          carbon_reduction: insights?.metadata?.carbon_reduction ? 
            parseInt(insights.metadata.carbon_reduction.replace(/[^0-9]/g, '')) : 0,
          partnerships: matches?.length || 0,
          waste_diverted: materials?.length || 0,
          matches: matches?.filter(m => m.status === 'accepted').length || 0,
          sustainability_score: 65, // Default value since symbiosis_score is no longer in schema
          efficiency_improvement: 15
        },
        recommendations: materials?.map(m => ({
          id: m.id,
          category: 'material_exchange',
          title: `Exchange ${m.name}`,
          description: m.description,
          potential_impact: {
            savings: parseInt(m.potential_value?.replace(/[^0-9]/g, '') || '0'),
            carbon_reduction: 25,
            efficiency_gain: 20
          },
          implementation_difficulty: 'medium',
          time_to_implement: '2-4 weeks',
          priority: 'high',
          ai_reasoning: `AI identified ${m.name} as a valuable exchange opportunity`
        })) || [],
        recent_activity: matches?.map(m => ({
          id: m.id,
          date: m.created_at,
          action: `New match with ${m.partner_company_id}`,
          impact: `Potential savings: $${m.potential_savings}`,
          category: 'match'
        })) || [],
        next_milestones: insights?.metadata?.implementation_roadmap || [
          'Complete your first material exchange',
          'Connect with 3 potential partners',
          'Implement your first waste reduction initiative'
        ],
        industry_rank: 150
      };

      return portfolio;
    } catch (error) {
      console.error('Error getting portfolio from DB:', error);
      return null;
    }
  }

  static async triggerAIListingsGeneration(companyId: string): Promise<boolean> {
    try {
      console.log('Triggering AI listings generation for company:', companyId);
      
      const API_BASE_URL = import.meta.env.VITE_API_URL;
      const response = await fetch(`${API_BASE_URL}/api/v1/companies/${companyId}/generate-listings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        // Add timeout to prevent hanging
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });

      if (!response.ok) {
        if (response.status === 404) {
          console.error('Backend API endpoint not found. Is the backend server running?');
          throw new Error('Backend server is not running. Please start the backend server.');
        }
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('AI listings generation triggered successfully:', result);
      return result.success;
    } catch (error) {
      console.error('Error triggering AI listings generation:', error);
      
      // Check if it's a network error (backend not running)
      if (error instanceof TypeError && error.message.includes('fetch')) {
        console.error('Backend server appears to be down. Please start the backend server.');
        // Could show a user-friendly notification here
      }
      
      return false;
    }
  }
} 