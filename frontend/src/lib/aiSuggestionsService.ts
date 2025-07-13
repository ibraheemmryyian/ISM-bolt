import { supabase } from './supabase';

export interface AISuggestion {
  id: string;
  title: string;
  description: string;
  category: 'energy' | 'efficiency' | 'partnership' | 'waste' | 'materials' | 'technology';
  potential_savings: number;
  implementation_difficulty: 'easy' | 'medium' | 'hard';
  priority: 'high' | 'medium' | 'low';
  ai_reasoning: string;
  estimated_impact: string;
  implementation_time: string;
  carbon_reduction: number;
  confidence_score: number;
  created_at: string;
}

export class AISuggestionsService {
  static async generateSuggestions(userId: string): Promise<AISuggestion[]> {
    try {
      // Get user's company profile and onboarding data
      const { data: company, error: companyError } = await supabase
        .from('companies')
        .select('*')
        .eq('user_id', userId)
        .single();

      if (companyError || !company) {
        throw new Error('Company profile not found. Please complete AI onboarding first.');
      }

      // Get AI insights from onboarding
      const { data: aiInsights, error: insightsError } = await supabase
        .from('ai_insights')
        .select('*')
        .eq('company_id', userId)
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      // Generate AI suggestions based on company profile and insights
      const suggestions = await this.generateAIBasedSuggestions(company, aiInsights);
      
      // Store suggestions in database
      await this.storeSuggestions(userId, suggestions);
      
      return suggestions;
    } catch (error) {
      console.error('Error generating AI suggestions:', error);
      return [];
    }
  }

  private static async generateAIBasedSuggestions(company: any, aiInsights: any): Promise<AISuggestion[]> {
    const suggestions: AISuggestion[] = [];
    const timestamp = new Date().toISOString();

    // Analyze company profile to generate contextual suggestions
    const industry = company.industry?.toLowerCase() || '';
    const location = company.location || '';
    const employeeCount = company.employee_count || 0;
    const materials = company.main_materials?.toLowerCase() || '';
    const processes = company.process_description?.toLowerCase() || '';
    const sustainabilityGoals = company.sustainability_goals?.toLowerCase() || '';

    // Energy efficiency suggestions
    if (employeeCount > 50 || industry.includes('manufacturing') || industry.includes('production')) {
      suggestions.push({
        id: `suggestion_${Date.now()}_1`,
        title: 'Implement Smart Energy Management System',
        description: 'Install IoT sensors and smart controls to optimize energy consumption across your facility.',
        category: 'energy',
        potential_savings: 20000,
        implementation_difficulty: 'medium',
        priority: 'high',
        ai_reasoning: `Based on your ${industry} operations with ${employeeCount} employees, smart energy management could reduce energy costs by 15-25% through automated optimization and real-time monitoring.`,
        estimated_impact: '15-25% energy cost reduction',
        implementation_time: '3-6 months',
        carbon_reduction: 30,
        confidence_score: 92,
        created_at: timestamp
      });
    }

    // Waste management suggestions
    if (materials.includes('plastic') || materials.includes('metal') || materials.includes('paper')) {
      suggestions.push({
        id: `suggestion_${Date.now()}_2`,
        title: 'Establish Circular Material Recovery Program',
        description: 'Set up internal recycling systems and partner with local recyclers to recover valuable materials.',
        category: 'waste',
        potential_savings: 15000,
        implementation_difficulty: 'easy',
        priority: 'high',
        ai_reasoning: `Your use of ${materials} materials presents significant recycling opportunities. A circular recovery program could generate $15K+ annually in material sales and reduce disposal costs.`,
        estimated_impact: '30-40% waste reduction',
        implementation_time: '2-4 months',
        carbon_reduction: 25,
        confidence_score: 88,
        created_at: timestamp
      });
    }

    // Partnership suggestions based on location and industry
    if (location && industry) {
      suggestions.push({
        id: `suggestion_${Date.now()}_3`,
        title: 'Join Regional Industrial Symbiosis Network',
        description: `Connect with other ${industry} companies in ${location} to share resources and reduce costs.`,
        category: 'partnership',
        potential_savings: 25000,
        implementation_difficulty: 'medium',
        priority: 'medium',
        ai_reasoning: `Your location in ${location} and ${industry} operations make you an ideal candidate for regional industrial symbiosis partnerships, potentially saving $25K+ annually through shared resources.`,
        estimated_impact: '20-30% resource cost reduction',
        implementation_time: '4-8 months',
        carbon_reduction: 40,
        confidence_score: 85,
        created_at: timestamp
      });
    }

    // Technology upgrade suggestions
    if (processes.includes('automation') || employeeCount > 100) {
      suggestions.push({
        id: `suggestion_${Date.now()}_4`,
        title: 'Upgrade to Industry 4.0 Technologies',
        description: 'Implement advanced automation, IoT sensors, and data analytics to optimize production efficiency.',
        category: 'technology',
        potential_savings: 35000,
        implementation_difficulty: 'hard',
        priority: 'medium',
        ai_reasoning: `Your ${employeeCount}-employee operation would benefit from Industry 4.0 technologies, potentially increasing efficiency by 20-30% and reducing operational costs significantly.`,
        estimated_impact: '20-30% efficiency improvement',
        implementation_time: '6-12 months',
        carbon_reduction: 50,
        confidence_score: 78,
        created_at: timestamp
      });
    }

    // Material optimization suggestions
    if (materials && processes) {
      suggestions.push({
        id: `suggestion_${Date.now()}_5`,
        title: 'Optimize Material Usage with AI Analytics',
        description: 'Use AI-powered analytics to optimize material consumption and reduce waste in your production processes.',
        category: 'materials',
        potential_savings: 12000,
        implementation_difficulty: 'medium',
        priority: 'high',
        ai_reasoning: `Your ${materials} usage in ${processes} processes can be optimized through AI analytics, potentially reducing material costs by 10-15% while improving quality.`,
        estimated_impact: '10-15% material cost reduction',
        implementation_time: '3-5 months',
        carbon_reduction: 20,
        confidence_score: 90,
        created_at: timestamp
      });
    }

    // Add suggestions from AI insights if available
    if (aiInsights) {
      if (aiInsights.top_opportunities && aiInsights.top_opportunities.length > 0) {
        suggestions.push({
          id: `suggestion_${Date.now()}_6`,
          title: aiInsights.top_opportunities[0] || 'Custom Symbiosis Opportunity',
          description: `Based on your AI analysis, this opportunity has a ${aiInsights.symbiosis_score || '75%'} symbiosis potential.`,
          category: 'partnership',
          potential_savings: parseInt(aiInsights.estimated_savings?.replace(/[^0-9]/g, '') || '20000'),
          implementation_difficulty: 'medium',
          priority: 'high',
          ai_reasoning: `AI analysis indicates ${aiInsights.symbiosis_score || '75%'} symbiosis potential with estimated savings of ${aiInsights.estimated_savings || '$20K annually'}.`,
          estimated_impact: `${aiInsights.carbon_reduction || '15 tons CO2'} reduction annually`,
          implementation_time: '4-6 months',
          carbon_reduction: parseInt(aiInsights.carbon_reduction?.replace(/[^0-9]/g, '') || '15'),
          confidence_score: parseInt(aiInsights.symbiosis_score?.replace(/[^0-9]/g, '') || '75'),
          created_at: timestamp
        });
      }
    }

    // Sort by priority and potential savings
    return suggestions.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
      if (priorityDiff !== 0) return priorityDiff;
      return b.potential_savings - a.potential_savings;
    });
  }

  private static async storeSuggestions(userId: string, suggestions: AISuggestion[]): Promise<void> {
    try {
      const { error } = await supabase
        .from('ai_suggestions')
        .insert(suggestions.map(suggestion => ({
          company_id: userId,
          title: suggestion.title,
          description: suggestion.description,
          category: suggestion.category,
          potential_savings: suggestion.potential_savings,
          implementation_difficulty: suggestion.implementation_difficulty,
          priority: suggestion.priority,
          ai_reasoning: suggestion.ai_reasoning,
          estimated_impact: suggestion.estimated_impact,
          implementation_time: suggestion.implementation_time,
          carbon_reduction: suggestion.carbon_reduction,
          confidence_score: suggestion.confidence_score
        })));

      if (error) {
        console.error('Error storing AI suggestions:', error);
      }
    } catch (error) {
      console.error('Error storing suggestions:', error);
    }
  }

  static async getStoredSuggestions(userId: string): Promise<AISuggestion[]> {
    try {
      const { data, error } = await supabase
        .from('ai_suggestions')
        .select('*')
        .eq('company_id', userId)
        .order('created_at', { ascending: false })
        .limit(10);

      if (error) {
        console.error('Error fetching stored suggestions:', error);
        return [];
      }

      return data || [];
    } catch (error) {
      console.error('Error fetching suggestions:', error);
      return [];
    }
  }
} 