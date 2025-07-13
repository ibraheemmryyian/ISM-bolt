import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { aiService } from '../lib/aiService';
import { X, Loader2, CheckCircle, ArrowRight, ArrowLeft, Building2, Factory, Truck, FlaskConical, Users, Target, TrendingUp } from 'lucide-react';

interface AIComprehensiveOnboardingProps {
  onClose?: () => void;
  onComplete?: (portfolio: any) => void;
}

interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  fields: OnboardingField[];
  isAI?: boolean;
  aiPrompt?: string;
}

interface OnboardingField {
  id: string;
  type: 'text' | 'textarea' | 'select' | 'multiselect' | 'number' | 'slider' | 'radio';
  label: string;
  placeholder?: string;
  required: boolean;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  value: any;
}

export function AIComprehensiveOnboarding({ onClose = () => {}, onComplete = () => {} }: AIComprehensiveOnboardingProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [aiGenerating, setAiGenerating] = useState(false);
  const [error, setError] = useState('');
  const [formData, setFormData] = useState({
    // Basic Information
    company_name: '',
    industry: '',
    location: '',
    employee_count: '',
    annual_revenue: '',
    
    // Operations
    products_services: '',
    main_materials: '',
    production_processes: '',
    production_volume: '',
    operating_hours: '',
    
    // Waste & Resources
    current_waste_streams: '',
    waste_quantities: '',
    waste_frequencies: '',
    resource_needs: '',
    energy_consumption: '',
    
    // Sustainability
    sustainability_goals: [],
    environmental_certifications: '',
    current_recycling_practices: '',
    
    // Partnerships
    partnership_interests: [],
    geographic_preferences: '',
    technology_interests: '',
    
    // AI-Generated Content
    ai_generated_materials: [],
    ai_generated_opportunities: [],
    ai_portfolio_summary: '',
    ai_recommendations: []
  });

  const steps: OnboardingStep[] = [
    {
      id: 'basic_info',
      title: 'Company Overview',
      description: 'Tell us about your company',
      icon: <Building2 className="h-6 w-6" />,
      fields: [
        {
          id: 'company_name',
          type: 'text',
          label: 'Company Name',
          placeholder: 'Your company name',
          required: true,
          value: formData.company_name
        },
        {
          id: 'industry',
          type: 'select',
          label: 'Primary Industry',
          required: true,
          options: [
            'Manufacturing', 'Construction', 'Food & Beverage', 'Chemicals', 
            'Electronics', 'Automotive', 'Textiles', 'Pharmaceuticals',
            'Oil & Gas', 'Water Treatment', 'Logistics', 'Healthcare',
            'Tourism & Hospitality', 'Agriculture', 'Mining', 'Other'
          ],
          value: formData.industry
        },
        {
          id: 'location',
          type: 'text',
          label: 'Location',
          placeholder: 'City, Country',
          required: true,
          value: formData.location
        },
        {
          id: 'employee_count',
          type: 'select',
          label: 'Number of Employees',
          required: true,
          options: ['1-10', '11-50', '51-200', '201-1000', '1000+'],
          value: formData.employee_count
        },
        {
          id: 'annual_revenue',
          type: 'select',
          label: 'Annual Revenue Range',
          required: false,
          options: ['Under $1M', '$1M-$10M', '$10M-$100M', '$100M-$1B', '$1B+'],
          value: formData.annual_revenue
        }
      ]
    },
    {
      id: 'operations',
      title: 'Operations & Production',
      description: 'Describe your production processes',
      icon: <Factory className="h-6 w-6" />,
      fields: [
        {
          id: 'products_services',
          type: 'textarea',
          label: 'Products or Services',
          placeholder: 'Describe your main products or services in detail...',
          required: true,
          value: formData.products_services
        },
        {
          id: 'main_materials',
          type: 'textarea',
          label: 'Primary Raw Materials',
          placeholder: 'List the main materials, chemicals, or resources you use...',
          required: true,
          value: formData.main_materials
        },
        {
          id: 'production_processes',
          type: 'textarea',
          label: 'Production Processes',
          placeholder: 'Describe your main manufacturing or operational processes...',
          required: true,
          value: formData.production_processes
        },
        {
          id: 'production_volume',
          type: 'text',
          label: 'Production Volume',
          placeholder: 'e.g., 1000 tons/month, 5000 units/day',
          required: true,
          value: formData.production_volume
        },
        {
          id: 'operating_hours',
          type: 'select',
          label: 'Operating Hours',
          required: false,
          options: ['8 hours/day', '12 hours/day', '16 hours/day', '24 hours/day'],
          value: formData.operating_hours
        }
      ]
    },
    {
      id: 'waste_resources',
      title: 'Waste & Resource Management',
      description: 'Tell us about your waste streams and resource needs',
      icon: <Truck className="h-6 w-6" />,
      fields: [
        {
          id: 'current_waste_streams',
          type: 'textarea',
          label: 'Current Waste Streams',
          placeholder: 'Describe all waste materials you generate (type, quantity, frequency)...',
          required: true,
          value: formData.current_waste_streams
        },
        {
          id: 'waste_quantities',
          type: 'text',
          label: 'Waste Quantities',
          placeholder: 'e.g., 500 kg/day plastic, 2000 liters/week wastewater',
          required: true,
          value: formData.waste_quantities
        },
        {
          id: 'waste_frequencies',
          type: 'select',
          label: 'Waste Generation Frequency',
          required: true,
          options: ['Continuous', 'Daily', 'Weekly', 'Monthly', 'Quarterly'],
          value: formData.waste_frequencies
        },
        {
          id: 'resource_needs',
          type: 'textarea',
          label: 'Resource Requirements',
          placeholder: 'What materials, energy, or resources do you need?',
          required: false,
          value: formData.resource_needs
        },
        {
          id: 'energy_consumption',
          type: 'text',
          label: 'Energy Consumption',
          placeholder: 'e.g., 1000 kWh/day, 5000 MWh/year',
          required: false,
          value: formData.energy_consumption
        }
      ]
    },
    {
      id: 'sustainability',
      title: 'Sustainability Goals',
      description: 'What are your environmental objectives?',
      icon: <Target className="h-6 w-6" />,
      fields: [
        {
          id: 'sustainability_goals',
          type: 'multiselect',
          label: 'Sustainability Goals',
          required: false,
          options: [
            'Reduce waste by 50%', 'Achieve zero waste', 'Reduce energy consumption',
            'Increase recycling rates', 'Reduce carbon footprint', 'Use renewable energy',
            'Implement circular economy', 'Improve resource efficiency', 'Reduce water usage'
          ],
          value: formData.sustainability_goals
        },
        {
          id: 'environmental_certifications',
          type: 'text',
          label: 'Environmental Certifications',
          placeholder: 'ISO 14001, LEED, etc.',
          required: false,
          value: formData.environmental_certifications
        },
        {
          id: 'current_recycling_practices',
          type: 'textarea',
          label: 'Current Recycling Practices',
          placeholder: 'Describe your current waste management and recycling practices...',
          required: false,
          value: formData.current_recycling_practices
        }
      ]
    },
    {
      id: 'partnerships',
      title: 'Partnership Preferences',
      description: 'What types of partnerships interest you?',
      icon: <Users className="h-6 w-6" />,
      fields: [
        {
          id: 'partnership_interests',
          type: 'multiselect',
          label: 'Partnership Interests',
          required: false,
          options: [
            'Material exchange', 'Waste recycling', 'Energy sharing', 'Technology transfer',
            'Joint research', 'Supply chain optimization', 'Shared infrastructure',
            'Knowledge sharing', 'Joint ventures', 'Resource pooling'
          ],
          value: formData.partnership_interests
        },
        {
          id: 'geographic_preferences',
          type: 'text',
          label: 'Geographic Preferences',
          placeholder: 'Preferred locations for partnerships',
          required: false,
          value: formData.geographic_preferences
        },
        {
          id: 'technology_interests',
          type: 'textarea',
          label: 'Technology Interests',
          placeholder: 'What technologies or innovations interest you?',
          required: false,
          value: formData.technology_interests
        }
      ]
    },
    {
      id: 'ai_generation',
      title: 'AI Portfolio Generation',
      description: 'AI is analyzing your information and generating your portfolio',
      icon: <TrendingUp className="h-6 w-6" />,
      isAI: true,
      aiPrompt: 'Generate comprehensive industrial symbiosis portfolio based on company profile',
      fields: []
    }
  ];

  const updateField = (fieldId: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [fieldId]: value
    }));
  };

  const generateAIPortfolio = async () => {
    setAiGenerating(true);
    setError('');

    try {
      // Prepare company data for AI analysis
      const companyData = {
        companyProfile: formData,
        generateMaterials: true,
        generateOpportunities: true,
        generateRecommendations: true,
        user_id: (await supabase.auth.getUser()).data.user?.id,
        timestamp: new Date().toISOString()
      };

      // Call AI service to generate portfolio
      const response = await aiService.generateComprehensivePortfolio(companyData);

      if (!response.success) {
        throw new Error(response.error || 'Failed to generate AI portfolio');
      }

      const aiData = response.data;
      
      // Update form data with AI-generated content
      setFormData(prev => ({
        ...prev,
        ai_generated_materials: aiData.materials || [],
        ai_generated_opportunities: aiData.opportunities || [],
        ai_portfolio_summary: aiData.summary || '',
        ai_recommendations: aiData.recommendations || []
      }));

      // Save to database
      await saveToDatabase(aiData);

    } catch (err: any) {
      setError(err.message || 'Failed to generate AI portfolio');
      console.error('AI Portfolio generation error:', err);
    } finally {
      setAiGenerating(false);
    }
  };

  const saveToDatabase = async (aiData: any) => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('Not authenticated');

      // Save comprehensive company profile
      const { error: companyError } = await supabase.from('companies').upsert([{
        user_id: user.id,
        name: formData.company_name,
        industry: formData.industry,
        location: formData.location,
        employee_count: formData.employee_count,
        products: formData.products_services,
        main_materials: formData.main_materials,
        production_volume: formData.production_volume,
        process_description: formData.production_processes,
        sustainability_goals: formData.sustainability_goals,
        current_waste_management: formData.current_waste_streams,
        onboarding_completed: true,
        // Additional fields
        annual_revenue: formData.annual_revenue,
        operating_hours: formData.operating_hours,
        waste_quantities: formData.waste_quantities,
        waste_frequencies: formData.waste_frequencies,
        resource_needs: formData.resource_needs,
        energy_consumption: formData.energy_consumption,
        environmental_certifications: formData.environmental_certifications,
        current_recycling_practices: formData.current_recycling_practices,
        partnership_interests: formData.partnership_interests,
        geographic_preferences: formData.geographic_preferences,
        technology_interests: formData.technology_interests,
        // AI-generated content
        ai_portfolio_summary: aiData.summary,
        ai_recommendations: aiData.recommendations
      }]);

      if (companyError) throw companyError;

      // Get the company ID for materials and opportunities
      const { data: company } = await supabase
        .from('companies')
        .select('id')
        .eq('user_id', user.id)
        .single();

      // Save AI-generated materials
      if (aiData.materials && aiData.materials.length > 0) {
        const materials = aiData.materials.map((material: any) => ({
          ...material,
          company_id: company?.id,
          ai_generated: true
        }));

        const { data: existingMaterials, error: materialsError } = await supabase
          .from('materials')
          .select('*')
          .eq('company_id', company?.id);

        if (materialsError) throw materialsError;

        const newMaterials = materials.filter((newMaterial: any) =>
          !existingMaterials.some((existingMaterial: any) => existingMaterial.id === newMaterial.id)
        );

        if (newMaterials.length > 0) {
          const { error: newMaterialsError } = await supabase
            .from('materials')
            .insert(newMaterials);

          if (newMaterialsError) throw newMaterialsError;
        }
      }

      // Save AI-generated opportunities
      if (aiData.opportunities && aiData.opportunities.length > 0) {
        const { error: opportunitiesError } = await supabase
          .from('symbiosis_opportunities')
          .insert(aiData.opportunities.map((opp: any) => ({
            ...opp,
            company_id: company?.id,
            ai_generated: true
          })));

        if (opportunitiesError) throw opportunitiesError;
      }

      // Save AI insights
      const { data: aiInsights, error: aiInsightsError } = await supabase
        .from('ai_insights')
        .select('impact, description, metadata, confidence_score, created_at')
        .eq('company_id', company?.id)
        .order('created_at', { ascending: false })
        .limit(1);

      if (aiInsightsError) throw aiInsightsError;

    } catch (err: any) {
      throw new Error(`Database error: ${err.message}`);
    }
  };

  const handleNext = async () => {
    if (currentStep === steps.length - 2) {
      // Start AI generation
      await generateAIPortfolio();
    }
    setCurrentStep(prev => prev + 1);
  };

  const handlePrevious = () => {
    setCurrentStep(prev => prev - 1);
  };

  const handleComplete = () => {
    const portfolio = {
      company: formData,
      materials: formData.ai_generated_materials,
      opportunities: formData.ai_generated_opportunities,
      summary: formData.ai_portfolio_summary,
      recommendations: formData.ai_recommendations
    };
    onComplete?.(portfolio);
  };

  const currentStepData = steps[currentStep];
  const isLastStep = currentStep === steps.length - 1;
  const isAIStep = currentStepData.isAI;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-xl p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto relative">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 text-gray-500 hover:text-gray-700"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">
              Step {currentStep + 1} of {steps.length}
            </span>
            <span className="text-sm text-gray-600">
              {Math.round(((currentStep + 1) / steps.length) * 100)}% Complete
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Step Header */}
        <div className="flex items-center space-x-3 mb-6">
          <div className="p-2 bg-emerald-100 rounded-lg">
            {currentStepData.icon}
          </div>
          <div>
            <h2 className="text-2xl font-bold">{currentStepData.title}</h2>
            <p className="text-gray-600">{currentStepData.description}</p>
          </div>
        </div>

        {error && (
          <div className="bg-red-100 text-red-700 p-3 rounded mb-4 text-center text-sm">
            {error}
          </div>
        )}

        {/* AI Generation Step */}
        {isAIStep ? (
          <div className="text-center py-12">
            {aiGenerating ? (
              <div>
                <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4 text-emerald-500" />
                <h3 className="text-xl font-semibold mb-2">AI is analyzing your information...</h3>
                <p className="text-gray-600 mb-4">
                  Generating materials, opportunities, and recommendations based on your profile
                </p>
                <div className="space-y-2 text-sm text-gray-500">
                  <div>✓ Analyzing production processes</div>
                  <div>✓ Identifying waste streams</div>
                  <div>✓ Finding symbiosis opportunities</div>
                  <div>✓ Generating material listings</div>
                  <div>✓ Creating recommendations</div>
                </div>
              </div>
            ) : (
              <div>
                <CheckCircle className="h-12 w-12 mx-auto mb-4 text-emerald-500" />
                <h3 className="text-xl font-semibold mb-2">AI Portfolio Generated!</h3>
                <p className="text-gray-600 mb-6">
                  Your comprehensive industrial symbiosis portfolio has been created
                </p>
                
                {/* Portfolio Preview */}
                <div className="bg-gray-50 rounded-lg p-4 text-left">
                  <h4 className="font-semibold mb-2">Generated Content:</h4>
                  <div className="space-y-2 text-sm">
                    <div>• {formData.ai_generated_materials.length} material listings</div>
                    <div>• {formData.ai_generated_opportunities.length} symbiosis opportunities</div>
                    <div>• {formData.ai_recommendations.length} recommendations</div>
                    <div>• Portfolio summary and analysis</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Regular Form Step */
          <form className="space-y-6">
            {currentStepData.fields.map((field) => (
              <div key={field.id}>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {field.label}
                  {field.required && <span className="text-red-500 ml-1">*</span>}
                </label>
                
                {field.type === 'text' && (
                  <input
                    type="text"
                    required={field.required}
                    placeholder={field.placeholder}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                    value={field.value || ''}
                    onChange={(e) => updateField(field.id, e.target.value)}
                  />
                )}

                {field.type === 'textarea' && (
                  <textarea
                    required={field.required}
                    rows={4}
                    placeholder={field.placeholder}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                    value={field.value || ''}
                    onChange={(e) => updateField(field.id, e.target.value)}
                  />
                )}

                {field.type === 'select' && (
                  <select
                    required={field.required}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                    value={field.value || ''}
                    onChange={(e) => updateField(field.id, e.target.value)}
                  >
                    <option value="">Select an option</option>
                    {field.options?.map((option) => (
                      <option key={option} value={option}>{option}</option>
                    ))}
                  </select>
                )}

                {field.type === 'multiselect' && (
                  <div className="space-y-2">
                    {field.options?.map((option) => (
                      <label key={option} className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={field.value?.includes(option) || false}
                          onChange={(e) => {
                            const currentValues = field.value || [];
                            const newValues = e.target.checked
                              ? [...currentValues, option]
                              : currentValues.filter((v: string) => v !== option);
                            updateField(field.id, newValues);
                          }}
                          className="rounded border-gray-300 text-emerald-600 focus:ring-emerald-500"
                        />
                        <span className="text-sm">{option}</span>
                      </label>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </form>
        )}

        {/* Navigation */}
        <div className="flex justify-between pt-6 border-t">
          <button
            type="button"
            onClick={handlePrevious}
            disabled={currentStep === 0}
            className="flex items-center px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Previous
          </button>

          {isLastStep ? (
            <button
              onClick={handleComplete}
              className="flex items-center px-6 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600"
            >
              View Portfolio
              <ArrowRight className="h-4 w-4 ml-2" />
            </button>
          ) : (
            <button
              onClick={handleNext}
              disabled={loading || aiGenerating}
              className="flex items-center px-6 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50"
            >
              {currentStep === steps.length - 2 ? 'Generate AI Portfolio' : 'Next'}
              <ArrowRight className="h-4 w-4 ml-2" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
} 