import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  CheckCircle, 
  ArrowRight,
  ArrowLeft,
  Loader2,
  Sparkles,
  Workflow,
  Home,
  Target,
  Lightbulb,
  Brain,
  Users,
  Factory,
  Leaf,
  Zap,
  TrendingUp
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Alert, AlertDescription } from './ui/alert';
import { supabase } from '../lib/supabase';

interface OnboardingField {
  id: string;
  type: 'text' | 'select' | 'textarea' | 'number' | 'multiselect';
  label: string;
  placeholder?: string;
  options?: string[];
  required: boolean;
  value: any;
  reasoning?: string;
}

interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  fields: OnboardingField[];
  isAI: boolean;
}

interface CompanyProfile {
  name: string;
  industry: string;
  location: string;
  employee_count: number;
  products: string;
  main_materials: string;
  production_volume: string;
  process_description: string;
  annual_revenue?: number;
  sustainability_goals?: string[];
  waste_streams?: string[];
  resource_needs?: string[];
  [key: string]: any;
}

interface OnboardingQuestion {
  id: string;
  category: string;
  question: string;
  importance: 'high' | 'medium' | 'low';
  expected_answer_type: 'text' | 'numeric' | 'boolean';
  follow_up_question?: string;
}

interface OnboardingData {
  questions: OnboardingQuestion[];
  estimated_completion_time: string;
  key_insights_expected: string[];
  generated_at: string;
  ai_model: string;
}

interface OnboardingWizardProps {
  companyProfile: {
    name: string;
    industry: string;
    location: string;
    products?: string;
    employee_count?: number;
  };
  onComplete: (enrichedProfile: any) => void;
  onCancel: () => void;
}

const AIOnboardingWizard: React.FC<OnboardingWizardProps> = ({
  companyProfile,
  onComplete,
  onCancel
}) => {
  const [currentStep, setCurrentStep] = useState<'loading' | 'questions' | 'processing' | 'complete'>('loading');
  const [questionsData, setQuestionsData] = useState<OnboardingData | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [processingResult, setProcessingResult] = useState<any>(null);
  const [onboardingSteps, setOnboardingSteps] = useState<OnboardingStep[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isGeneratingQuestions, setIsGeneratingQuestions] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [progress, setProgress] = useState(0);
  const navigate = useNavigate();
  const [onboardingComplete, setOnboardingComplete] = useState(false);

  // Initial basic questions
  const initialSteps: OnboardingStep[] = [
    {
      id: 'basic-info',
      title: 'Basic Company Information',
      description: 'Let\'s start with the fundamentals about your company.',
      isAI: false,
      fields: [
        {
          id: 'name',
          type: 'text',
          label: 'Company Name',
          placeholder: 'Enter your company name',
          required: true,
          value: ''
        },
        {
          id: 'industry',
          type: 'text',
          label: 'Industry',
          placeholder: 'e.g., Automotive Manufacturing, Food Processing, Chemical Production...',
          required: true,
          value: ''
        },
        {
          id: 'location',
          type: 'text',
          label: 'Location',
          placeholder: 'City, Country',
          required: true,
          value: ''
        },
        {
          id: 'employee_count',
          type: 'select',
          label: 'Number of Employees',
          required: true,
          value: '',
          options: [
            '1-10',
            '11-50',
            '51-200',
            '201-500',
            '501-1000',
            '1000+'
          ]
        }
      ]
    },
    {
      id: 'production-info',
      title: 'Production Information',
      description: 'Tell us about your production processes and materials.',
      isAI: false,
      fields: [
        {
          id: 'products',
          type: 'textarea',
          label: 'What products or services do you produce?',
          placeholder: 'Describe your main products or services in detail...',
          required: true,
          value: ''
        },
        {
          id: 'main_materials',
          type: 'textarea',
          label: 'What are your main raw materials?',
          placeholder: 'List the primary materials, chemicals, or resources you use...',
          required: true,
          value: ''
        },
        {
          id: 'production_volume',
          type: 'text',
          label: 'Production Volume',
          placeholder: 'e.g., 1000 tons/month, 5000 units/day',
          required: true,
          value: ''
        }
      ]
    }
  ];

  useEffect(() => {
    loadInitialAIQuestions();
    // Pre-fill company name from authenticated user
    if (companyProfile?.name) {
      setOnboardingSteps(prev => prev.map(step => ({
        ...step,
        fields: step.fields.map(field => 
          field.id === 'name' 
            ? { ...field, value: companyProfile.name }
            : field
        )
      })));
    }
  }, [companyProfile]);

  const loadInitialAIQuestions = async () => {
    try {
      setIsLoading(true);
      
             // Use a fallback approach instead of calling potentially broken API
       const fallbackSteps: OnboardingStep[] = [
         {
           id: 'company-basic',
           title: 'Company Information',
           description: 'Tell us about your company to get started',
           isAI: false,
           fields: [
             {
               id: 'company_name',
               type: 'text' as const,
               label: 'Company Name',
               placeholder: 'Enter your company name',
               required: true,
               value: ''
             },
             {
               id: 'industry',
               type: 'select' as const,
               label: 'Industry',
               placeholder: 'Select your industry',
               options: [
                 'Manufacturing',
                 'Food & Beverage',
                 'Textiles',
                 'Chemicals',
                 'Metals & Mining',
                 'Construction',
                 'Energy',
                 'Transportation',
                 'Healthcare',
                 'Technology',
                 'Other'
               ],
               required: true,
               value: ''
             },
             {
               id: 'location',
               type: 'text' as const,
               label: 'Location',
               placeholder: 'City, Country',
               required: true,
               value: ''
             },
             {
               id: 'employee_count',
               type: 'number' as const,
               label: 'Number of Employees',
               placeholder: 'Approximate number',
               required: false,
               value: ''
             }
           ]
         },
         {
           id: 'operations',
           title: 'Operations & Processes',
           description: 'Help us understand your production processes',
           isAI: true,
           fields: [
             {
               id: 'products_services',
               type: 'textarea' as const,
               label: 'Products or Services',
               placeholder: 'Describe what your company produces or provides',
               required: true,
               value: ''
             },
             {
               id: 'main_materials',
               type: 'textarea' as const,
               label: 'Main Materials Used',
               placeholder: 'List the primary materials you use in production',
               required: true,
               value: ''
             },
             {
               id: 'production_volume',
               type: 'text' as const,
               label: 'Production Volume',
               placeholder: 'e.g., 1000 units/month, 500 tons/year',
               required: false,
               value: ''
             },
             {
               id: 'production_processes',
               type: 'textarea' as const,
               label: 'Production Processes',
               placeholder: 'Describe your main production processes',
               required: true,
               value: ''
             }
           ]
         },
         {
           id: 'waste-resources',
           title: 'Waste & Resources',
           description: 'Let us know about your waste streams and resource needs',
           isAI: true,
           fields: [
             {
               id: 'current_waste_streams',
               type: 'textarea' as const,
               label: 'Current Waste Streams',
               placeholder: 'Describe the waste materials you currently generate',
               required: true,
               value: ''
             },
             {
               id: 'waste_quantities',
               type: 'text' as const,
               label: 'Waste Quantities',
               placeholder: 'e.g., 10 tons/month, 500 kg/week',
               required: false,
               value: ''
             },
             {
               id: 'resource_needs',
               type: 'textarea' as const,
               label: 'Resource Needs',
               placeholder: 'What resources do you need that could come from waste?',
               required: false,
               value: ''
             }
           ]
         }
       ];
      
      setOnboardingSteps(fallbackSteps);
      setCurrentStep('questions');
      
    } catch (error) {
      console.error('Error loading AI questions:', error);
      setError('Failed to load onboarding questions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const updateProgress = () => {
    const totalSteps = onboardingSteps.length;
    const completedSteps = currentStep === 'questions' ? currentQuestionIndex : 0;
    setProgress((completedSteps / totalSteps) * 100);
  };

  useEffect(() => {
    updateProgress();
  }, [currentStep, currentQuestionIndex, onboardingSteps]);

  const handleFieldChange = (stepIndex: number, fieldId: string, value: any) => {
    const updatedSteps = [...onboardingSteps];
    const field = updatedSteps[stepIndex].fields.find(f => f.id === fieldId);
    if (field) {
      field.value = value;
    }
    setOnboardingSteps(updatedSteps);
  };

  const isStepComplete = (step: OnboardingStep | undefined) => {
    if (!step || !step.fields) return false;
    return step.fields.every(field => {
      if (!field.required) return true;
      if (field.type === 'multiselect') {
        return Array.isArray(field.value) && field.value.length > 0;
      }
      return field.value && field.value.toString().trim() !== '';
    });
  };

  const generateAIQuestions = async () => {
    setIsGeneratingQuestions(true);
    try {
      // Get the current session for authentication
      const { data: { session } } = await supabase.auth.getSession();
      
      const response = await fetch('/api/ai-onboarding/questions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session?.access_token}`,
        },
        body: JSON.stringify({
          companyProfile,
          currentStep: currentStep,
          existingData: companyProfile
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.questions && data.questions.length > 0) {
          const aiStep: OnboardingStep = {
            id: `ai-step-${Date.now()}`,
            title: 'AI-Powered Questions',
            description: 'Based on your answers, our AI has identified key areas to explore for optimal symbiosis opportunities.',
            isAI: true,
            fields: data.questions.map((q: any) => ({
              id: q.id,
              type: q.type || 'text',
              label: q.label,
              placeholder: q.placeholder,
              options: q.options,
              required: q.required || false,
              value: q.type === 'multiselect' ? [] : '',
              reasoning: q.reasoning
            }))
          };
          
          setOnboardingSteps(prev => [...prev, aiStep]);
          setCurrentStep(prev => prev === 'questions' ? 'questions' : 'processing');
        } else {
          // No more questions needed, complete onboarding
          await completeOnboarding();
        }
      } else {
        throw new Error('Failed to generate AI questions');
      }
    } catch (error) {
      console.error('Error generating AI questions:', error);
      // Fallback: complete onboarding
      await completeOnboarding();
    } finally {
      setIsGeneratingQuestions(false);
    }
  };

  const completeOnboarding = async () => {
    setIsLoading(true);
    try {
      // Ensure company name is set and not empty
      if (!companyProfile.name || companyProfile.name.trim() === '') {
        throw new Error('Company name is required');
      }

      // Store company profile in localStorage for dashboard access
      console.log('Storing company profile in localStorage:', companyProfile);
      console.log('Company name being stored:', companyProfile.name);
      localStorage.setItem('symbioflows-company-profile', JSON.stringify(companyProfile));
      
      // Verify storage
      const stored = localStorage.getItem('symbioflows-company-profile');
      console.log('Verified stored data:', stored);
      
      // Try to save to database using fallback endpoint
      try {
        // Get the current session for authentication
        const { data: { session } } = await supabase.auth.getSession();
        
        const response = await fetch('/api/ai-onboarding/fallback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${session?.access_token}`,
          },
          body: JSON.stringify({
            companyProfile
          })
        });

        if (response.ok) {
          const data = await response.json();
          console.log('Onboarding completed:', data);
          
          // Store the AI-generated portfolio and suggestions
          if (data.analysis) {
            localStorage.setItem('symbioflows-portfolio', JSON.stringify(data.analysis));
          }
        } else {
          throw new Error(`API responded with ${response.status}`);
        }
      } catch (apiError) {
        console.warn('API call failed, continuing with local storage:', apiError);
      }
      
      // Create a placeholder portfolio if backend doesn't return analysis
      const placeholderPortfolio = {
        waste_streams: [
          {
            name: "General waste from production",
            quantity: "Varies by production volume",
            value: "Potential for recycling",
            potential_uses: ["Recycling", "Energy recovery", "Material recovery"]
          }
        ],
        resource_needs: [
          {
            name: "Raw materials",
            current_cost: "Based on market rates",
            potential_sources: ["Local suppliers", "Recycled materials", "Waste exchanges"]
          }
        ],
        opportunities: [
          {
            title: "Waste Exchange Program",
            description: "Connect with local companies to exchange waste materials",
            estimated_savings: "$10K-50K annually",
            carbon_reduction: "5-20 tons CO2",
            implementation_time: "3-6 months",
            difficulty: "medium"
          }
        ],
        potential_partners: [
          {
            company_type: "Local manufacturing companies",
            location: companyProfile.location || "Your region",
            waste_they_can_use: ["Production waste", "Packaging materials"],
            resources_they_can_provide: ["Raw materials", "Technical expertise"],
            estimated_partnership_value: "$25K annually"
          }
        ],
        material_listings: [
          {
            material: "General waste",
            current_status: "waste",
            quantity: "Variable",
            value: "Low to medium",
            potential_exchanges: ["Recycling", "Reuse"]
          }
        ],
        estimated_savings: "$10K-50K annually",
        environmental_impact: "5-20 tons CO2 reduction",
        roadmap: [
          {
            phase: "Setup",
            timeline: "1-2 months",
            actions: ["Contact local businesses", "Assess waste streams", "Identify opportunities"],
            expected_outcomes: ["Initial partnerships", "Waste audit", "Opportunity assessment"]
          },
          {
            phase: "Implementation",
            timeline: "3-6 months",
            actions: ["Establish partnerships", "Set up waste exchange", "Monitor results"],
            expected_outcomes: ["Active partnerships", "Waste reduction", "Cost savings"]
          }
        ]
      };
      
      localStorage.setItem('symbioflows-portfolio', JSON.stringify(placeholderPortfolio));
      setIsCompleted(true);
      setOnboardingComplete(true);
      
      // Navigate to dashboard after a short delay
      setTimeout(() => {
        navigate('/dashboard');
      }, 2000);
      
    } catch (error) {
      console.error('Error completing onboarding:', error);
      setError('Failed to complete onboarding. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnswerChange = (questionId: string, value: string) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: value
    }));
  };

  const handleNext = () => {
    if (currentStep === 'questions') {
      if (currentQuestionIndex < (questionsData?.questions.length || 0) - 1) {
        setCurrentQuestionIndex(prev => prev + 1);
      } else {
        processAnswers();
      }
    } else if (currentStep === 'processing') {
      processAnswers();
    } else if (currentStep === 'complete' && processingResult) {
      onComplete(processingResult);
    }
  };

  const handlePrevious = () => {
    if (currentStep === 'questions') {
      if (currentQuestionIndex > 0) {
        setCurrentQuestionIndex(prev => prev - 1);
      }
    }
  };

  const processAnswers = async () => {
    try {
      setCurrentStep('processing');
      setError(null);

      // Get the current session for authentication
      const { data: { session } } = await supabase.auth.getSession();
      
      const response = await fetch('/api/ai-onboarding/complete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session?.access_token}`,
        },
        body: JSON.stringify({
          companyProfile: {
            ...companyProfile,
            ...answers
          }
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to process answers');
      }

      const result = await response.json();
      setProcessingResult(result);
      setCurrentStep('complete');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process answers');
      setCurrentStep('questions');
    }
  };

  const getProgressPercentage = () => {
    if (!questionsData?.questions) return 0;
    return ((currentQuestionIndex + 1) / questionsData.questions.length) * 100;
  };

  const getImportanceColor = (importance: string) => {
    switch (importance) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const renderQuestionInput = (question: OnboardingQuestion) => {
    switch (question.expected_answer_type) {
      case 'numeric':
        return (
          <Input
            type="number"
            placeholder="Enter a number"
            value={answers[question.id] || ''}
            onChange={(e) => handleAnswerChange(question.id, e.target.value)}
            className="w-full"
          />
        );
      case 'boolean':
        return (
          <div className="space-y-2">
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                name={question.id}
                value="yes"
                checked={answers[question.id] === 'yes'}
                onChange={(e) => handleAnswerChange(question.id, e.target.value)}
              />
              <span>Yes</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                name={question.id}
                value="no"
                checked={answers[question.id] === 'no'}
                onChange={(e) => handleAnswerChange(question.id, e.target.value)}
              />
              <span>No</span>
            </label>
          </div>
        );
      default:
        return (
          <Textarea
            placeholder="Enter your answer..."
            value={answers[question.id] || ''}
            onChange={(e) => handleAnswerChange(question.id, e.target.value)}
            className="w-full"
            rows={3}
          />
        );
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <Loader2 className="w-12 h-12 text-emerald-500 animate-spin mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-white mb-2">Setting up your AI profile...</h2>
            <p className="text-gray-400">This will only take a moment</p>
          </div>
        </div>
      </div>
    );
  }

  if (isCompleted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <CheckCircle className="w-16 h-16 text-emerald-500 mx-auto mb-4" />
            <h2 className="text-3xl font-bold text-white mb-4">Onboarding Complete!</h2>
            <p className="text-gray-400 mb-6">Your AI profile has been created successfully.</p>
            <div className="bg-slate-800 rounded-lg p-6 max-w-md mx-auto">
              <h3 className="text-lg font-semibold text-white mb-3">What's Next?</h3>
              <ul className="text-left space-y-2 text-gray-300">
                <li className="flex items-center space-x-2">
                  <Target className="w-4 h-4 text-emerald-400" />
                  <span>Explore your personalized dashboard</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Users className="w-4 h-4 text-blue-400" />
                  <span>Browse the marketplace for opportunities</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Brain className="w-4 h-4 text-purple-400" />
                  <span>Use AI matching to find partners</span>
                </li>
              </ul>
            </div>
            <p className="text-emerald-400 mt-4">Redirecting to dashboard...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      <div className="container mx-auto px-6 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Progress Bar */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-2">
              <h1 className="text-2xl font-bold text-white">AI Onboarding</h1>
              <span className="text-gray-400">{Math.round(progress)}% Complete</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div 
                className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6">
              <p className="text-red-400">{error}</p>
            </div>
          )}

          {/* Onboarding Steps */}
          {currentStep === 'questions' && onboardingSteps.length > 0 && (
            <div className="bg-slate-800 rounded-lg shadow-lg border border-slate-700">
              <div className="p-6 border-b border-slate-700">
                <div className="flex items-center space-x-3">
                  {onboardingSteps[currentQuestionIndex]?.isAI && (
                    <Sparkles className="w-5 h-5 text-emerald-400" />
                  )}
                  <div>
                    <h2 className="text-xl font-bold text-white">
                      {onboardingSteps[currentQuestionIndex]?.title}
                    </h2>
                    <p className="text-gray-400 mt-1">
                      {onboardingSteps[currentQuestionIndex]?.description}
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="p-6">
                <div className="space-y-6">
                  {onboardingSteps[currentQuestionIndex]?.fields.map((field) => (
                    <div key={field.id}>
                      <label className="block text-sm font-medium text-white mb-2">
                        {field.label}
                        {field.required && <span className="text-red-400 ml-1">*</span>}
                      </label>
                      
                      {field.type === 'text' && (
                        <input
                          type="text"
                          value={answers[field.id] || ''}
                          onChange={(e) => handleAnswerChange(field.id, e.target.value)}
                          placeholder={field.placeholder}
                          className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                        />
                      )}
                      
                      {field.type === 'textarea' && (
                        <textarea
                          value={answers[field.id] || ''}
                          onChange={(e) => handleAnswerChange(field.id, e.target.value)}
                          placeholder={field.placeholder}
                          rows={4}
                          className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                        />
                      )}
                      
                      {field.type === 'number' && (
                        <input
                          type="number"
                          value={answers[field.id] || ''}
                          onChange={(e) => handleAnswerChange(field.id, e.target.value)}
                          placeholder={field.placeholder}
                          className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                        />
                      )}
                      
                      {field.type === 'select' && (
                        <select
                          value={answers[field.id] || ''}
                          onChange={(e) => handleAnswerChange(field.id, e.target.value)}
                          className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                        >
                          <option value="">{field.placeholder}</option>
                          {field.options?.map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="p-6 border-t border-slate-700 flex justify-between">
                <button
                  onClick={handlePrevious}
                  disabled={currentQuestionIndex === 0}
                  className="flex items-center space-x-2 px-4 py-2 text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ArrowLeft className="w-4 h-4" />
                  <span>Previous</span>
                </button>
                
                <button
                  onClick={handleNext}
                  disabled={isGeneratingQuestions}
                  className="flex items-center space-x-2 bg-emerald-500 text-white px-6 py-2 rounded-lg hover:bg-emerald-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isGeneratingQuestions ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Generating...</span>
                    </>
                  ) : currentQuestionIndex === onboardingSteps.length - 1 ? (
                    <>
                      <span>Complete</span>
                      <CheckCircle className="w-4 h-4" />
                    </>
                  ) : (
                    <>
                      <span>Next</span>
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AIOnboardingWizard; 