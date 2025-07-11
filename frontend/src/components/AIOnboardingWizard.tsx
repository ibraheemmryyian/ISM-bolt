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
  Brain
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
      
      // Fetch initial AI-generated questions from the backend
      const response = await fetch('/api/ai-onboarding/initial-questions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          companyProfile: {
            name: '',
            industry: '',
            location: '',
            employee_count: 0,
            products: '',
            main_materials: '',
            production_volume: '',
            process_description: ''
          }
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.questions && data.questions.length > 0) {
          // Replace initial steps with AI-generated questions
          const aiSteps = data.questions.map((questionSet: any, index: number) => ({
            id: `ai-step-${index}`,
            title: questionSet.title || `AI-Powered Questions ${index + 1}`,
            description: questionSet.description || 'Our AI has identified key areas to explore for optimal symbiosis opportunities.',
            isAI: true,
            fields: questionSet.fields.map((field: any) => ({
              id: field.id,
              type: field.type || 'text',
              label: field.label,
              placeholder: field.placeholder,
              options: field.options,
              required: field.required || false,
              value: field.type === 'multiselect' ? [] : '',
              reasoning: field.reasoning
            }))
          }));
          
          setOnboardingSteps(aiSteps);
        } else {
          // Fallback to initial steps if no AI questions
          setOnboardingSteps(initialSteps);
        }
      } else {
        // Fallback to initial steps if API fails
        setOnboardingSteps(initialSteps);
      }
    } catch (error) {
      console.error('Error loading AI questions:', error);
      // Fallback to initial steps
      setOnboardingSteps(initialSteps);
    } finally {
      setIsLoading(false);
      updateProgress();
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
      const response = await fetch('/api/ai-onboarding/questions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
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
      
      const response = await fetch('/api/ai-onboarding/complete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          companyProfile,
          onboardingData: onboardingSteps
        })
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Onboarding completed:', data);
        
        // Store the AI-generated portfolio and suggestions
        if (data.analysis) {
          localStorage.setItem('symbioflows-portfolio', JSON.stringify(data.analysis));
        } else {
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
                material: "Production waste",
                current_status: "waste",
                quantity: "Based on production volume",
                value: "Variable",
                potential_exchanges: ["Recycling", "Material recovery", "Energy generation"]
              }
            ],
            estimated_savings: "$15K-75K annually",
            environmental_impact: "10-40 tons CO2 reduction",
            roadmap: [
              {
                phase: "Initial Assessment",
                timeline: "1-2 months",
                actions: ["Audit current waste streams", "Identify potential partners", "Assess market opportunities"],
                expected_outcomes: ["Waste inventory", "Partner list", "Opportunity assessment"]
              }
            ]
          };
          localStorage.setItem('symbioflows-portfolio', JSON.stringify(placeholderPortfolio));
        }
        if (data.recommendations) {
          localStorage.setItem('symbioflows-recommendations', JSON.stringify(data.recommendations));
        }
        if (data.next_steps) {
          localStorage.setItem('symbioflows-next-steps', JSON.stringify(data.next_steps));
        }
        
        setIsCompleted(true);
      } else {
        throw new Error('Failed to complete onboarding');
      }
    } catch (error) {
      console.error('Error completing onboarding:', error);
      // Even if API fails, store the company profile locally and create placeholder portfolio
      localStorage.setItem('symbioflows-company-profile', JSON.stringify(companyProfile));
      
      // Create a basic placeholder portfolio for offline functionality
      const basicPortfolio = {
        waste_streams: [
          {
            name: "Production waste",
            quantity: "Based on your production volume",
            value: "Variable",
            potential_uses: ["Recycling", "Material recovery"]
          }
        ],
        resource_needs: [
          {
            name: "Raw materials",
            current_cost: "Market dependent",
            potential_sources: ["Local suppliers", "Recycled sources"]
          }
        ],
        opportunities: [
          {
            title: "Basic Waste Exchange",
            description: "Start with simple waste exchange opportunities",
            estimated_savings: "$5K-25K annually",
            carbon_reduction: "2-10 tons CO2",
            implementation_time: "1-3 months",
            difficulty: "easy"
          }
        ],
        potential_partners: [
          {
            company_type: "Local businesses",
            location: companyProfile.location || "Your area",
            waste_they_can_use: ["General waste"],
            resources_they_can_provide: ["Materials", "Services"],
            estimated_partnership_value: "$10K annually"
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
        estimated_savings: "$5K-25K annually",
        environmental_impact: "2-10 tons CO2 reduction",
        roadmap: [
          {
            phase: "Setup",
            timeline: "1 month",
            actions: ["Contact local businesses", "Assess waste streams"],
            expected_outcomes: ["Initial partnerships", "Waste audit"]
          }
        ]
      };
      localStorage.setItem('symbioflows-portfolio', JSON.stringify(basicPortfolio));
      setIsCompleted(true);
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

      const response = await fetch('/api/ai/process-onboarding-answers', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          questions: questionsData?.questions,
          answers
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

  if (currentStep === 'loading') {
    return (
      <Card className="w-full max-w-2xl mx-auto">
        <CardContent className="flex flex-col items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">AI is generating personalized questions...</h3>
          <p className="text-gray-600 text-center">
            Our AI is analyzing your company profile to create the most relevant questions for accurate industrial symbiosis matching.
          </p>
        </CardContent>
      </Card>
    );
  }

  if (currentStep === 'questions' && questionsData) {
    const currentQuestion = questionsData.questions[currentQuestionIndex];
    const isLastQuestion = currentQuestionIndex === questionsData.questions.length - 1;

    return (
      <Card className="w-full max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-blue-600" />
            AI-Powered Onboarding
          </CardTitle>
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-gray-600">
              <span>Question {currentQuestionIndex + 1} of {questionsData.questions.length}</span>
              <span>{questionsData.estimated_completion_time}</span>
            </div>
            <Progress value={getProgressPercentage()} className="h-2" />
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {error && (
            <Alert>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="space-y-4">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h3 className="text-lg font-medium mb-2">{currentQuestion.question}</h3>
                {currentQuestion.follow_up_question && (
                  <p className="text-sm text-gray-600 mb-3">{currentQuestion.follow_up_question}</p>
                )}
              </div>
              <Badge className={getImportanceColor(currentQuestion.importance)}>
                {currentQuestion.importance} priority
              </Badge>
            </div>

            {renderQuestionInput(currentQuestion)}

            <div className="flex justify-between pt-4">
              <Button
                variant="outline"
                onClick={handlePrevious}
                disabled={currentQuestionIndex === 0}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Previous
              </Button>
              <Button
                onClick={handleNext}
                disabled={!answers[currentQuestion.id]}
              >
                {isLastQuestion ? 'Complete' : 'Next'}
                {!isLastQuestion && <ArrowRight className="h-4 w-4 ml-2" />}
              </Button>
            </div>
          </div>

          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-medium text-blue-900 mb-2">Expected Insights:</h4>
            <ul className="text-sm text-blue-800 space-y-1">
              {questionsData.key_insights_expected.map((insight, index) => (
                <li key={index} className="flex items-center gap-2">
                  <CheckCircle className="h-3 w-3" />
                  {insight}
                </li>
              ))}
            </ul>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (currentStep === 'processing') {
    return (
      <Card className="w-full max-w-2xl mx-auto">
        <CardContent className="flex flex-col items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Processing your answers...</h3>
          <p className="text-gray-600 text-center">
            Our AI is analyzing your responses to create a comprehensive company profile for optimal industrial symbiosis matching.
          </p>
        </CardContent>
      </Card>
    );
  }

  if (currentStep === 'complete' && processingResult) {
    return (
      <Card className="w-full max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            Onboarding Complete!
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <Alert>
            <CheckCircle className="h-4 w-4" />
            <AlertDescription>
              Your company profile has been successfully enriched with {Object.keys(processingResult.enriched_profile).length} categories of information.
            </AlertDescription>
          </Alert>

          <div className="space-y-4">
            <h4 className="font-medium">Profile Confidence Score</h4>
            <div className="flex items-center gap-2">
              <Progress value={processingResult.confidence_score * 100} className="flex-1" />
              <span className="text-sm font-medium">
                {Math.round(processingResult.confidence_score * 100)}%
              </span>
            </div>
          </div>

          <div className="space-y-4">
            <h4 className="font-medium">Key Insights Generated</h4>
            <div className="grid grid-cols-1 gap-2">
              {processingResult.insights.map((insight: string, index: number) => (
                <div key={index} className="flex items-center gap-2 text-sm">
                  <CheckCircle className="h-3 w-3 text-green-600" />
                  {insight}
                </div>
              ))}
            </div>
          </div>

          <div className="flex justify-between pt-4">
            <Button variant="outline" onClick={onCancel}>
              Cancel
            </Button>
            <Button onClick={handleNext}>
              Complete Setup
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (onboardingComplete) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-white to-blue-50 flex items-center justify-center">
        <div className="bg-white rounded-xl shadow-lg p-8 max-w-md w-full text-center">
          <div className="animate-pulse">
            <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <CheckCircle className="w-8 h-8 text-emerald-500" />
            </div>
            <h2 className="text-2xl font-bold text-emerald-700 mb-2">Onboarding Complete!</h2>
            <p className="text-slate-600 mb-4">
              Your AI profile has been created and we're generating your personalized materials and matches.
            </p>
            <div className="flex items-center justify-center space-x-2">
              <div className="w-4 h-4 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-emerald-600">Preparing your dashboard...</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default AIOnboardingWizard; 