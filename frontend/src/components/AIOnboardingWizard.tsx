import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  ArrowLeft, 
  ArrowRight, 
  CheckCircle, 
  Loader2,
  Home,
  Target,
  Lightbulb,
  Brain,
  AlertCircle,
  Info,
  TrendingUp
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Alert, AlertDescription } from './ui/alert';
import { supabase } from '../lib/supabase';

interface OnboardingField {
  id: string;
  type: 'text' | 'select' | 'textarea' | 'number' | 'multiselect' | 'boolean';
  label: string;
  placeholder?: string;
  options?: string[];
  required: boolean;
  value: any;
  reasoning?: string;
  importance?: 'high' | 'medium' | 'low';
}

interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  fields: OnboardingField[];
  isAI: boolean;
  category?: string;
}

interface CompanyProfile {
  name: string;
  industry: string;
  location: string;
  employee_count: number;
  products?: string;
  main_materials?: string;
  production_volume?: string;
  process_description?: string;
  waste_streams?: string;
  sustainability_goals?: string;
}

interface OnboardingQuestion {
  id: string;
  category: string;
  question: string;
  importance: 'high' | 'medium' | 'low';
  expected_answer_type: 'text' | 'numeric' | 'boolean' | 'multiselect';
  follow_up_question?: string;
  options?: string[];
  reasoning?: string;
}

interface OnboardingData {
  questions: OnboardingQuestion[];
  estimated_completion_time: string;
  key_insights_expected: string[];
  material_listings_focus: string[];
  ai_model: string;
}

interface KnowledgeAssessment {
  confidence_score: number;
  existing_data: Record<string, boolean>;
  critical_gaps: string[];
  knowledge_areas: string[];
  data_completeness: string;
}

interface OnboardingWizardProps {
  companyProfile: {
    name: string;
    industry: string;
    location: string;
    employee_count: number;
  };
  onComplete: (profile: any) => void;
  onCancel: () => void;
}

const AIOnboardingWizard: React.FC<OnboardingWizardProps> = ({
  companyProfile,
  onComplete,
  onCancel
}) => {
  const [currentStep, setCurrentStep] = useState<'loading' | 'assessment' | 'questions' | 'processing' | 'complete'>('loading');
  const [questionsData, setQuestionsData] = useState<OnboardingData | null>(null);
  const [knowledgeAssessment, setKnowledgeAssessment] = useState<KnowledgeAssessment | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [processingResult, setProcessingResult] = useState<any>(null);
  const [isGeneratingQuestions, setIsGeneratingQuestions] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [industryCategory, setIndustryCategory] = useState<string>('');
  const navigate = useNavigate();
  const [onboardingComplete, setOnboardingComplete] = useState(false);

  const [onboardingSteps, setOnboardingSteps] = useState<OnboardingStep[]>([
    {
      id: 'basic-info',
      title: 'Basic Company Information',
      description: 'Let\'s start with some basic information about your company',
      isAI: false,
      fields: [
        {
          id: 'name',
          type: 'text',
          label: 'Company Name',
          placeholder: 'Enter your company name',
          required: true,
          value: '',
          importance: 'high'
        },
        {
          id: 'industry',
          type: 'text',
          label: 'Industry',
          placeholder: 'e.g., Chemical Manufacturing, Food Processing, Steel Production...',
          required: true,
          value: '',
          importance: 'high'
        },
        {
          id: 'location',
          type: 'text',
          label: 'Location',
          placeholder: 'City, Country',
          required: true,
          value: '',
          importance: 'high'
        },
        {
          id: 'employee_count',
          type: 'select',
          label: 'Number of Employees',
          required: true,
          value: '',
          options: [
            '1-50',
            '51-200',
            '201-500',
            '501-1000',
            '1000+'
          ],
          importance: 'medium'
        }
      ]
    }
  ]);

  useEffect(() => {
    startKnowledgeAssessment();
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

  const startKnowledgeAssessment = async () => {
    try {
      setIsLoading(true);
      setCurrentStep('loading');

      // Call the enhanced AI onboarding assessment
      const response = await fetch('/api/ai-onboarding/assess-knowledge', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          companyProfile: {
            name: companyProfile?.name || '',
            industry: companyProfile?.industry || '',
            location: companyProfile?.location || '',
            employee_count: companyProfile?.employee_count || 0,
            products: (companyProfile as any)?.products || '',
            main_materials: '',
            production_volume: '',
            process_description: ''
          }
        })
      });

      if (response.ok) {
        const data = await response.json();

        if (data.knowledge_assessment) {
          setKnowledgeAssessment(data.knowledge_assessment);
          setIndustryCategory(data.industry_category || '');

          // If confidence is low, show assessment screen
          if (data.knowledge_assessment.confidence_score < 0.5) {
            setCurrentStep('assessment');
          } else {
            // If confidence is high, proceed to questions
            if (data.questions_data) {
              setQuestionsData(data.questions_data);
              setCurrentStep('questions');
            }
          }
        }
      } else {
        throw new Error('Failed to assess knowledge gaps');
      }
    } catch (error) {
      console.error('Error starting knowledge assessment:', error);
      setError('Failed to start knowledge assessment. Please try again.');
      setCurrentStep('questions'); // Fallback to basic questions
    } finally {
      setIsLoading(false);
    }
  };

  const generateAIQuestions = async () => {
    try {
      setIsGeneratingQuestions(true);

      // Get current form data
      const formData = getCurrentFormData();

      // Call AI to generate targeted questions
      const response = await fetch('/api/ai-onboarding/generate-questions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          companyProfile: formData,
          knowledgeAssessment: knowledgeAssessment
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.questions && data.questions.length > 0) {
          setQuestionsData(data);
          setCurrentStep('questions');
        } else {
          throw new Error('No questions generated');
        }
      } else {
        throw new Error('Failed to generate questions');
      }
    } catch (error) {
      console.error('Error generating AI questions:', error);
      setError('Failed to generate AI questions. Using standard questions.');
      // Fallback to standard questions
      setCurrentStep('questions');
    } finally {
      setIsGeneratingQuestions(false);
    }
  };

  const getCurrentFormData = () => {
    const formData: any = {};
    onboardingSteps.forEach(step => {
      step.fields.forEach(field => {
        formData[field.id] = field.value;
      });
    });
    return formData;
  };

  const updateProgress = () => {
    const totalSteps = onboardingSteps.length;
    const completedSteps = onboardingSteps.filter(step => isStepComplete(step)).length;
    const progressPercentage = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;
    setProgress(progressPercentage);
  };

  const handleFieldChange = (stepIndex: number, fieldId: string, value: any) => {
    setOnboardingSteps(prev => prev.map((step, index) => 
      index === stepIndex 
        ? {
            ...step,
            fields: step.fields.map(field => 
              field.id === fieldId 
                ? { ...field, value }
                : field
            )
          }
        : step
    ));
    updateProgress();
  };

  const isStepComplete = (step: OnboardingStep | undefined): boolean => {
    if (!step) return false;
    return step.fields.every((field: OnboardingField) => {
      if (field.required) {
        if (field.type === 'multiselect') {
          return Array.isArray(field.value) && field.value.length > 0;
        }
        return field.value !== '' && field.value !== null && field.value !== undefined;
      }
      return true;
    });
  };

  const handleAnswerChange = (questionId: string, value: string) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: value
    }));
  };

  const handleNext = () => {
    if (currentStep === 'assessment') {
      generateAIQuestions();
    } else if (currentStep === 'questions') {
      if (currentQuestionIndex < (questionsData?.questions?.length || 0) - 1) {
        setCurrentQuestionIndex(prev => prev + 1);
      } else {
        processAnswers();
      }
    }
  };

  const handlePrevious = () => {
    if (currentStep === 'questions' && currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prev => prev - 1);
    }
  };

  const processAnswers = async () => {
    try {
      setCurrentStep('processing');

      // Get all form data
      const formData = getCurrentFormData();
      const allAnswers = { ...formData, ...answers };

      // Generate material listings from answers
      const response = await fetch('/api/ai-onboarding/generate-listings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          companyProfile: formData,
          answers: allAnswers
        })
      });

      if (response.ok) {
        const result = await response.json();
        setProcessingResult(result);
        setCurrentStep('complete');
        setIsCompleted(true);

        // Call onComplete with enriched profile
        const enrichedProfile = {
          ...formData,
          ...allAnswers,
          material_listings: result.material_listings,
          waste_requirements: result.waste_requirements,
          ai_generated: true,
          confidence_score: result.confidence_score
        };

        onComplete(enrichedProfile);
      } else {
        throw new Error('Failed to generate material listings');
      }
    } catch (error) {
      console.error('Error processing answers:', error);
      setError('Failed to process answers. Please try again.');
    }
  };

  const getProgressPercentage = () => {
    if (currentStep === 'loading') return 0;
    if (currentStep === 'assessment') return 20;
    if (currentStep === 'questions') return 40 + (currentQuestionIndex / (questionsData?.questions?.length || 1)) * 40;
    if (currentStep === 'processing') return 80;
    if (currentStep === 'complete') return 100;
    return 0;
  };

  const getImportanceColor = (importance: string) => {
    switch (importance) {
      case 'high': return 'text-red-600 bg-red-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const renderQuestionInput = (question: OnboardingQuestion) => {
    const value = answers[question.id] || '';

    switch (question.expected_answer_type) {
      case 'multiselect':
        return (
          <div className="space-y-2">
            {question.options?.map((option, index) => (
              <label key={index} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={Array.isArray(value) ? value.includes(option) : false}
                  onChange={(e) => {
                    const currentValues = Array.isArray(value) ? value : [];
                    const newValues = e.target.checked
                      ? [...currentValues, option]
                      : currentValues.filter(v => v !== option);
                    handleAnswerChange(question.id, JSON.stringify(newValues));
                  }}
                  className="rounded border-gray-300"
                />
                <span className="text-sm">{option}</span>
              </label>
            ))}
          </div>
        );

      case 'boolean':
        return (
          <div className="space-y-2">
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                name={question.id}
                value="true"
                checked={value === 'true'}
                onChange={(e) => handleAnswerChange(question.id, e.target.value)}
                className="rounded border-gray-300"
              />
              <span className="text-sm">Yes</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                name={question.id}
                value="false"
                checked={value === 'false'}
                onChange={(e) => handleAnswerChange(question.id, e.target.value)}
                className="rounded border-gray-300"
              />
              <span className="text-sm">No</span>
            </label>
          </div>
        );

      case 'numeric':
        return (
          <Input
            type="number"
            value={value}
            onChange={(e) => handleAnswerChange(question.id, e.target.value)}
            placeholder="Enter a number"
            className="w-full"
          />
        );

      default:
        return (
          <Textarea
            value={value}
            onChange={(e) => handleAnswerChange(question.id, e.target.value)}
            placeholder="Enter your answer..."
            className="w-full min-h-[100px]"
          />
        );
    }
  };

  const renderAssessmentScreen = () => (
    <div className="max-w-4xl mx-auto space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-6 w-6 text-blue-600" />
            <span>AI Knowledge Assessment</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {knowledgeAssessment && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Data Completeness</span>
                    <Badge variant={knowledgeAssessment.confidence_score > 0.7 ? 'default' : 'outline'}>
                      {knowledgeAssessment.data_completeness}
                    </Badge>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Confidence Score</span>
                      <span className="text-sm font-medium">
                        {Math.round(knowledgeAssessment.confidence_score * 100)}%
                      </span>
                    </div>
                    <Progress value={knowledgeAssessment.confidence_score * 100} className="w-full" />
                  </div>

                  {industryCategory && (
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <Info className="h-4 w-4 text-blue-600" />
                        <span className="text-sm font-medium">Industry Category</span>
                      </div>
                      <p className="text-sm text-blue-700 mt-1">
                        {industryCategory.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </p>
                    </div>
                  )}
                </div>

                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium mb-2">Critical Knowledge Gaps</h4>
                    <div className="space-y-1">
                      {knowledgeAssessment.critical_gaps.map((gap, index) => (
                        <div key={index} className="flex items-center space-x-2">
                          <AlertCircle className="h-4 w-4 text-red-500" />
                          <span className="text-sm text-red-700">
                            {gap.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium mb-2">Knowledge Areas</h4>
                    <div className="flex flex-wrap gap-1">
                      {knowledgeAssessment.knowledge_areas.map((area, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {area.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <Alert>
                <Lightbulb className="h-4 w-4" />
                <AlertDescription>
                  Our AI has identified areas where we need more information to provide you with the best industrial symbiosis opportunities. 
                  We'll ask targeted questions to fill these knowledge gaps and generate personalized material listings.
                </AlertDescription>
              </Alert>
            </>
          )}

          <div className="flex justify-between">
            <Button variant="outline" onClick={onCancel}>
              Cancel
            </Button>
            <Button onClick={handleNext} disabled={isGeneratingQuestions}>
              {isGeneratingQuestions ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating Questions...
                </>
              ) : (
                <>
                  Generate AI Questions
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  if (currentStep === 'loading') {
    return (
      <Card className="w-full max-w-2xl mx-auto">
        <CardContent className="flex flex-col items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600 mb-4" />
          <h3 className="text-lg font-semibold mb-2">AI Assessment in Progress</h3>
          <p className="text-gray-600 text-center">
            Our AI is analyzing your company profile to identify knowledge gaps and generate personalized questions...
          </p>
        </CardContent>
      </Card>
    );
  }

  if (currentStep === 'assessment') {
    return renderAssessmentScreen();
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