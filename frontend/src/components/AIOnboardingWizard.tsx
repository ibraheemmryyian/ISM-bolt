import React, { useState, useEffect } from 'react';
import {
  Home,
  Target,
  Lightbulb,
  Brain,
  AlertCircle,
  Info,
  TrendingUp,
  ArrowRight,
  Loader2
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Alert, AlertDescription } from './ui/alert';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Textarea } from './ui/textarea';
import { Input } from './ui/input';
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
  title: string;
  description: string;
  fields: OnboardingField[];
  isAI: boolean;
  category?: string;
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
  key_insights_expected?: string[];
  material_listings_focus?: string[];
  note?: string;
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
    products?: string;
    main_materials?: string;
    production_volume?: string;
    process_description?: string;
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
  const [answers, setAnswers] = useState<Record<string, any>>({});
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isGeneratingQuestions, setIsGeneratingQuestions] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [industryCategory, setIndustryCategory] = useState<string>('');
  const [processingResult, setProcessingResult] = useState<any>(null);
  const navigate = useNavigate();

  // Initial onboarding steps (basic info)
  const [onboardingSteps, setOnboardingSteps] = useState<OnboardingStep[]>([
    {
      title: 'Basic Company Information',
      description: 'Tell us about your company to get started.',
      isAI: false,
      fields: [
        {
          id: 'name',
          type: 'text',
          label: 'Company Name',
          placeholder: 'Enter your company name',
          required: true,
          value: companyProfile?.name || '',
          importance: 'high'
        },
        {
          id: 'industry',
          type: 'text',
          label: 'Industry',
          placeholder: 'e.g., Chemical Manufacturing, Food Processing, Steel Production...',
          required: true,
          value: companyProfile?.industry || '',
          importance: 'high'
        },
        {
          id: 'location',
          type: 'text',
          label: 'Location',
          placeholder: 'City, Country',
          required: true,
          value: companyProfile?.location || '',
          importance: 'high'
        },
        {
          id: 'employee_count',
          type: 'select',
          label: 'Employee Count',
          placeholder: 'Select range',
          required: true,
          value: companyProfile?.employee_count || '',
          options: [
            '1-10',
            '11-50',
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
          field.id === 'name' ? { ...field, value: companyProfile.name } : field
        )
      })));
    }
    // eslint-disable-next-line
  }, [companyProfile]);

  const startKnowledgeAssessment = async () => {
    try {
      setIsLoading(true);
      setCurrentStep('loading');
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
            products: companyProfile?.products || '',
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
          if (data.knowledge_assessment.confidence_score < 0.5) {
            setCurrentStep('assessment');
          } else {
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
      setCurrentStep('questions');
    } finally {
      setIsLoading(false);
    }
  };

  const generateAIQuestions = async () => {
    try {
      setIsGeneratingQuestions(true);
      const formData = getCurrentFormData();
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
  };

  const handleAnswerChange = (questionId: string, value: any) => {
    setAnswers(prev => ({ ...prev, [questionId]: value }));
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
      const formData = getCurrentFormData();
      const allAnswers = { ...formData, ...answers };
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
                    handleAnswerChange(question.id, newValues);
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
                        {industryCategory.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
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
                            {gap.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
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
                          {area.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
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
      <Card className="w-full max-w-2xl mx-auto mt-10">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
            <span>Loading AI Onboarding...</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Progress value={getProgressPercentage()} className="w-full" />
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
      <div className="max-w-2xl mx-auto mt-10">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="h-6 w-6 text-green-600" />
              <span>AI-Generated Onboarding Questions</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="mb-4">
              <Progress value={getProgressPercentage()} className="w-full" />
              <div className="text-xs text-gray-500 mt-1">
                Question {currentQuestionIndex + 1} of {questionsData.questions.length}
              </div>
            </div>
            <div className={`p-4 rounded-lg ${getImportanceColor(currentQuestion.importance)}`}> 
              <div className="flex items-center space-x-2 mb-2">
                <TrendingUp className="h-4 w-4" />
                <span className="font-semibold">{currentQuestion.category.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}</span>
                <Badge variant="outline" className="ml-2 text-xs">{currentQuestion.importance.toUpperCase()}</Badge>
              </div>
              <div className="text-lg font-medium mb-2">{currentQuestion.question}</div>
              {currentQuestion.reasoning && (
                <div className="text-xs text-gray-600 mb-2">{currentQuestion.reasoning}</div>
              )}
              {renderQuestionInput(currentQuestion)}
              {currentQuestion.follow_up_question && (
                <div className="mt-2 text-xs text-blue-700">
                  <Lightbulb className="inline h-4 w-4 mr-1" />
                  <span>Follow-up: {currentQuestion.follow_up_question}</span>
                </div>
              )}
            </div>
            <div className="flex justify-between">
              <Button variant="outline" onClick={handlePrevious} disabled={currentQuestionIndex === 0}>
                Previous
              </Button>
              <Button onClick={handleNext}>
                {isLastQuestion ? 'Finish' : 'Next'}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (currentStep === 'processing') {
    return (
      <Card className="w-full max-w-2xl mx-auto mt-10">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Loader2 className="h-6 w-6 animate-spin text-green-600" />
            <span>Generating Material Listings...</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Progress value={getProgressPercentage()} className="w-full" />
        </CardContent>
      </Card>
    );
  }

  if (currentStep === 'complete' && processingResult) {
    return (
      <div className="max-w-3xl mx-auto mt-10">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Home className="h-6 w-6 text-blue-600" />
              <span>Onboarding Complete</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="mb-4">
              <Progress value={getProgressPercentage()} className="w-full" />
            </div>
            <div className="text-lg font-semibold text-green-700 mb-2">
              Congratulations! Your onboarding is complete.
            </div>
            <div className="mb-4">
              <div className="font-medium mb-1">AI-Generated Material Listings:</div>
              <ul className="list-disc pl-6">
                {processingResult.material_listings?.map((item: any, idx: number) => (
                  <li key={idx} className="mb-1">
                    <span className="font-semibold">{item.material_name}</span> ({item.type}) - {item.quantity} {item.unit} <span className="text-xs text-gray-500">({item.description})</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="mb-4">
              <div className="font-medium mb-1">Waste Management Requirements:</div>
              <ul className="list-disc pl-6">
                {processingResult.waste_requirements?.map((req: any, idx: number) => (
                  <li key={idx} className="mb-1">
                    <span className="font-semibold">{req.requirement_type.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}</span>: {req.description} <span className="text-xs text-gray-500">(Priority: {req.priority})</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="flex justify-between">
              <Button variant="outline" onClick={onCancel}>
                Close
              </Button>
              <Button onClick={() => navigate('/dashboard')}>
                Go to Dashboard
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-2xl mx-auto mt-10">
        <Alert>
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
        <div className="mt-4 flex justify-center">
          <Button onClick={onCancel}>Back</Button>
        </div>
      </div>
    );
  }

  // Fallback: show basic info form if nothing else
  return (
    <div className="max-w-2xl mx-auto mt-10">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Home className="h-6 w-6 text-blue-600" />
            <span>Company Onboarding</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {onboardingSteps.map((step, stepIdx) => (
            <div key={stepIdx} className="space-y-4">
              <div className="font-semibold text-lg mb-2">{step.title}</div>
              <div className="text-sm text-gray-600 mb-2">{step.description}</div>
              {step.fields.map((field, fieldIdx) => (
                <div key={field.id} className="mb-2">
                  <label className="block text-sm font-medium mb-1">{field.label}{field.required && <span className="text-red-500">*</span>}</label>
                  {field.type === 'text' || field.type === 'number' ? (
                    <Input
                      type={field.type === 'number' ? 'number' : 'text'}
                      value={field.value}
                      placeholder={field.placeholder}
                      onChange={e => handleFieldChange(stepIdx, field.id, e.target.value)}
                      className="w-full"
                    />
                  ) : field.type === 'select' ? (
                    <select
                      value={field.value}
                      onChange={e => handleFieldChange(stepIdx, field.id, e.target.value)}
                      className="w-full border rounded px-2 py-1"
                    >
                      <option value="">Select...</option>
                      {field.options?.map((opt, idx) => (
                        <option key={idx} value={opt}>{opt}</option>
                      ))}
                    </select>
                  ) : field.type === 'textarea' ? (
                    <Textarea
                      value={field.value}
                      placeholder={field.placeholder}
                      onChange={e => handleFieldChange(stepIdx, field.id, e.target.value)}
                      className="w-full min-h-[100px]"
                    />
                  ) : null}
                </div>
              ))}
            </div>
          ))}
          <div className="flex justify-between">
            <Button variant="outline" onClick={onCancel}>Cancel</Button>
            <Button onClick={startKnowledgeAssessment}>Start AI Onboarding</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AIOnboardingWizard; 