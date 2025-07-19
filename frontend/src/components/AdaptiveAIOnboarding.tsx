import React, { useState, useEffect, useRef } from 'react';
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
  Shield,
  TrendingUp,
  MessageCircle,
  Clock,
  AlertCircle
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Alert, AlertDescription } from './ui/alert';
import { supabase } from '../lib/supabase';

interface OnboardingQuestion {
  id: string;
  question: string;
  type: string;
  category: string;
  importance: string;
  options?: string[];
  reasoning?: string;
  compliance_related: boolean;
}

interface OnboardingSession {
  session_id: string;
  initial_questions: OnboardingQuestion[];
  completion_percentage: number;
}

interface UserResponse {
  question_id: string;
  answer: any;
  response_time: number;
}

interface AdaptiveAIOnboardingProps {
  onClose: () => void;
  onComplete: (analysis: any) => void;
}

export function AdaptiveAIOnboarding({ onClose, onComplete }: AdaptiveAIOnboardingProps) {
  const [session, setSession] = useState<OnboardingSession | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [responses, setResponses] = useState<UserResponse[]>([]);
  const [currentAnswer, setCurrentAnswer] = useState<any>('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCompleting, setIsCompleting] = useState(false);
  const [complianceStatus, setComplianceStatus] = useState<any>(null);
  const [aiInsights, setAiInsights] = useState<any>(null);
  const [showReasoning, setShowReasoning] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  
  const navigate = useNavigate();

  // Pre-defined initial questions for instant loading
  const initialQuestions: OnboardingQuestion[] = [
    {
      id: "basic_0",
      question: "What is your company name?",
      type: "text",
      category: "basic_info",
      importance: "high",
      reasoning: "Essential for identification and compliance",
      compliance_related: false
    },
    {
      id: "basic_1", 
      question: "What industry are you in?",
      type: "text",
      category: "basic_info",
      importance: "high",
      reasoning: "Determines relevant symbiosis opportunities and compliance requirements",
      compliance_related: false
    },
    {
      id: "basic_2",
      question: "Where is your company located?",
      type: "text", 
      category: "basic_info",
      importance: "high",
      reasoning: "Critical for logistics optimization and local compliance",
      compliance_related: false
    },
    {
      id: "basic_3",
      question: "How many employees do you have?",
      type: "select",
      category: "basic_info", 
      importance: "medium",
      reasoning: "Helps assess company scale and resource needs",
      options: ["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"],
      compliance_related: false
    }
  ];

  useEffect(() => {
    // Show initial form immediately
    setSession({
      session_id: 'temp_session',
      initial_questions: initialQuestions,
      completion_percentage: 0
    });
    setIsInitializing(false);
    
    // Start AI session in background
    startOnboardingSession();
  }, []);

  useEffect(() => {
    // Reset state when question changes
    setCurrentAnswer('');
    setShowReasoning(false);
  }, [currentQuestionIndex]);

  const startOnboardingSession = async () => {
    try {
      // Don't show loading state - run in background
      setError(null);

      // Get the authenticated user first
      const { data: { user }, error: authError } = await supabase.auth.getUser();
      if (authError || !user) {
        console.warn('Authentication issue, continuing with local session');
        return;
      }

      // Get the session token
      const { data: { session }, error: sessionError } = await supabase.auth.getSession();
      if (sessionError || !session) {
        console.warn('Session issue, continuing with local session');
        return;
      }

      const response = await fetch('/api/adaptive-onboarding/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${session.access_token}`
        },
        body: JSON.stringify({
          user_id: user.id,
          initial_profile: {}
        })
      });

      if (!response.ok) {
        console.warn(`Server responded with ${response.status}, continuing with local session`);
        return;
      }

      const data = await response.json();
      
      if (data.success) {
        // Update session with real data from server
        setSession({
          session_id: data.session.session_id,
          initial_questions: data.session.initial_questions || initialQuestions,
          completion_percentage: data.session.completion_percentage || 0
        });
      }
      
    } catch (err: any) {
      console.warn('Background session start failed, continuing with local session:', err.message);
      // Don't show error to user - continue with local session
    }
  };

  const handleAnswerChange = (value: any) => {
    setCurrentAnswer(value);
  };

  const handleNext = async () => {
    if (!session || !currentAnswer) return;

    try {
      setIsLoading(true);
      
      const currentQuestion = session.initial_questions[currentQuestionIndex];
      
      // Add response to local state
      const newResponse: UserResponse = {
        question_id: currentQuestion.id,
        answer: currentAnswer,
        response_time: 0 // No longer tracking response time
      };
      
      setResponses(prev => [...prev, newResponse]);

      // Get the session token
      const { data: { session: authSession }, error: sessionError } = await supabase.auth.getSession();
      if (sessionError || !authSession) {
        setError('Authentication required. Please log in.');
        return;
      }

      // Send response to backend
      const response = await fetch('/api/adaptive-onboarding/respond', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authSession.access_token}`
        },
        body: JSON.stringify({
          session_id: session.session_id,
          question_id: currentQuestion.id,
          answer: currentAnswer
        })
      });

      if (!response.ok) {
        throw new Error('Failed to process response');
      }

      const result = await response.json();
      
      // Update compliance status and AI insights
      if (result.compliance_status) {
        setComplianceStatus(result.compliance_status);
      }
      
      if (result.ai_insights) {
        setAiInsights(result.ai_insights);
      }

      // Check if we should continue or complete
      if (result.next_actions?.completion_ready) {
        await completeOnboarding();
      } else {
        // Move to next question or wait for new questions
        if (currentQuestionIndex < session.initial_questions.length - 1) {
          setCurrentQuestionIndex(prev => prev + 1);
        } else {
          // Wait for AI to generate more questions
          await waitForNextQuestions();
        }
      }
      
    } catch (err: any) {
      setError(err.message || 'Failed to process response');
      console.error('Error processing response:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const waitForNextQuestions = async () => {
    try {
      // Get the session token
      const { data: { session: authSession }, error: sessionError } = await supabase.auth.getSession();
      if (sessionError || !authSession) {
        setError('Authentication required. Please log in.');
        return;
      }

      // Poll for new questions
      const checkStatus = async () => {
        const response = await fetch(`/api/adaptive-onboarding/status/${session?.session_id}`, {
          headers: {
            'Authorization': `Bearer ${authSession.access_token}`
          }
        });
        if (response.ok) {
          const status = await response.json();
          if (status.next_questions && status.next_questions.length > 0) {
            // Update session with new questions
            setSession(prev => prev ? {
              ...prev,
              initial_questions: [...prev.initial_questions, ...status.next_questions]
            } : null);
            setCurrentQuestionIndex(prev => prev + 1);
            return true;
          }
        }
        return false;
      };

      // Poll every 2 seconds for up to 30 seconds
      for (let i = 0; i < 15; i++) {
        const hasNewQuestions = await checkStatus();
        if (hasNewQuestions) return;
        await new Promise(resolve => setTimeout(resolve, 2000));
      }

      // If no new questions after 30 seconds, complete onboarding
      await completeOnboarding();
      
    } catch (err: any) {
      console.error('Error waiting for next questions:', err);
      await completeOnboarding();
    }
  };

  const completeOnboarding = async () => {
    try {
      setIsCompleting(true);
      setError(null);

      // Get the session token
      const { data: { session: authSession }, error: sessionError } = await supabase.auth.getSession();
      if (sessionError || !authSession) {
        setError('Authentication required. Please log in.');
        return;
      }

      const response = await fetch('/api/adaptive-onboarding/complete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authSession.access_token}`
        },
        body: JSON.stringify({
          session_id: session?.session_id
        })
      });

      if (!response.ok) {
        throw new Error('Failed to complete onboarding');
      }

      const result = await response.json();
      
      // Call onComplete with the analysis
      onComplete(result.analysis);
      
    } catch (err: any) {
      setError(err.message || 'Failed to complete onboarding');
      console.error('Error completing onboarding:', err);
    } finally {
      setIsCompleting(false);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prev => prev - 1);
    }
  };

  const getCurrentQuestion = () => {
    if (!session || currentQuestionIndex >= session.initial_questions.length) {
      return null;
    }
    return session.initial_questions[currentQuestionIndex];
  };

  const getProgressPercentage = () => {
    if (!session) return 0;
    return session.completion_percentage * 100;
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
    switch (question.type) {
      case 'text':
        return (
          <Input
            value={currentAnswer}
            onChange={(e) => handleAnswerChange(e.target.value)}
            placeholder="Type your answer here..."
            className="w-full"
          />
        );
      
      case 'textarea':
        return (
          <Textarea
            value={currentAnswer}
            onChange={(e) => handleAnswerChange(e.target.value)}
            placeholder="Provide detailed information..."
            className="w-full min-h-[100px]"
          />
        );
      
      case 'select':
        return (
          <select
            value={currentAnswer}
            onChange={(e) => handleAnswerChange(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md"
          >
            <option value="">Select an option...</option>
            {question.options?.map((option, index) => (
              <option key={index} value={option}>{option}</option>
            ))}
          </select>
        );
      
      case 'multiselect':
        return (
          <div className="space-y-2">
            {question.options?.map((option, index) => (
              <label key={index} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={Array.isArray(currentAnswer) && currentAnswer.includes(option)}
                  onChange={(e) => {
                    const newValue = Array.isArray(currentAnswer) ? [...currentAnswer] : [];
                    if (e.target.checked) {
                      newValue.push(option);
                    } else {
                      const index = newValue.indexOf(option);
                      if (index > -1) {
                        newValue.splice(index, 1);
                      }
                    }
                    handleAnswerChange(newValue);
                  }}
                  className="rounded"
                />
                <span>{option}</span>
              </label>
            ))}
          </div>
        );
      
      case 'boolean':
        return (
          <div className="flex space-x-4">
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                name="boolean"
                value="true"
                checked={currentAnswer === true}
                onChange={() => handleAnswerChange(true)}
                className="rounded"
              />
              <span>Yes</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                name="boolean"
                value="false"
                checked={currentAnswer === false}
                onChange={() => handleAnswerChange(false)}
                className="rounded"
              />
              <span>No</span>
            </label>
          </div>
        );
      
      default:
        return (
          <Input
            value={currentAnswer}
            onChange={(e) => handleAnswerChange(e.target.value)}
            placeholder="Type your answer here..."
            className="w-full"
          />
        );
    }
  };

  const handleClose = () => {
    // Navigate to dashboard instead of home
    navigate('/dashboard');
    onClose();
  };

  if (isInitializing) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <Card className="w-full max-w-md">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
              <div>
                <h3 className="text-lg font-semibold">Loading Onboarding Form</h3>
                <p className="text-sm text-gray-600">Preparing your personalized experience...</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isLoading && !session) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <Card className="w-full max-w-md">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
              <div>
                <h3 className="text-lg font-semibold">Processing Response</h3>
                <p className="text-sm text-gray-600">AI is analyzing your answer...</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isCompleting) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <Card className="w-full max-w-md">
          <CardContent className="p-6">
            <div className="flex items-center space-x-3">
              <Loader2 className="h-6 w-6 animate-spin text-green-600" />
              <div>
                <h3 className="text-lg font-semibold">Completing Onboarding</h3>
                <p className="text-sm text-gray-600">Generating your comprehensive analysis...</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const currentQuestion = getCurrentQuestion();

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <CardHeader className="border-b">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Brain className="h-6 w-6 text-purple-600" />
              </div>
              <div>
                <CardTitle className="text-xl">Adaptive AI Onboarding</CardTitle>
                <p className="text-sm text-gray-600">
                  Question {currentQuestionIndex + 1} of {session?.initial_questions.length || 0}
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClose}
              className="text-gray-500 hover:text-gray-700"
            >
              ✕
            </Button>
          </div>
          
          {/* Progress Bar */}
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
              <span>Progress</span>
              <span>{Math.round(getProgressPercentage())}%</span>
            </div>
            <Progress value={getProgressPercentage()} className="h-2" />
          </div>
        </CardHeader>

        <CardContent className="p-6">
          {error && (
            <Alert className="mb-4 border-red-200 bg-red-50">
              <AlertCircle className="h-4 w-4 text-red-600" />
              <AlertDescription className="text-red-800">{error}</AlertDescription>
            </Alert>
          )}

          {currentQuestion && (
            <div className="space-y-6">
              {/* Question Header */}
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    {currentQuestion.question}
                  </h3>
                  
                  <div className="flex items-center space-x-2 mb-3">
                    <Badge className={getImportanceColor(currentQuestion.importance)}>
                      {currentQuestion.importance} priority
                    </Badge>
                    {currentQuestion.compliance_related && (
                      <Badge className="bg-blue-100 text-blue-800">
                        <Shield className="h-3 w-3 mr-1" />
                        Compliance
                      </Badge>
                    )}
                    <Badge className="bg-purple-100 text-purple-800">
                      <Brain className="h-3 w-3 mr-1" />
                      AI-Powered
                    </Badge>
                  </div>
                </div>
                
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowReasoning(!showReasoning)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <Lightbulb className="h-4 w-4 mr-1" />
                  Why?
                </Button>
              </div>

              {/* Reasoning */}
              {showReasoning && currentQuestion.reasoning && (
                <Alert className="border-blue-200 bg-blue-50">
                  <Lightbulb className="h-4 w-4 text-blue-600" />
                  <AlertDescription className="text-blue-800">
                    {currentQuestion.reasoning}
                  </AlertDescription>
                </Alert>
              )}

              {/* Question Input */}
              <div className="space-y-4">
                {renderQuestionInput(currentQuestion)}
              </div>

              {/* Compliance Status */}
              {complianceStatus && (
                <Alert className="border-green-200 bg-green-50">
                  <Shield className="h-4 w-4 text-green-600" />
                  <AlertDescription className="text-green-800">
                    <strong>Compliance Status:</strong> {complianceStatus.current_status}
                    {complianceStatus.recommendations?.length > 0 && (
                      <ul className="mt-2 list-disc list-inside">
                        {complianceStatus.recommendations.map((rec: string, index: number) => (
                          <li key={index}>{rec}</li>
                        ))}
                      </ul>
                    )}
                  </AlertDescription>
                </Alert>
              )}

              {/* AI Insights */}
              {aiInsights && (
                <Alert className="border-purple-200 bg-purple-50">
                  <Brain className="h-4 w-4 text-purple-600" />
                  <AlertDescription className="text-purple-800">
                    <strong>AI Insights:</strong> {aiInsights.key_findings?.join(', ')}
                  </AlertDescription>
                </Alert>
              )}

              {/* Navigation */}
              <div className="flex items-center justify-between pt-4 border-t">
                <Button
                  variant="outline"
                  onClick={handlePrevious}
                  disabled={currentQuestionIndex === 0 || isLoading}
                  className="flex items-center space-x-2"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Previous
                </Button>

                <Button
                  onClick={handleNext}
                  disabled={!currentAnswer || isLoading}
                  className="flex items-center space-x-2 bg-purple-600 hover:bg-purple-700"
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <ArrowRight className="h-4 w-4" />
                  )}
                  {currentQuestionIndex === (session?.initial_questions.length || 0) - 1 ? 'Complete' : 'Next'}
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
} 