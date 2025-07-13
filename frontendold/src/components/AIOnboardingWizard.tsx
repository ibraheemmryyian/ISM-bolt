import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  Bot, 
  User, 
  Send, 
  CheckCircle, 
  ArrowRight,
  Loader2,
  Sparkles,
  Minimize2,
  Maximize2,
  X
} from 'lucide-react';

interface OnboardingStep {
  id: string;
  type: 'question' | 'info' | 'confirmation';
  question: string;
  field: string;
  required: boolean;
  options?: string[];
  placeholder?: string;
  validation?: (value: string) => boolean;
}

interface CompanyData {
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
}

const AIOnboardingWizard: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [companyData, setCompanyData] = useState<CompanyData>({
    name: '',
    industry: '',
    location: '',
    employee_count: 0,
    products: '',
    main_materials: '',
    production_volume: '',
    process_description: ''
  });
  const [aiResponse, setAiResponse] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set());
  const [conversationHistory, setConversationHistory] = useState<Array<{
    type: 'ai' | 'user';
    message: string;
    timestamp: Date;
  }>>([]);
  const [userInput, setUserInput] = useState('');
  const [isMinimized, setIsMinimized] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const onboardingSteps: OnboardingStep[] = [
    {
      id: 'welcome',
      type: 'info',
      question: "Hi! I'm your AI onboarding assistant. I'll help you set up your company profile for industrial symbiosis opportunities. Let's start with the basics - what's your company name?",
      field: 'name',
      required: true,
      placeholder: 'Enter your company name'
    },
    {
      id: 'industry',
      type: 'question',
      question: "Great! What industry does {name} operate in? This helps me find the most relevant opportunities for you.",
      field: 'industry',
      required: true,
      options: [
        'Manufacturing',
        'Textiles & Apparel',
        'Food & Beverage',
        'Chemicals',
        'Construction',
        'Electronics',
        'Automotive',
        'Pharmaceuticals',
        'Energy',
        'Other'
      ]
    },
    {
      id: 'location',
      type: 'question',
      question: "Where is {name} located? This helps me find nearby partners and calculate logistics costs.",
      field: 'location',
      required: true,
      placeholder: 'City, Country'
    },
    {
      id: 'size',
      type: 'question',
      question: "How many employees does {name} have? This helps me understand your scale and potential impact.",
      field: 'employee_count',
      required: true,
      options: [
        '1-10 employees',
        '11-50 employees',
        '51-200 employees',
        '201-500 employees',
        '501-1000 employees',
        '1000+ employees'
      ]
    },
    {
      id: 'products',
      type: 'question',
      question: "What products or services does {name} produce? Be as specific as possible - this helps me identify waste streams and resource needs.",
      field: 'products',
      required: true,
      placeholder: 'Describe your main products or services'
    },
    {
      id: 'materials',
      type: 'question',
      question: "What are the main materials {name} uses in production? This is crucial for finding symbiosis opportunities.",
      field: 'main_materials',
      required: true,
      placeholder: 'List your main raw materials, chemicals, etc.'
    },
    {
      id: 'volume',
      type: 'question',
      question: "What's your typical production volume? This helps me calculate potential savings and impact.",
      field: 'production_volume',
      required: true,
      placeholder: 'e.g., 1000 tons/month, 5000 units/day'
    },
    {
      id: 'process',
      type: 'question',
      question: "Can you describe your main production processes? This helps me identify specific waste streams and optimization opportunities.",
      field: 'process_description',
      required: false,
      placeholder: 'Describe your key manufacturing processes'
    },
    {
      id: 'sustainability',
      type: 'question',
      question: "What are {name}'s sustainability goals? This helps me prioritize opportunities that align with your objectives.",
      field: 'sustainability_goals',
      required: false,
      options: [
        'Reduce waste',
        'Lower carbon footprint',
        'Cut costs',
        'Improve efficiency',
        'Meet regulatory requirements',
        'Enhance brand reputation',
        'Access new markets'
      ]
    }
  ];

  useEffect(() => {
    // Start the conversation
    if (conversationHistory.length === 0) {
      addAIMessage(onboardingSteps[0].question.replace('{name}', 'your company'));
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [conversationHistory]);

  useEffect(() => {
    // Focus input when step changes
    if (inputRef.current && getCurrentStep().type === 'question') {
      inputRef.current.focus();
    }
  }, [currentStep]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const addAIMessage = (message: string) => {
    setConversationHistory(prev => [...prev, {
      type: 'ai',
      message,
      timestamp: new Date()
    }]);
  };

  const addUserMessage = (message: string) => {
    setConversationHistory(prev => [...prev, {
      type: 'user',
      message,
      timestamp: new Date()
    }]);
  };

  const processAIResponse = async (userInput: string) => {
    setIsProcessing(true);
    
    try {
      // Simulate AI processing
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const currentStepData = getCurrentStep();
      const fieldName = currentStepData.field;
      
      // Update company data
      setCompanyData(prev => ({
        ...prev,
        [fieldName]: userInput
      }));
      
      // Mark step as completed
      setCompletedSteps(prev => new Set([...prev, currentStepData.id]));
      
      // Move to next step
      if (currentStep < onboardingSteps.length - 1) {
        setCurrentStep(prev => prev + 1);
        const nextStep = onboardingSteps[currentStep + 1];
        const nextQuestion = nextStep.question.replace('{name}', userInput);
        addAIMessage(nextQuestion);
      } else {
        // Onboarding completed
        setIsCompleted(true);
        addAIMessage("Perfect! I've collected all the information I need. Let me analyze your company profile and generate personalized industrial symbiosis opportunities for you.");
      }
      
    } catch (error) {
      console.error('Error processing AI response:', error);
      addAIMessage("I'm sorry, I encountered an error. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleUserInput = async (input: string) => {
    if (!input.trim() || isProcessing) return;
    
    addUserMessage(input);
    setUserInput('');
    await processAIResponse(input);
  };

  const handleQuickResponse = async (response: string) => {
    await handleUserInput(response);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleUserInput(userInput);
    }
  };

  const getCurrentStep = () => onboardingSteps[currentStep];

  const getProgressPercentage = () => {
    return ((completedSteps.size + (currentStep > 0 ? 1 : 0)) / onboardingSteps.length) * 100;
  };

  const renderQuickResponses = () => {
    const step = getCurrentStep();
    if (!step.options) return null;

    return (
      <div className="flex flex-wrap gap-2 mt-4">
        {step.options.map((option, index) => (
          <Button
            key={index}
            variant="outline"
            onClick={() => handleQuickResponse(option)}
            disabled={isProcessing}
            className="text-sm"
          >
            {option}
          </Button>
        ))}
      </div>
    );
  };

  const renderInputField = () => {
    const step = getCurrentStep();
    
    if (step.type === 'info' || isCompleted) return null;

    return (
      <div className="mt-4">
        {step.type === 'question' && step.options ? (
          renderQuickResponses()
        ) : (
          <div className="flex space-x-2">
            <Input
              ref={inputRef}
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder={step.placeholder || "Type your response..."}
              onKeyPress={handleKeyPress}
              disabled={isProcessing}
              className="flex-1"
            />
            <Button
              onClick={() => handleUserInput(userInput)}
              disabled={isProcessing || !userInput.trim()}
            >
              {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </Button>
          </div>
        )}
      </div>
    );
  };

  // If minimized, show a compact version
  if (isMinimized) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <Card className="w-80 shadow-lg">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center space-x-2 text-sm">
                <Sparkles className="w-4 h-4 text-blue-600" />
                <span>AI Assistant</span>
              </CardTitle>
              <div className="flex space-x-1">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsMinimized(false)}
                  className="p-1 h-6 w-6"
                >
                  <Maximize2 className="w-3 h-3" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsMinimized(false)}
                  className="p-1 h-6 w-6"
                >
                  <X className="w-3 h-3" />
                </Button>
              </div>
            </div>
            <Progress value={getProgressPercentage()} className="h-1" />
          </CardHeader>
          <CardContent className="pt-0">
            <p className="text-xs text-gray-600 mb-2">
              {isCompleted ? 'Onboarding completed!' : `Step ${currentStep + 1} of ${onboardingSteps.length}`}
            </p>
            <Button
              onClick={() => setIsMinimized(false)}
              className="w-full text-xs"
            >
              {isCompleted ? 'View Results' : 'Continue'}
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <Card className="mb-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center space-x-2">
              <Sparkles className="w-6 h-6 text-blue-600" />
              <span>AI Onboarding Assistant</span>
              {isCompleted && <Badge className="bg-green-100 text-green-800">Completed</Badge>}
            </CardTitle>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsMinimized(true)}
                className="flex items-center space-x-1"
              >
                <Minimize2 className="w-4 h-4" />
                <span>Minimize</span>
              </Button>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <Progress value={getProgressPercentage()} className="flex-1" />
            <Badge variant="outline">
              Step {currentStep + 1} of {onboardingSteps.length}
            </Badge>
          </div>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Conversation Area */}
        <div className="lg:col-span-2">
          <Card className="h-[600px] flex flex-col">
            <CardHeader className="border-b">
              <CardTitle className="flex items-center space-x-2">
                <Bot className="w-5 h-5" />
                <span>Conversation</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto p-4 space-y-4">
              {conversationHistory.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-lg ${
                      message.type === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-900'
                    }`}
                  >
                    <div className="flex items-start space-x-2">
                      {message.type === 'ai' && <Bot className="w-4 h-4 mt-1 flex-shrink-0" />}
                      <div>
                        <p className="text-sm">{message.message}</p>
                        <p className="text-xs opacity-70 mt-1">
                          {message.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                      {message.type === 'user' && <User className="w-4 h-4 mt-1 flex-shrink-0" />}
                    </div>
                  </div>
                </div>
              ))}
              {isProcessing && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 p-3 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </CardContent>
            <div className="p-4 border-t">
              {renderInputField()}
            </div>
          </Card>
        </div>

        {/* Progress Sidebar */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {onboardingSteps.map((step, index) => (
                  <div
                    key={step.id}
                    className={`flex items-center space-x-2 p-2 rounded ${
                      index === currentStep
                        ? 'bg-blue-50 border border-blue-200'
                        : completedSteps.has(step.id)
                        ? 'bg-green-50 border border-green-200'
                        : 'bg-gray-50'
                    }`}
                  >
                    {completedSteps.has(step.id) ? (
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : index === currentStep ? (
                      <div className="w-4 h-4 bg-blue-600 rounded-full" />
                    ) : (
                      <div className="w-4 h-4 bg-gray-300 rounded-full" />
                    )}
                    <span className="text-sm font-medium">{step.field.replace('_', ' ')}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Company Info</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                {Object.entries(companyData).map(([key, value]) => {
                  if (!value || key === 'sustainability_goals') return null;
                  return (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-600">{key.replace('_', ' ')}:</span>
                      <span className="font-medium">{value}</span>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          {isCompleted && (
            <Card>
              <CardHeader>
                <CardTitle>Next Steps</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <Button className="w-full" onClick={() => window.location.href = '/dashboard'}>
                    Go to Dashboard
                  </Button>
                  <Button variant="outline" className="w-full" onClick={() => window.location.href = '/marketplace'}>
                    Browse Marketplace
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default AIOnboardingWizard; 