import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Building2, 
  Factory, 
  MapPin, 
  Package, 
  Scale, 
  FileText, 
  Upload, 
  ArrowRight, 
  CheckCircle,
  Brain,
  Loader2,
  MessageSquare,
  Send,
  Bot
} from 'lucide-react';

interface OnboardingWizardProps {
  onComplete: () => void;
}

interface OnboardingField {
  label: string;
  type: 'text' | 'textarea' | 'number' | 'checkbox' | 'info';
  key: string;
  required: boolean;
}

interface OnboardingStep {
  step: number;
  title: string;
  fields: OnboardingField[];
}

interface ChatMessage {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  intent?: string;
  entities?: any[];
}

export function OnboardingWizard({ onComplete }: OnboardingWizardProps) {
  const [currentStep, setCurrentStep] = useState(1);
  const [onboardingFlow, setOnboardingFlow] = useState<OnboardingStep[]>([]);
  const [formData, setFormData] = useState<{[key: string]: any}>({});
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<{[key: string]: string}>({});
  const [isLoadingFlow, setIsLoadingFlow] = useState(true);
  const [validationResults, setValidationResults] = useState<any>(null);
  const [qualityScore, setQualityScore] = useState<any>(null);
  const [autoSuggestions, setAutoSuggestions] = useState<{[key: string]: any[]}>({});
  const [showConversationalMode, setShowConversationalMode] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'ai',
      content: "Hello! I'm your AI onboarding assistant. I can help you set up your company profile through natural conversation. You can tell me about your company, industry, processes, and materials, and I'll guide you through the onboarding process. Would you like to start with conversational onboarding, or would you prefer the traditional form-based approach?",
      timestamp: new Date()
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const navigate = useNavigate();

  // Load dynamic onboarding flow
  useEffect(() => {
    loadOnboardingFlow();
  }, []);

  async function loadOnboardingFlow() {
    try {
      // For now, we'll generate a basic flow - in production this would come from backend
      const basicCompanyData = {
        id: 'temp',
        name: '',
        industry: '',
        location: '',
        employee_count: null
      };
      
      const response = await fetch('/api/ai-infer-listings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
    companyName: '',
    industry: '',
    products: '',
    location: '',
    productionVolume: '',
    mainMaterials: '',
          processDescription: ''
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.onboarding_flow) {
          setOnboardingFlow(data.onboarding_flow);
        }
      }
    } catch (error) {
      console.error('Failed to load onboarding flow:', error);
      // Fallback to basic flow
      setOnboardingFlow([
        {
          step: 1,
          title: "Company Overview",
          fields: [
            { label: "Company Name", "type": "text", key: "name", required: true },
            { label: "Industry", "type": "text", key: "industry", required: true },
            { label: "Location", "type": "text", key: "location", required: true },
            { label: "Company Size", "type": "number", key: "employee_count", required: false }
          ]
        },
        {
          step: 2,
          title: "Production Details",
          fields: [
            { label: "Describe your main production process", "type": "textarea", key: "processes", required: true },
            { label: "What are your main input materials?", "type": "text", key: "materials", required: true },
            { label: "Production Volume", "type": "text", key: "productionVolume", required: true }
          ]
        },
        {
          step: 3,
          title: "Review & Submit",
          fields: [
            { label: "Review all information and submit onboarding.", "type": "info", key: "review", required: false }
          ]
        }
      ]);
    } finally {
      setIsLoadingFlow(false);
    }
  }

  async function loadAIQuestions(companyData: any) {
    try {
      const response = await fetch('/api/ai-questions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ companyData })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.questions && data.questions.length > 0) {
          // Convert AI questions to onboarding flow format
          const aiQuestionsStep = {
            step: onboardingFlow.length + 1,
            title: "AI-Generated Questions",
            fields: data.questions.map((q: any) => ({
              label: q.question,
              type: q.type,
              key: q.key,
              required: q.required,
              reasoning: q.reasoning
            }))
          };
          
          setOnboardingFlow([...onboardingFlow, aiQuestionsStep]);
        }
      }
    } catch (error) {
      console.error('Failed to load AI questions:', error);
    }
  }

  // Load AI questions when company data changes
  useEffect(() => {
    if (formData.industry && formData.location && !isLoadingFlow) {
      const companyData = {
        industry: formData.industry,
        location: formData.location,
        employee_count: formData.employee_count,
        processes: formData.processes,
        materials: formData.materials ? [formData.materials] : []
      };
      loadAIQuestions(companyData);
    }
  }, [formData.industry, formData.location, formData.employee_count, formData.processes, formData.materials]);

  // Conversational AI functions
  async function sendChatMessage(message: string) {
    if (!message.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: message,
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setIsChatLoading(true);

    try {
      const response = await fetch('/api/conversational-onboarding', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: message,
          conversation_history: chatMessages.map(msg => ({
            role: msg.type === 'user' ? 'user' : 'assistant',
            content: msg.content
          })),
          current_form_data: formData
        })
      });

      if (response.ok) {
        const data = await response.json();
        
        const aiMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'ai',
          content: data.response,
          timestamp: new Date(),
          intent: data.intent,
          entities: data.entities
        };

        setChatMessages(prev => [...prev, aiMessage]);

        // Update form data if AI extracted information
        if (data.extracted_data) {
          setFormData(prev => ({ ...prev, ...data.extracted_data }));
        }

        // Generate dynamic questions if AI suggests
        if (data.suggested_questions && data.suggested_questions.length > 0) {
          const questionsStep: OnboardingStep = {
            step: onboardingFlow.length + 1,
            title: "AI-Generated Questions",
            fields: data.suggested_questions.map((q: any) => ({
              label: q.question,
              type: (q.type || 'text') as 'text' | 'textarea' | 'number' | 'checkbox' | 'info',
              key: q.key,
              required: q.required || false,
              reasoning: q.reasoning
            }))
          };
          
          setOnboardingFlow(prev => [...prev, questionsStep]);
        }

        // Show contextual recommendations if available
        if (data.contextual_recommendations) {
          const recommendationsStep: OnboardingStep = {
            step: onboardingFlow.length + 1,
            title: "AI Recommendations",
            fields: [{
              label: data.contextual_recommendations.summary,
              type: 'info' as const,
              key: 'recommendations',
              required: false
            }]
          };
          
          setOnboardingFlow(prev => [...prev, recommendationsStep]);
        }
      } else {
        throw new Error('Failed to get AI response');
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: "I apologize, but I'm having trouble processing your message right now. Please try again or switch to the form-based approach.",
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false);
    }
  }

  function handleChatSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (chatInput.trim() && !isChatLoading) {
      sendChatMessage(chatInput);
    }
  }

  function switchToConversationalMode() {
    setShowConversationalMode(true);
  }

  function switchToFormMode() {
    setShowConversationalMode(false);
  }

  function handleFieldChange(key: string, value: any) {
    setFormData({ ...formData, [key]: value });
    
    // Clear error when user starts typing
    if (errors[key]) {
      setErrors({ ...errors, [key]: '' });
    }
    
    // Trigger real-time validation after a short delay
    setTimeout(() => {
      validateDataRealTime();
    }, 500);
  }

  async function validateDataRealTime() {
    try {
      const response = await fetch('/api/validate-company-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ companyData: formData })
      });

      if (response.ok) {
        const data = await response.json();
        setValidationResults(data.validation);
        setQualityScore(data.quality_score);
        
        // Process auto-completion suggestions
        const suggestions: {[key: string]: any[]} = {};
        if (data.validation.auto_completed) {
          data.validation.auto_completed.forEach((item: any) => {
            suggestions[item.field] = suggestions[item.field] || [];
            suggestions[item.field].push(item);
          });
        }
        setAutoSuggestions(suggestions);
      }
    } catch (error) {
      console.error('Real-time validation error:', error);
    }
  }

  function applyAutoSuggestion(field: string, suggestion: any) {
    if (suggestion.suggestion) {
      if (Array.isArray(suggestion.suggestion)) {
        setFormData({ ...formData, [field]: suggestion.suggestion.join(', ') });
      } else {
        setFormData({ ...formData, [field]: suggestion.suggestion });
      }
      validateDataRealTime();
    }
  }

  function validateCurrentStep() {
    const currentStepData = onboardingFlow.find(step => step.step === currentStep);
    if (!currentStepData) return true;

    const newErrors: {[key: string]: string} = {};
    
    currentStepData.fields.forEach(field => {
      if (field.required && (!formData[field.key] || formData[field.key].toString().trim() === '')) {
        newErrors[field.key] = `${field.label} is required`;
      }
    });
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  async function handleNext() {
    if (currentStep < onboardingFlow.length) {
      if (validateCurrentStep()) {
        setCurrentStep(currentStep + 1);
      }
    } else {
      if (validateCurrentStep()) {
        await handleSubmit();
      }
    }
  }

  async function handleSubmit() {
      setLoading(true);
      try {
      const response = await fetch('/api/ai-infer-listings', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
        body: JSON.stringify({
          companyName: formData.name || '',
          industry: formData.industry || '',
          products: formData.products || '',
          location: formData.location || '',
          productionVolume: formData.productionVolume || '',
          mainMaterials: formData.materials || '',
          processDescription: formData.processes || '',
          employee_count: formData.employee_count || null
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `AI inference failed (${response.status})`);
      }

      const aiListingsResponse = await response.json();
        setLoading(false);
        navigate('/review-ai-listings', { state: aiListingsResponse });
      } catch (err: any) {
        console.error('AI onboarding error:', err);
        setLoading(false);
      alert(`AI Onboarding Error: ${err.message}`);
  }
  }

  function handleBack() {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  }

  function renderField(field: OnboardingField & { reasoning?: string }) {
    const value = formData[field.key] || '';
    const error = errors[field.key];
    const fieldSuggestions = autoSuggestions[field.key] || [];

    const fieldElement = (() => {
      switch (field.type) {
        case 'textarea':
          return (
            <div key={field.key} className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                {field.label}
                {field.required && <span className="text-red-500 ml-1">*</span>}
              </label>
              <textarea
                value={value}
                onChange={(e) => handleFieldChange(field.key, e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500 ${
                  error ? 'border-red-500' : 'border-gray-300'
                }`}
                rows={4}
              />
              {error && <p className="text-red-500 text-sm">{error}</p>}
            </div>
          );

        case 'number':
          return (
            <div key={field.key} className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                {field.label}
                {field.required && <span className="text-red-500 ml-1">*</span>}
              </label>
              <input
                type="number"
                value={value}
                onChange={(e) => handleFieldChange(field.key, e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500 ${
                  error ? 'border-red-500' : 'border-gray-300'
                }`}
              />
              {error && <p className="text-red-500 text-sm">{error}</p>}
            </div>
          );

        case 'checkbox':
          return (
            <div key={field.key} className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={value || false}
                onChange={(e) => handleFieldChange(field.key, e.target.checked)}
                className="h-4 w-4 text-emerald-600 focus:ring-emerald-500 border-gray-300 rounded"
              />
              <label className="text-sm font-medium text-gray-700">
                {field.label}
                {field.required && <span className="text-red-500 ml-1">*</span>}
              </label>
              {error && <p className="text-red-500 text-sm">{error}</p>}
            </div>
          );

        case 'info':
          return (
            <div key={field.key} className="bg-blue-50 border border-blue-200 rounded-md p-4">
              <p className="text-blue-800">{field.label}</p>
            </div>
          );

        default: // text
          return (
            <div key={field.key} className="space-y-2">
              <label className="block text-sm font-medium text-gray-700">
                {field.label}
                {field.required && <span className="text-red-500 ml-1">*</span>}
              </label>
              <input
                type="text"
                value={value}
                onChange={(e) => handleFieldChange(field.key, e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500 ${
                  error ? 'border-red-500' : 'border-gray-300'
                }`}
              />
              {error && <p className="text-red-500 text-sm">{error}</p>}
            </div>
          );
      }
    })();

    // Add auto-suggestions if available
    const suggestionElements = fieldSuggestions.map((suggestion, index) => (
      <div key={index} className="bg-yellow-50 border border-yellow-200 rounded-md p-3">
        <p className="text-sm text-yellow-800 mb-2">
          <span className="font-medium">Suggestion:</span> {suggestion.reason}
        </p>
        <button
          onClick={() => applyAutoSuggestion(field.key, suggestion)}
          className="text-sm bg-yellow-100 text-yellow-800 px-3 py-1 rounded hover:bg-yellow-200 transition"
        >
          Apply: {Array.isArray(suggestion.suggestion) ? suggestion.suggestion.join(', ') : suggestion.suggestion}
        </button>
      </div>
    ));

    // Add reasoning if available
    const reasoningElement = field.reasoning ? (
      <div className="bg-emerald-50 border border-emerald-200 rounded-md p-3">
        <p className="text-sm text-emerald-800">
          <span className="font-medium">Why we ask this:</span> {field.reasoning}
        </p>
      </div>
    ) : null;

    return (
      <div key={field.key} className="space-y-3">
        {fieldElement}
        {suggestionElements}
        {reasoningElement}
      </div>
    );
  }

  function renderValidationPanel() {
    if (!validationResults && !qualityScore) return null;

    return (
      <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Quality & Validation</h3>
        
        {/* Quality Score */}
        {qualityScore && (
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Data Quality Score:</span>
              <span className={`text-lg font-bold ${
                qualityScore.grade === 'A' ? 'text-green-600' :
                qualityScore.grade === 'B' ? 'text-blue-600' :
                qualityScore.grade === 'C' ? 'text-yellow-600' :
                qualityScore.grade === 'D' ? 'text-orange-600' : 'text-red-600'
              }`}>
                {qualityScore.grade} ({qualityScore.percentage.toFixed(1)}%)
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${
                  qualityScore.grade === 'A' ? 'bg-green-500' :
                  qualityScore.grade === 'B' ? 'bg-blue-500' :
                  qualityScore.grade === 'C' ? 'bg-yellow-500' :
                  qualityScore.grade === 'D' ? 'bg-orange-500' : 'bg-red-500'
                }`}
                style={{ width: `${qualityScore.percentage}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Validation Results */}
        {validationResults && (
          <div className="space-y-4">
            {/* Errors */}
            {validationResults.errors.length > 0 && (
              <div className="bg-red-50 border border-red-200 rounded-md p-4">
                <h4 className="text-sm font-medium text-red-800 mb-2">Required Fixes:</h4>
                <ul className="text-sm text-red-700 space-y-1">
                  {validationResults.errors.map((error: string, index: number) => (
                    <li key={index}>• {error}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Warnings */}
            {validationResults.warnings.length > 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                <h4 className="text-sm font-medium text-yellow-800 mb-2">Warnings:</h4>
                <ul className="text-sm text-yellow-700 space-y-1">
                  {validationResults.warnings.map((warning: string, index: number) => (
                    <li key={index}>• {warning}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Suggestions */}
            {validationResults.suggestions.length > 0 && (
              <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                <h4 className="text-sm font-medium text-blue-800 mb-2">Improvement Suggestions:</h4>
                <ul className="text-sm text-blue-700 space-y-1">
                  {validationResults.suggestions.map((suggestion: string, index: number) => (
                    <li key={index}>• {suggestion}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Quality Improvements */}
            {qualityScore && qualityScore.improvements.length > 0 && (
              <div className="bg-purple-50 border border-purple-200 rounded-md p-4">
                <h4 className="text-sm font-medium text-purple-800 mb-2">To improve your score:</h4>
                <ul className="text-sm text-purple-700 space-y-1">
                  {qualityScore.improvements.map((improvement: string, index: number) => (
                    <li key={index}>• {improvement}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  if (isLoadingFlow) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-emerald-500 mx-auto mb-4" />
          <p className="text-gray-600">Loading onboarding flow...</p>
        </div>
      </div>
    );
  }

  const currentStepData = onboardingFlow.find(step => step.step === currentStep);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-8">
      <div className="max-w-4xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-between mb-4">
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center space-x-2 text-emerald-600 hover:text-emerald-700 transition"
            >
              <ArrowRight className="h-4 w-4 rotate-180" />
              <span>Back to Dashboard</span>
            </button>
            <div></div>
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome to SymbioFlows</h1>
          <p className="text-gray-600">Let's get your company set up with AI-powered symbiosis matching</p>
          
          {/* Mode Selection */}
          <div className="flex justify-center mt-6 space-x-4">
            <button
              onClick={switchToConversationalMode}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg transition ${
                showConversationalMode
                  ? 'bg-emerald-500 text-white shadow-lg'
                  : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
              }`}
            >
              <Bot className="h-5 w-5" />
              <span>AI Chat Assistant</span>
            </button>
            <button
              onClick={switchToFormMode}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg transition ${
                !showConversationalMode
                  ? 'bg-emerald-500 text-white shadow-lg'
                  : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
              }`}
            >
              <FileText className="h-5 w-5" />
              <span>Traditional Form</span>
            </button>
          </div>
        </div>

        {/* Conversational Mode */}
        {showConversationalMode ? (
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center space-x-2 mb-6">
              <Bot className="h-6 w-6 text-emerald-500" />
              <h2 className="text-xl font-bold text-gray-900">AI Onboarding Assistant</h2>
            </div>
            
            {/* Chat Messages */}
            <div className="h-96 overflow-y-auto mb-4 space-y-4 p-4 bg-gray-50 rounded-lg">
              {chatMessages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                      message.type === 'user'
                        ? 'bg-emerald-500 text-white'
                        : 'bg-white border border-gray-200 text-gray-800'
                    }`}
                  >
                    <p className="text-sm">{message.content}</p>
                    {message.intent && (
                      <p className="text-xs opacity-70 mt-1">
                        Intent: {message.intent}
                      </p>
                    )}
                  </div>
                </div>
              ))}
              {isChatLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-gray-200 text-gray-800 px-4 py-2 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            {/* Chat Input */}
            <form onSubmit={handleChatSubmit} className="flex space-x-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Tell me about your company, industry, processes, or materials..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500"
                disabled={isChatLoading}
              />
              <button
                type="submit"
                disabled={!chatInput.trim() || isChatLoading}
                className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="h-4 w-4" />
              </button>
            </form>
            
            {/* Quick Actions */}
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                onClick={() => sendChatMessage("Tell me about your company name and industry")}
                className="text-xs bg-gray-100 text-gray-700 px-3 py-1 rounded-full hover:bg-gray-200 transition"
              >
                Company Info
              </button>
              <button
                onClick={() => sendChatMessage("What materials do you use or produce?")}
                className="text-xs bg-gray-100 text-gray-700 px-3 py-1 rounded-full hover:bg-gray-200 transition"
              >
                Materials
              </button>
              <button
                onClick={() => sendChatMessage("Describe your production processes")}
                className="text-xs bg-gray-100 text-gray-700 px-3 py-1 rounded-full hover:bg-gray-200 transition"
              >
                Processes
              </button>
              <button
                onClick={() => sendChatMessage("Where is your company located?")}
                className="text-xs bg-gray-100 text-gray-700 px-3 py-1 rounded-full hover:bg-gray-200 transition"
              >
                Location
              </button>
            </div>
            
            {/* Form Data Preview */}
            {Object.keys(formData).length > 0 && (
              <div className="mt-6 p-4 bg-emerald-50 border border-emerald-200 rounded-lg">
                <h3 className="text-sm font-medium text-emerald-800 mb-2">Extracted Information:</h3>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(formData).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-emerald-700">{key}:</span>
                      <span className="text-emerald-800 font-medium">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <>
            {/* Progress Steps */}
            <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
              <div className="flex items-center justify-between mb-6">
                {onboardingFlow.map((step, index) => (
                  <div key={index} className="flex items-center">
                    <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 ${
                      currentStep > step.step ? 'bg-emerald-500 border-emerald-500 text-white' :
                      currentStep === step.step ? 'border-emerald-500 text-emerald-500' :
                      'border-gray-300 text-gray-400'
                    }`}>
                      {currentStep > step.step ? (
                        <CheckCircle className="h-5 w-5" />
                      ) : (
                        <Building2 className="h-5 w-5" />
                      )}
                    </div>
                    <div className="ml-3">
                      <p className={`text-sm font-medium ${
                        currentStep >= step.step ? 'text-gray-900' : 'text-gray-500'
                      }`}>
                        {step.title}
                      </p>
                    </div>
                    {index < onboardingFlow.length - 1 && (
                      <div className={`w-16 h-0.5 mx-4 ${
                        currentStep > step.step ? 'bg-emerald-500' : 'bg-gray-300'
                      }`} />
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Form Content */}
            <div className="bg-white rounded-xl shadow-sm p-8">
              {currentStepData && (
                <div className="space-y-6">
                  <div className="text-center mb-6">
                    <Building2 className="h-12 w-12 text-emerald-500 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">{currentStepData.title}</h2>
                    <p className="text-gray-600">Step {currentStep} of {onboardingFlow.length}</p>
                  </div>
                  
                  {/* Validation Panel */}
                  {renderValidationPanel()}
                  
                <div className="space-y-6">
                    {currentStepData.fields.map(field => renderField(field))}
                  </div>
                  
                  {/* Navigation */}
                  <div className="flex justify-between pt-6">
                <button
                  onClick={handleBack}
                      disabled={currentStep === 1}
                      className={`px-6 py-2 rounded-md transition ${
                        currentStep === 1
                          ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                          : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      }`}
                    >
                      Back
                </button>
                
            <button
              onClick={handleNext}
              disabled={loading}
                      className="px-6 py-2 bg-emerald-600 text-white rounded-md hover:bg-emerald-700 transition flex items-center space-x-2"
            >
                  {loading ? (
                    <>
                          <Loader2 className="h-4 w-4 animate-spin" />
                          <span>Processing...</span>
                    </>
                      ) : currentStep === onboardingFlow.length ? (
                        <span>Submit</span>
                  ) : (
                        <span>Next</span>
                  )}
            </button>
                  </div>
              </div>
            )}
            </div>
          </>
        )}
      </div>
    </div>
  );
} 