import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  MessageSquare, 
  Bot, 
  User, 
  Loader2, 
  RefreshCw, 
  ThumbsUp, 
  ThumbsDown,
  Brain,
  Search,
  Truck,
  Leaf,
  Calculator,
  Target,
  ArrowRight,
  Sparkles
} from 'lucide-react';

interface ChatMessage {
  id: string;
  message: string;
  response: string;
  intent: string;
  confidence: number;
  suggestions: string[];
  actions: Array<{
    type: string;
    data?: any;
  }>;
  timestamp: string;
  isUser: boolean;
}

interface ChatInterfaceProps {
  userId?: string;
  onAction?: (action: any) => void;
}

export function ChatInterface({ userId, onAction }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Quick action suggestions
  const quickActions = [
    {
      icon: <Search className="h-4 w-4" />,
      text: "I have 5 tons of HDPE waste in Amsterdam",
      intent: "material_search"
    },
    {
      icon: <Target className="h-4 w-4" />,
      text: "Find me compatible materials",
      intent: "matching_request"
    },
    {
      icon: <Truck className="h-4 w-4" />,
      text: "Calculate logistics costs to Rotterdam",
      intent: "logistics_query"
    },
    {
      icon: <Leaf className="h-4 w-4" />,
      text: "What's the carbon impact?",
      intent: "carbon_calculation"
    }
  ];

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load chat history on mount
  useEffect(() => {
    if (userId) {
      loadChatHistory();
    }
  }, [userId]);

  const loadChatHistory = async () => {
    try {
      const response = await fetch(`/api/chat/history/${userId}/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        if (data.history) {
          setMessages(data.history.map((item: any) => ({
            ...item,
            isUser: true
          })));
        }
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  const sendMessage = async (message: string) => {
    if (!message.trim() || !userId) return;

    const userMessage: ChatMessage = {
      id: `user_${Date.now()}`,
      message,
      response: '',
      intent: '',
      confidence: 0,
      suggestions: [],
      actions: [],
      timestamp: new Date().toISOString(),
      isUser: true
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat/send', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message,
          user_id: userId,
          session_id: sessionId
        })
      });

      if (response.ok) {
        const data = await response.json();
        
        const botMessage: ChatMessage = {
          id: `bot_${Date.now()}`,
          message: '',
          response: data.message || 'I apologize, but I couldn\'t process your request.',
          intent: data.intent || 'general_query',
          confidence: data.confidence || 0.5,
          suggestions: data.suggestions || [],
          actions: data.actions || [],
          timestamp: new Date().toISOString(),
          isUser: false
        };

        setMessages(prev => [...prev, botMessage]);
      } else {
        throw new Error('Failed to send message');
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      
      const errorMessage: ChatMessage = {
        id: `error_${Date.now()}`,
        message: '',
        response: 'Sorry, I encountered an error. Please try again.',
        intent: 'error',
        confidence: 0,
        suggestions: ['Try rephrasing your message', 'Check your connection'],
        actions: [],
        timestamp: new Date().toISOString(),
        isUser: false
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickAction = (action: any) => {
    setInputMessage(action.text);
    sendMessage(action.text);
  };

  const handleSuggestion = (suggestion: string) => {
    sendMessage(suggestion);
  };

  const handleAction = (action: any) => {
    if (onAction) {
      onAction(action);
    }
  };

  const handleFeedback = async (messageId: string, rating: number) => {
    try {
      await fetch('/api/chat/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message_id: messageId,
          rating,
          user_id: userId
        })
      });
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    }
  };

  const clearHistory = async () => {
    try {
      await fetch('/api/chat/clear-history', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: userId,
          session_id: sessionId
        })
      });
      setMessages([]);
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  };

  const getIntentIcon = (intent: string) => {
    switch (intent) {
      case 'material_search':
        return <Search className="h-4 w-4 text-blue-500" />;
      case 'matching_request':
        return <Target className="h-4 w-4 text-green-500" />;
      case 'logistics_query':
        return <Truck className="h-4 w-4 text-purple-500" />;
      case 'carbon_calculation':
        return <Leaf className="h-4 w-4 text-green-500" />;
      default:
        return <Brain className="h-4 w-4 text-gray-500" />;
    }
  };

  const getIntentColor = (intent: string) => {
    switch (intent) {
      case 'material_search':
        return 'bg-blue-50 border-blue-200';
      case 'matching_request':
        return 'bg-green-50 border-green-200';
      case 'logistics_query':
        return 'bg-purple-50 border-purple-200';
      case 'carbon_calculation':
        return 'bg-green-50 border-green-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm h-[600px] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <div className="p-2 bg-purple-100 rounded-lg">
            <Bot className="h-5 w-5 text-purple-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">AI Assistant</h3>
            <p className="text-sm text-gray-500">Industrial Symbiosis Expert</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={clearHistory}
            className="p-2 text-gray-400 hover:text-gray-600 transition"
            title="Clear chat history"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && showSuggestions && (
          <div className="text-center py-8">
            <div className="p-3 bg-purple-100 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
              <Sparkles className="h-8 w-8 text-purple-600" />
            </div>
            <h4 className="text-lg font-medium text-gray-900 mb-2">Welcome to AI Assistant!</h4>
            <p className="text-gray-600 mb-6">I can help you with material matching, logistics, and sustainability analysis.</p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {quickActions.map((action, index) => (
                <button
                  key={index}
                  onClick={() => handleQuickAction(action)}
                  className="p-3 text-left border border-gray-200 rounded-lg hover:border-purple-300 hover:bg-purple-50 transition flex items-center space-x-2"
                >
                  {action.icon}
                  <span className="text-sm text-gray-700">{action.text}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] ${message.isUser ? 'order-2' : 'order-1'}`}>
              {message.isUser ? (
                <div className="bg-purple-500 text-white rounded-lg px-4 py-2">
                  <p className="text-sm">{message.message}</p>
                </div>
              ) : (
                <div className={`border rounded-lg px-4 py-3 ${getIntentColor(message.intent)}`}>
                  <div className="flex items-start space-x-2 mb-2">
                    {getIntentIcon(message.intent)}
                    <div className="flex-1">
                      <p className="text-sm text-gray-800 whitespace-pre-wrap">{message.response}</p>
                    </div>
                  </div>
                  
                  {message.confidence > 0 && (
                    <div className="flex items-center justify-between text-xs text-gray-500 mb-2">
                      <span>Confidence: {(message.confidence * 100).toFixed(0)}%</span>
                      <div className="flex items-center space-x-1">
                        <button
                          onClick={() => handleFeedback(message.id, 1)}
                          className="p-1 hover:bg-gray-200 rounded"
                        >
                          <ThumbsUp className="h-3 w-3" />
                        </button>
                        <button
                          onClick={() => handleFeedback(message.id, 0)}
                          className="p-1 hover:bg-gray-200 rounded"
                        >
                          <ThumbsDown className="h-3 w-3" />
                        </button>
                      </div>
                    </div>
                  )}

                  {message.suggestions && message.suggestions.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-xs text-gray-500">Suggestions:</p>
                      <div className="flex flex-wrap gap-1">
                        {message.suggestions.map((suggestion, index) => (
                          <button
                            key={index}
                            onClick={() => handleSuggestion(suggestion)}
                            className="text-xs bg-white border border-gray-300 rounded-full px-2 py-1 hover:bg-gray-50 transition"
                          >
                            {suggestion}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {message.actions && message.actions.length > 0 && (
                    <div className="mt-3 space-y-1">
                      <p className="text-xs text-gray-500">Actions:</p>
                      <div className="flex flex-wrap gap-1">
                        {message.actions.map((action, index) => (
                          <button
                            key={index}
                            onClick={() => handleAction(action)}
                            className="text-xs bg-purple-100 text-purple-700 border border-purple-200 rounded-full px-2 py-1 hover:bg-purple-200 transition flex items-center space-x-1"
                          >
                            <span>{action.type.replace('_', ' ')}</span>
                            <ArrowRight className="h-3 w-3" />
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            
            {!message.isUser && (
              <div className="order-2 ml-2">
                <div className="p-2 bg-purple-100 rounded-full">
                  <Bot className="h-4 w-4 text-purple-600" />
                </div>
              </div>
            )}
            
            {message.isUser && (
              <div className="order-1 mr-2">
                <div className="p-2 bg-gray-100 rounded-full">
                  <User className="h-4 w-4 text-gray-600" />
                </div>
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="border rounded-lg px-4 py-3 bg-gray-50">
              <div className="flex items-center space-x-2">
                <Loader2 className="h-4 w-4 animate-spin text-purple-500" />
                <span className="text-sm text-gray-600">AI is thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(inputMessage);
              }
            }}
            placeholder="Ask me about materials, logistics, or sustainability..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            disabled={isLoading}
          />
          <button
            onClick={() => sendMessage(inputMessage)}
            disabled={!inputMessage.trim() || isLoading}
            className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
        
        <div className="mt-2 text-xs text-gray-500">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  );
} 