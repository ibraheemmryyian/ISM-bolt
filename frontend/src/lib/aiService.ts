import { supabase, withRetry } from './supabase';
import { subscriptionService } from './subscriptionService';
import { toast } from 'react-toastify';

// Enhanced interfaces with better typing
interface AIRecommendation {
  id: string;
  type: 'connection' | 'material' | 'opportunity';
  title: string;
  description: string;
  confidence: number;
  action_url?: string;
  status: string;
  created_at: string;
  error?: string;
}

interface MaterialMatch {
  material_id: string;
  matched_material_id: string;
  match_score: number;
  compatibility_factors: string[];
  error?: string;
}

export interface AIResponse {
  success: boolean;
  data?: any;
  error?: string;
  message?: string;
}

// Production-ready error handling and logging
class ProductionLogger {
  private static instance: ProductionLogger;
  
  static getInstance(): ProductionLogger {
    if (!ProductionLogger.instance) {
      ProductionLogger.instance = new ProductionLogger();
    }
    return ProductionLogger.instance;
  }

  log(level: 'info' | 'warn' | 'error', message: string, data?: any) {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      message,
      data,
      userAgent: navigator.userAgent,
      url: window.location.href
    };

    // Console logging for development
    if (process.env.NODE_ENV === 'development') {
      console.log(`[${level.toUpperCase()}] ${message}`, data || '');
    }

    // Send to backend logging service in production
    if (process.env.NODE_ENV === 'production') {
      this.sendToLoggingService(logEntry);
    }
  }

  private async sendToLoggingService(logEntry: any) {
    try {
      await fetch('http://localhost:5001/api/logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(logEntry)
      });
    } catch (error: any) {
      // Fallback to console if logging service fails
      console.error('Failed to send log to backend:', error);
    }
  }
}

// Enhanced API client with retry logic and error handling
class APIClient {
  private static instance: APIClient;
  private logger = ProductionLogger.getInstance();
  private retryAttempts = 3;
  private retryDelay = 1000;

  static getInstance(): APIClient {
    if (!APIClient.instance) {
      APIClient.instance = new APIClient();
    }
    return APIClient.instance;
  }

  async request<T>(
    endpoint: string, 
    options: RequestInit = {}, 
    retryCount = 0
  ): Promise<AIResponse> {
    const url = `http://localhost:5001/api${endpoint}`;
    
    try {
      this.logger.log('info', `API Request: ${options.method || 'GET'} ${url}`, {
        body: options.body,
        headers: options.headers
      });

      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      this.logger.log('info', `API Response: ${url}`, { status: response.status, data });

      return {
        success: true,
        data: data.data || data,
        message: data.message
      };

    } catch (error: any) {
      this.logger.log('error', `API Error: ${url}`, { error: error.message, retryCount });

      // Retry logic for network errors
      if (retryCount < this.retryAttempts && this.isRetryableError(error)) {
        await this.delay(this.retryDelay * Math.pow(2, retryCount));
        return this.request<T>(endpoint, options, retryCount + 1);
      }

      return {
        success: false,
        error: this.getUserFriendlyError(error),
        message: 'Service temporarily unavailable. Please try again.'
      };
    }
  }

  private isRetryableError(error: any): boolean {
    return error.message.includes('Network') || 
           error.message.includes('fetch') ||
           error.message.includes('500') ||
           error.message.includes('502') ||
           error.message.includes('503');
  }

  private getUserFriendlyError(error: any): string {
    if (error.message.includes('Network')) {
      return 'Network connection failed. Please check your internet connection.';
    }
    if (error.message.includes('500')) {
      return 'Server error. Our team has been notified.';
    }
    if (error.message.includes('404')) {
      return 'Service not found. Please contact support.';
    }
    return 'An unexpected error occurred. Please try again.';
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Production-ready AI Service
class AIService {
  private apiClient = APIClient.getInstance();
  private logger = ProductionLogger.getInstance();
  private baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:5001';

  private async makeRequest(endpoint: string, options: RequestInit = {}): Promise<AIResponse> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await withRetry(async () => {
        const res = await fetch(url, defaultOptions);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        return res;
      });

      const data = await response.json();
      return { success: true, data };
    } catch (error: any) {
      console.error(`AI Service Error (${endpoint}):`, error);
      return {
        success: false,
        error: error.message || 'An unexpected error occurred',
        message: this.getUserFriendlyMessage(error)
      };
    }
  }

  private getUserFriendlyMessage(error: any): string {
    if (error.message?.includes('Failed to fetch')) {
      return 'Unable to connect to the server. Please check your internet connection.';
    }
    if (error.message?.includes('401')) {
      return 'Your session has expired. Please log in again.';
    }
    if (error.message?.includes('403')) {
      return 'You do not have permission to perform this action.';
    }
    if (error.message?.includes('429')) {
      return 'Too many requests. Please wait a moment and try again.';
    }
    if (error.message?.includes('500')) {
      return 'Server error. Please try again later.';
    }
    return 'Something went wrong. Please try again.';
  }

  async generateListings(materialData: any): Promise<AIResponse> {
    return this.makeRequest('/api/ai/generate-listings', {
      method: 'POST',
      body: JSON.stringify(materialData),
    });
  }

  async analyzeMatch(buyerData: any, sellerData: any): Promise<AIResponse> {
    return this.makeRequest('/api/ai/analyze-match', {
      method: 'POST',
      body: JSON.stringify({ buyerData, sellerData }),
    });
  }

  async getRecommendations(userId: string): Promise<AIResponse> {
    return this.makeRequest(`/api/ai/recommendations/${userId}`);
  }

  async processOnboarding(data: any): Promise<AIResponse> {
    return this.makeRequest('/api/ai/onboarding', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async generateComprehensivePortfolio(companyData: any): Promise<AIResponse> {
    return this.makeRequest('/api/ai/portfolio/generate', {
      method: 'POST',
      body: JSON.stringify(companyData)
    });
  }

  async analyzeCompanyProfile(companyData: any): Promise<AIResponse> {
    return this.makeRequest('/api/ai/company/analyze', {
      method: 'POST',
      body: JSON.stringify(companyData)
    });
  }

  async getSymbiosisOpportunities(companyId: string): Promise<AIResponse> {
    return this.makeRequest(`/api/ai/symbiosis/${companyId}`);
  }

  async forecastImpact(transactionData: any): Promise<AIResponse> {
    return this.makeRequest('/api/ai/forecast-impact', {
      method: 'POST',
      body: JSON.stringify(transactionData),
    });
  }

  async getCostAnalysis(materialId: string): Promise<AIResponse> {
    return this.makeRequest(`/api/ai/cost-analysis/${materialId}`);
  }

  async validateMaterial(materialData: any): Promise<AIResponse> {
    return this.makeRequest('/api/ai/validate-material', {
      method: 'POST',
      body: JSON.stringify(materialData),
    });
  }

  async getMarketInsights(): Promise<AIResponse> {
    return this.makeRequest('/api/ai/market-insights');
  }

  async optimizeLogistics(routeData: any): Promise<AIResponse> {
    return this.makeRequest('/api/ai/optimize-logistics', {
      method: 'POST',
      body: JSON.stringify(routeData),
    });
  }

  async generateRecommendations(userId: string, userProfile: any): Promise<AIRecommendation[]> {
    try {
      // Check subscription access
      const hasAccess = await subscriptionService.checkFeatureAccess(userId, 'aiRecommendations');
      if (!hasAccess) {
        toast.error('AI recommendations require a Pro subscription. Upgrade to unlock this feature.');
        return [];
      }

      this.logger.log('info', 'Generating AI recommendations', { userId, profileType: userProfile.organization_type });

      // Call backend AI service
      const response = await this.apiClient.request<AIRecommendation[]>('/ai-recommendations', {
        method: 'POST',
        body: JSON.stringify({ userId, userProfile })
      });

      if (!response.success) {
        toast.error(response.error || 'Failed to generate recommendations');
        return [];
      }

      // Store recommendations in database
      if (response.data && response.data.length > 0) {
        await this.storeRecommendations(userId, response.data);
      }

      toast.success(`Generated ${response.data?.length || 0} AI recommendations`);
      return response.data || [];

    } catch (error: any) {
      this.logger.log('error', 'Error generating AI recommendations', { error: error.message, userId });
      toast.error('Failed to generate AI recommendations. Please try again.');
      return [];
    }
  }

  async findMaterialMatches(materialId: string): Promise<MaterialMatch[]> {
    try {
      this.logger.log('info', 'Finding material matches', { materialId });

      const response = await this.apiClient.request<MaterialMatch[]>('/ai-material-matches', {
        method: 'POST',
        body: JSON.stringify({ materialId })
      });

      if (!response.success) {
        toast.error(response.error || 'Failed to find material matches');
        return [];
      }

      return response.data || [];

    } catch (error: any) {
      this.logger.log('error', 'Error finding material matches', { error: error.message, materialId });
      toast.error('Failed to find material matches. Please try again.');
      return [];
    }
  }

  async analyzeSymbiosisNetwork(participants: any[]): Promise<any> {
    try {
      this.logger.log('info', 'Analyzing symbiosis network', { participantCount: participants.length });

      const response = await this.apiClient.request('/symbiosis-network', {
        method: 'POST',
        body: JSON.stringify({ participants })
      });

      if (!response.success) {
        toast.error(response.error || 'Failed to analyze symbiosis network');
        return null;
      }

      return response.data;

    } catch (error: any) {
      this.logger.log('error', 'Error analyzing symbiosis network', { error: error.message });
      toast.error('Failed to analyze symbiosis network. Please try again.');
      return null;
    }
  }

  async getAIExplanation(matchData: any): Promise<string> {
    try {
      const response = await this.apiClient.request<{ explanation: string }>('/explain-match', {
        method: 'POST',
        body: JSON.stringify(matchData)
      });

      if (!response.success) {
        return 'Unable to generate AI explanation at this time.';
      }

      return response.data?.explanation || 'No explanation available.';

    } catch (error: any) {
      this.logger.log('error', 'Error getting AI explanation', { error: error.message });
      return 'Unable to generate AI explanation at this time.';
    }
  }

  async runGnnAnalysis(participants: any[], modelType: string = 'gcn'): Promise<any> {
    try {
      this.logger.log('info', 'Running GNN analysis', { participantCount: participants.length, modelType });

      const response = await this.apiClient.request('/ai-gnn-links', {
        method: 'POST',
        body: JSON.stringify({ participants, modelType })
      });

      if (!response.success) {
        toast.error(response.error || 'Failed to run GNN analysis');
        return null;
      }

      return response.data;

    } catch (error: any) {
      this.logger.log('error', 'Error running GNN analysis', { error: error.message });
      toast.error('Failed to run GNN analysis. Please try again.');
      return null;
    }
  }

  async getProactiveOpportunities(userId: string): Promise<any[]> {
    try {
      const response = await this.apiClient.request<any[]>('/proactive-opportunities', {
        method: 'POST',
        body: JSON.stringify({ userId })
      });

      if (!response.success) {
        this.logger.log('warn', 'Failed to get proactive opportunities', { error: response.error });
        return [];
      }

      return response.data || [];

    } catch (error: any) {
      this.logger.log('error', 'Error getting proactive opportunities', { error: error.message });
      return [];
    }
  }

  async checkCompliance(match: any): Promise<any> {
    try {
      const response = await this.apiClient.request('/compliance-check', {
        method: 'POST',
        body: JSON.stringify({ match })
      });

      if (!response.success) {
        this.logger.log('warn', 'Failed to check compliance', { error: response.error });
        return { compliant: false, issues: ['Unable to verify compliance'] };
      }

      return response.data;

    } catch (error: any) {
      this.logger.log('error', 'Error checking compliance', { error: error.message });
      return { compliant: false, issues: ['Compliance check failed'] };
    }
  }

  private async storeRecommendations(userId: string, recommendations: AIRecommendation[]): Promise<void> {
    try {
      const { error } = await supabase
        .from('ai_recommendations')
        .insert(
          recommendations.map(rec => ({
            company_id: userId,
            type: rec.type,
            title: rec.title,
            description: rec.description,
            confidence: rec.confidence,
            action_url: rec.action_url,
            status: 'pending'
          }))
        );

      if (error) {
        this.logger.log('error', 'Failed to store recommendations', { error: error.message });
      }
    } catch (error: any) {
      this.logger.log('error', 'Error storing recommendations', { error: error.message });
    }
  }
}

// Export singleton instance
export const aiService = new AIService();

// Legacy function exports for backward compatibility
export async function fetchProactiveOpportunities(): Promise<any[]> {
  return aiService.getProactiveOpportunities('system');
}

export async function fetchMultiHopSymbiosisPaths(): Promise<any[]> {
  try {
    const response = await APIClient.getInstance().request<any[]>('/multi-hop-symbiosis');
    return response.success ? (response.data || []) : [];
  } catch (error: any) {
    ProductionLogger.getInstance().log('error', 'Error fetching multi-hop symbiosis paths', { error: error.message });
    return [];
  }
}

export async function checkCompliance(match: any): Promise<any> {
  return aiService.checkCompliance(match);
}