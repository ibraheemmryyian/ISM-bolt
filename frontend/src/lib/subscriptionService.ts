import { supabase } from './supabase';

export interface Subscription {
  id: string;
  company_id: string;
  tier: 'free' | 'pro' | 'enterprise';
  status: 'active' | 'expired' | 'suspended';
  created_at: string;
  expires_at?: string;
}

export interface FeatureAccess {
  aiRecommendations: boolean;
  aiMatching: boolean;
  advancedAnalytics: boolean;
  bulkOperations: boolean;
  prioritySupport: boolean;
  customIntegrations: boolean;
  unlimitedListings: boolean;
  exportData: boolean;
}

class SubscriptionService {
  private subscriptionCache: Map<string, Subscription | null> = new Map();

  async getUserSubscription(userId: string): Promise<Subscription | null> {
    // Check cache first
    if (this.subscriptionCache.has(userId)) {
      return this.subscriptionCache.get(userId) || null;
    }

    try {
      const { data, error } = await supabase
        .from('subscriptions')
        .select('*')
        .eq('company_id', userId)
        .eq('status', 'active')
        .single();

      if (error && error.code !== 'PGRST116') {
        console.error('Error fetching subscription:', error);
      }

      const subscription = data || null;
      this.subscriptionCache.set(userId, subscription);
      return subscription;
    } catch (error) {
      console.error('Error fetching subscription:', error);
      return null;
    }
  }

  async checkFeatureAccess(userId: string, feature: keyof FeatureAccess): Promise<boolean> {
    const subscription = await this.getUserSubscription(userId);
    
    if (!subscription) {
      // Free tier - limited features
      const freeFeatures = this.getFreeTierFeatures();
      return freeFeatures[feature];
    }

    switch (subscription.tier) {
      case 'enterprise':
        return true; // All features
      case 'pro':
        const proFeatures = this.getProTierFeatures();
        return proFeatures.includes(feature);
      case 'free':
      default:
        const freeFeatures = this.getFreeTierFeatures();
        return freeFeatures[feature];
    }
  }

  async getFeatureAccess(userId: string): Promise<FeatureAccess> {
    const subscription = await this.getUserSubscription(userId);
    
    if (!subscription) {
      return this.getFreeTierFeatures();
    }

    switch (subscription.tier) {
      case 'enterprise':
        return {
          aiRecommendations: true,
          aiMatching: true,
          advancedAnalytics: true,
          bulkOperations: true,
          prioritySupport: true,
          customIntegrations: true,
          unlimitedListings: true,
          exportData: true,
        };
      case 'pro':
        return {
          aiRecommendations: true,
          aiMatching: true,
          advancedAnalytics: true,
          bulkOperations: false,
          prioritySupport: false,
          customIntegrations: false,
          unlimitedListings: true,
          exportData: true,
        };
      case 'free':
      default:
        return this.getFreeTierFeatures();
    }
  }

  private getFreeTierFeatures(): FeatureAccess {
    return {
      aiRecommendations: false,
      aiMatching: false,
      advancedAnalytics: false,
      bulkOperations: false,
      prioritySupport: false,
      customIntegrations: false,
      unlimitedListings: false,
      exportData: false,
    };
  }

  private getProTierFeatures(): (keyof FeatureAccess)[] {
    return [
      'aiRecommendations',
      'aiMatching', 
      'advancedAnalytics',
      'unlimitedListings',
      'exportData'
    ];
  }

  async requireFeature(userId: string, feature: keyof FeatureAccess): Promise<boolean> {
    const hasAccess = await this.checkFeatureAccess(userId, feature);
    
    if (!hasAccess) {
      throw new Error(`Feature '${feature}' requires a paid subscription. Please upgrade to access this feature.`);
    }
    
    return true;
  }

  clearCache(userId?: string) {
    if (userId) {
      this.subscriptionCache.delete(userId);
    } else {
      this.subscriptionCache.clear();
    }
  }

  // Subscription tier information
  getSubscriptionTiers() {
    return {
      free: {
        name: 'Free',
        price: '$0/month',
        features: [
          'Basic marketplace access',
          'Up to 5 material listings',
          'Basic search and browse',
          'Standard support'
        ],
        limitations: [
          'No AI recommendations',
          'No AI matching',
          'No advanced analytics',
          'No data export'
        ]
      },
      pro: {
        name: 'Pro',
        price: '$199/month',
        features: [
          'Everything in Free',
          'AI-powered recommendations',
          'Advanced AI matching',
          'Unlimited material listings',
          'Advanced analytics dashboard',
          'Data export capabilities',
          'Priority support',
          'Custom material categories',
          'Advanced search filters'
        ]
      },
      enterprise: {
        name: 'Enterprise',
        price: '$999/month',
        features: [
          'Everything in Pro',
          'Bulk operations',
          'Custom integrations',
          'Dedicated account manager',
          'Advanced reporting & insights',
          'White-label options',
          'API access',
          'Custom AI model training',
          'Multi-location management',
          'Compliance reporting',
          '24/7 phone support'
        ]
      }
    };
  }
}

export const subscriptionService = new SubscriptionService(); 