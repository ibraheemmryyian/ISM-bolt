import { supabase } from './supabase';

export interface UserActivity {
  id: string;
  company_id: string;
  activity_type: string;
  title: string;
  description?: string;
  impact_level: 'high' | 'medium' | 'low';
  metadata?: any;
  created_at: string;
}

class ActivityService {
  async logActivity(
    companyId: string,
    activityType: string,
    title: string,
    description?: string,
    impactLevel: 'high' | 'medium' | 'low' = 'medium',
    metadata?: any
  ): Promise<string | null> {
    try {
      const { data, error } = await supabase
        .rpc('log_user_activity', {
          p_company_id: companyId,
          p_activity_type: activityType,
          p_title: title,
          p_description: description,
          p_impact_level: impactLevel,
          p_metadata: metadata || {}
        });

      if (error) {
        console.error('Error logging activity:', error);
        return null;
      }

      return data;
    } catch (error) {
      console.error('Error logging activity:', error);
      return null;
    }
  }

  async getUserActivities(companyId: string, limit: number = 10): Promise<UserActivity[]> {
    try {
      const { data, error } = await supabase
        .from('user_activities')
        .select('*')
        .eq('company_id', companyId)
        .order('created_at', { ascending: false })
        .limit(limit);

      if (error) {
        console.error('Error fetching user activities:', error);
        return [];
      }

      return data || [];
    } catch (error) {
      console.error('Error fetching user activities:', error);
      return [];
    }
  }

  async getRecentActivities(companyId: string, days: number = 7): Promise<UserActivity[]> {
    try {
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - days);

      const { data, error } = await supabase
        .from('user_activities')
        .select('*')
        .eq('company_id', companyId)
        .gte('created_at', cutoffDate.toISOString())
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Error fetching recent activities:', error);
        return [];
      }

      return data || [];
    } catch (error) {
      console.error('Error fetching recent activities:', error);
      return [];
    }
  }

  // Convenience methods for common activities
  async logOnboardingCompleted(companyId: string, companyName: string): Promise<string | null> {
    return this.logActivity(
      companyId,
      'onboarding_completed',
      'Completed AI Onboarding',
      `Successfully completed AI onboarding for ${companyName}`,
      'high'
    );
  }

  async logMaterialListed(companyId: string, materialName: string, quantity: string): Promise<string | null> {
    return this.logActivity(
      companyId,
      'material_listed',
      'Material Listed',
      `Listed ${quantity} of ${materialName}`,
      'medium',
      { material_name: materialName, quantity }
    );
  }

  async logMatchFound(companyId: string, matchScore: number, partnerCompany: string): Promise<string | null> {
    return this.logActivity(
      companyId,
      'match_found',
      'AI Match Found',
      `Found ${matchScore}% match with ${partnerCompany}`,
      'high',
      { match_score: matchScore, partner_company: partnerCompany }
    );
  }

  async logConnectionRequested(companyId: string, targetCompany: string): Promise<string | null> {
    return this.logActivity(
      companyId,
      'connection_requested',
      'Connection Requested',
      `Requested connection with ${targetCompany}`,
      'medium',
      { target_company: targetCompany }
    );
  }

  async logSubscriptionUpgraded(companyId: string, newTier: string): Promise<string | null> {
    return this.logActivity(
      companyId,
      'subscription_upgraded',
      'Subscription Upgraded',
      `Upgraded to ${newTier} subscription`,
      'high',
      { new_tier: newTier }
    );
  }

  async logProfileUpdated(companyId: string, fieldUpdated: string): Promise<string | null> {
    return this.logActivity(
      companyId,
      'profile_updated',
      'Profile Updated',
      `Updated ${fieldUpdated} in company profile`,
      'low',
      { field_updated: fieldUpdated }
    );
  }

  // Format activity for display
  formatActivityForDisplay(activity: UserActivity): {
    action: string;
    date: string;
    impact: string;
  } {
    return {
      action: activity.title,
      date: new Date(activity.created_at).toLocaleDateString(),
      impact: activity.impact_level === 'high' ? 'High Impact' : 
              activity.impact_level === 'medium' ? 'Medium Impact' : 'Low Impact'
    };
  }
}

export const activityService = new ActivityService(); 