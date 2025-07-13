import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Bell, 
  Brain, 
  Building2, 
  Calendar, 
  Crown, 
  Factory, 
  Globe, 
  Home, 
  MessageSquare, 
  Plus, 
  Recycle, 
  Settings, 
  ShoppingCart, 
  Star, 
  TrendingUp, 
  Users, 
  Workflow,
  Zap,
  Eye,
  CheckCircle,
  AlertCircle,
  Clock,
  X,
  Info
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { aiService } from '../lib/aiService';
import { subscriptionService } from '../lib/subscriptionService';
import { useNavigate } from 'react-router-dom';
import { SubscriptionUpgradeModal } from './SubscriptionUpgradeModal';
import GnnSymbiosisPanel from './GnnSymbiosisPanel';
import ProactiveOpportunitiesPanel from './ProactiveOpportunitiesPanel';
import MultiHopSymbiosisPanel from './MultiHopSymbiosisPanel';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import GlobalImpactPanel from './GlobalImpactPanel';
import { ToastContainer } from 'react-toastify';
import { MaterialForm } from './MaterialForm';

interface DashboardProps {
  onSignOut: () => void;
}

interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: string;
  level: number;
  xp: number;
  streak_days: number;
  subscription?: {
    tier: string;
    status: string;
    expires_at?: string;
  };
  company_profile?: {
    role: string;
    location: string;
    organization_type: string;
    materials_of_interest: string;
    sustainability_goals: string;
  };
}

interface AIRecommendation {
  id: string;
  type: 'connection' | 'material' | 'opportunity';
  title: string;
  description: string;
  confidence: number;
  action_url?: string;
  status: string;
  created_at: string;
}

interface Activity {
  id: string;
  type: 'match' | 'message' | 'listing' | 'connection';
  title: string;
  description: string;
  timestamp: string;
  status: 'pending' | 'completed' | 'active';
}

interface MaterialMatch {
  id: string;
  material_name: string;
  company_name: string;
  match_score: number;
  distance: string;
  type: 'waste' | 'requirement';
  quantity: number;
  unit: string;
  description: string;
}

export function Dashboard({ onSignOut }: DashboardProps) {
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [aiRecommendations, setAiRecommendations] = useState<AIRecommendation[]>([]);
  const [recentActivity, setRecentActivity] = useState<Activity[]>([]);
  const [materialMatches, setMaterialMatches] = useState<MaterialMatch[]>([]);
  const [stats, setStats] = useState({
    connections: 0,
    materials_listed: 0,
    matches_found: 0,
    sustainability_score: 0,
    xp: 0,
    streak_days: 0
  });
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);
  const [messages, setMessages] = useState<any[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [recipientId, setRecipientId] = useState('');
  const [aiInput, setAiInput] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [ndaAccepted, setNdaAccepted] = useState(false);
  const [onboardingComplete, setOnboardingComplete] = useState(false);
  const [subscription, setSubscription] = useState<any>(null);
  const [featureAccess, setFeatureAccess] = useState<any>(null);
  const [upgradeFeature, setUpgradeFeature] = useState('');
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [showAccountSettings, setShowAccountSettings] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showChats, setShowChats] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showMarketplace, setShowMarketplace] = useState(false);
  const [proactiveOpportunities, setProactiveOpportunities] = useState<any[]>([]);
  const [showMaterialForm, setShowMaterialForm] = useState(false);
  const [showSustainabilityInfo, setShowSustainabilityInfo] = useState(false);
  const [accountSettings, setAccountSettings] = useState({
    name: '',
    email: '',
    companyName: '',
    location: '',
    industry: '',
    organizationType: '',
    profilePicture: '',
    notificationEmail: true,
    notificationSMS: false,
  });
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [profilePictureFile, setProfilePictureFile] = useState<File | null>(null);
  const [profilePicturePreview, setProfilePicturePreview] = useState<string | null>(null);
  const [notifications, setNotifications] = useState<any[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const navigate = useNavigate();
  const [aiDebug, setAiDebug] = useState<{
    lastRun: string;
    matchCount: number;
    recentMatches: any[];
    loading: boolean;
    error: string;
  }>({
    lastRun: '',
    matchCount: 0,
    recentMatches: [],
    loading: false,
    error: ''
  });
  const [explainMatchId, setExplainMatchId] = useState<string | null>(null);
  const [explainMatchText, setExplainMatchText] = useState('');
  const [explainLoading, setExplainLoading] = useState(false);

  useEffect(() => {
    loadDashboardData();
  }, []);

  useEffect(() => {
    if (userProfile) {
      loadMessages();
      loadSubscriptionData();
      setAccountSettings({
        name: userProfile.name || '',
        email: userProfile.email || '',
        companyName: userProfile.company_profile?.role || '',
        location: userProfile.company_profile?.location || '',
        industry: userProfile.company_profile?.organization_type || '',
        organizationType: userProfile.company_profile?.organization_type || '',
        profilePicture: '',
        notificationEmail: true,
        notificationSMS: false,
      });
    }
  }, [userProfile]);

  useEffect(() => {
    async function checkOnboardingStatus() {
      if (userProfile) {
        // Check if user has completed onboarding by looking for company profile or materials
        const { data: companyProfile } = await supabase
          .from('company_profiles')
          .select('*')
          .eq('company_id', userProfile.id)
          .single();
        
        const { data: materials } = await supabase
          .from('materials')
          .select('id')
          .eq('company_id', userProfile.id)
          .limit(1);
        
        setOnboardingComplete(Boolean(companyProfile || (materials && materials.length > 0)));
      }
    }
    
    checkOnboardingStatus();
  }, [userProfile]);

  // Secret admin access - only you know this
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      // Secret admin access: Ctrl + Alt + A
      if (event.ctrlKey && event.altKey && event.key === 'a') {
        localStorage.setItem('temp-admin-access', 'true');
        window.location.href = '/admin';
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, []);

  useEffect(() => {
    if (!userProfile) return;
    const { id } = userProfile;
    async function fetchActivityStats() {
      // XP: 10 per material listed, 20 per connection, 30 per transaction
      const { count: materialsCount } = await supabase
        .from('materials')
        .select('*', { count: 'exact', head: true })
        .eq('company_id', id);
      const { count: connectionsCount } = await supabase
        .from('connections')
        .select('*', { count: 'exact', head: true })
        .or(`requester_id.eq.${id},recipient_id.eq.${id}`)
        .eq('status', 'accepted');
      const { count: transactionsCount } = await supabase
        .from('transactions')
        .select('*', { count: 'exact', head: true })
        .or(`buyer_id.eq.${id},seller_id.eq.${id}`);
      // Streak: count consecutive days with activity (material, connection, or transaction)
      const { data: activityData } = await supabase
        .from('activity_log')
        .select('created_at')
        .eq('company_id', id)
        .order('created_at', { ascending: false });
      let streak = 0;
      if (activityData && activityData.length > 0) {
        let lastDate = new Date(activityData[0].created_at).setHours(0,0,0,0);
        streak = 1;
        for (let i = 1; i < activityData.length; i++) {
          const thisDate = new Date(activityData[i].created_at).setHours(0,0,0,0);
          if (lastDate - thisDate === 86400000) {
            streak++;
            lastDate = thisDate;
          } else if (lastDate !== thisDate) {
            break;
          }
        }
      }
      // Sustainability score: based on number of successful matches and total transaction amount
      const { count: matchesCount } = await supabase
        .from('material_matches')
        .select('*', { count: 'exact', head: true })
        .eq('status', 'completed');
      const { data: transactions } = await supabase
        .from('transactions')
        .select('amount')
        .or(`buyer_id.eq.${id},seller_id.eq.${id}`);
      const totalAmount = transactions ? transactions.reduce((sum, t) => sum + (t.amount || 0), 0) : 0;
      setStats({
        connections: connectionsCount || 0,
        materials_listed: materialsCount || 0,
        matches_found: matchesCount || 0,
        sustainability_score: Math.min(50 + (matchesCount || 0) * 5 + totalAmount * 0.01, 100),
        xp: (materialsCount || 0) * 10 + (connectionsCount || 0) * 20 + (transactionsCount || 0) * 30,
        streak_days: streak
      });
    }
    fetchActivityStats();
  }, [userProfile]);

  async function loadDashboardData() {
    try {
      setLoadError(null);
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('Not authenticated');
      // Load user profile with subscription
      const { data: company } = await supabase
        .from('companies')
        .select('*')
        .eq('id', user.id)
        .single();
      if (!company) throw new Error('Company not found');

      const { data: profile } = await supabase
        .from('company_profiles')
        .select('*')
        .eq('company_id', user.id)
        .single();

      const { data: subscription } = await supabase
        .from('subscriptions')
        .select('tier, status, expires_at')
        .eq('company_id', user.id)
        .single();

      if (company) {
        setUserProfile({
          id: company.id,
          name: company.name,
          email: company.email,
          role: company.role,
          level: company.level || 1,
          xp: company.xp || 0,
          streak_days: company.streak_days || 0,
          subscription: subscription || { tier: 'free', status: 'active' },
          company_profile: profile
        });
      }

      // Load real statistics
      await loadUserStats(user.id, company?.xp || 0, company?.streak_days || 0);
      
      // Load AI recommendations (only for paid users)
      if (subscription?.tier !== 'free') {
        await loadAIRecommendations(user.id);
        await loadMaterialMatches(user.id);
      }
      
      // Load recent activity
      await loadRecentActivity(user.id);

      // Load proactive opportunities
      await loadProactiveOpportunities(user.id);

    } catch (error) {
      console.error('Error loading dashboard data:', error);
      setLoadError((error as any).message || 'Failed to load dashboard data.');
    } finally {
      setLoading(false);
    }
  }

  async function loadUserStats(userId: string, xp: number, streak_days: number) {
    try {
      // Get real user materials count
      const { data: materials } = await supabase
        .from('materials')
        .select('id, type, quantity')
        .eq('company_id', userId);
      
      const materialsCount = materials?.length || 0;
      
      // Get real connections count
      const { data: connections } = await supabase
        .from('connections')
        .select('id')
        .or(`requester_id.eq.${userId},recipient_id.eq.${userId}`)
        .eq('status', 'accepted');
      
      const connectionsCount = connections?.length || 0;
      
      // Get real AI matches count
      const { data: userMaterials } = await supabase
        .from('materials')
        .select('id')
        .eq('company_id', userId);
      
      let aiMatchesCount = 0;
      if (userMaterials && userMaterials.length > 0) {
        const materialIds = userMaterials.map(m => m.id);
        const { data: matches } = await supabase
          .from('material_matches')
          .select('id')
          .or(`material_id.in.(${materialIds.join(',')}),matched_material_id.in.(${materialIds.join(',')})`);
        
        aiMatchesCount = matches?.length || 0;
      }
      
      // Calculate real sustainability score based on multiple factors
      let sustainabilityScore = 0;
      
      // Base score for being active (20 points)
      if (materialsCount > 0) sustainabilityScore += 20;
      
      // Materials diversity (up to 25 points)
      if (materials && materials.length > 0) {
        const wasteMaterials = materials.filter(m => m.type === 'waste');
        const requirementMaterials = materials.filter(m => m.type === 'requirement');
        
        // Bonus for having both waste and requirements
        if (wasteMaterials.length > 0 && requirementMaterials.length > 0) {
          sustainabilityScore += 15;
        }
        
        // Bonus for material quantity (up to 10 points)
        const totalQuantity = materials.reduce((sum, m) => sum + (m.quantity || 0), 0);
        if (totalQuantity > 100) sustainabilityScore += 10;
        else if (totalQuantity > 50) sustainabilityScore += 5;
      }
      
      // Connections score (up to 25 points)
      sustainabilityScore += Math.min(connectionsCount * 5, 25);
      
      // AI matches score (up to 30 points)
      sustainabilityScore += Math.min(aiMatchesCount * 3, 30);
      
      // Cap at 100
      sustainabilityScore = Math.min(100, sustainabilityScore);
      
      setStats({
        connections: connectionsCount,
        materials_listed: materialsCount,
        matches_found: aiMatchesCount,
        sustainability_score: sustainabilityScore,
        xp: xp || 0,
        streak_days: streak_days || 0
      });
    } catch (error) {
      console.error('Error loading user stats:', error);
    }
  }

  async function loadAIRecommendations(userId: string) {
    try {
      const { data: recommendations } = await supabase
        .from('ai_recommendations')
        .select('*')
        .eq('company_id', userId)
        .eq('status', 'pending')
        .order('confidence', { ascending: false })
        .limit(5);

      if (recommendations && recommendations.length > 0) {
        setAiRecommendations(recommendations);
      } else {
        // Generate new AI recommendations
        const { data: profile } = await supabase
          .from('company_profiles')
          .select('*')
          .eq('company_id', userId)
          .single();

        if (profile) {
          const response = await fetch('/api/generate-recommendations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId, profile })
          });
          
          if (response.ok) {
            const { recommendations: newRecommendations } = await response.json();
            setAiRecommendations(newRecommendations);
          }
        }
      }
    } catch (error) {
      console.error('Error loading AI recommendations:', error);
    }
  }

  async function loadMaterialMatches(userId: string) {
    try {
      // First, generate matches if none exist
      const { data: existingMatches } = await supabase
        .from('material_matches')
        .select('id')
        .limit(1);
      
      if (!existingMatches || existingMatches.length === 0) {
        // Generate initial matches
        await fetch('/api/generate-matches', { method: 'POST' });
      }
      
      // Get user's materials
      const { data: userMaterials } = await supabase
        .from('materials')
        .select('id')
        .eq('company_id', userId);

      if (!userMaterials || userMaterials.length === 0) return;

      // Find matches for user's materials
      const materialIds = userMaterials.map(m => m.id);
      
      const { data: matches } = await supabase
        .from('material_matches')
        .select(`
          *,
          materials!material_matches_material_id_fkey(
            material_name,
            quantity,
            unit,
            description,
            type,
            companies(name)
          ),
          matched_materials!material_matches_matched_material_id_fkey(
            material_name,
            quantity,
            unit,
            description,
            type,
            companies(name)
          )
        `)
        .or(`material_id.in.(${materialIds.join(',')}),matched_material_id.in.(${materialIds.join(',')})`)
        .order('match_score', { ascending: false })
        .limit(10);

      if (matches) {
        const formattedMatches = matches.map(match => {
          const isUserMaterial = materialIds.includes(match.material_id);
          const userMaterial = isUserMaterial ? match.materials : match.matched_materials;
          const otherMaterial = isUserMaterial ? match.matched_materials : match.materials;
          
          return {
            id: match.id,
            material_name: userMaterial.material_name,
            company_name: otherMaterial.companies.name,
            match_score: match.match_score,
            distance: `${Math.floor(Math.random() * 50) + 1}km`,
            type: userMaterial.type,
            quantity: userMaterial.quantity,
            unit: userMaterial.unit,
            description: userMaterial.description,
            matched_material: otherMaterial.material_name,
            sustainability_impact: match.sustainability_impact || Math.round(match.match_score * 100),
            economic_value: match.economic_value || Math.round(match.match_score * 50000)
          };
        });
        setMaterialMatches(formattedMatches);
      }
    } catch (error) {
      console.error('Error loading material matches:', error);
    }
  }

  async function loadRecentActivity(userId: string) {
    try {
      const activities: Activity[] = [];

      // Load recent connections
      const { data: connections } = await supabase
        .from('connections')
        .select(`
          *,
          requester:companies!connections_requester_id_fkey(name),
          recipient:companies!connections_recipient_id_fkey(name)
        `)
        .or(`requester_id.eq.${userId},recipient_id.eq.${userId}`)
        .order('created_at', { ascending: false })
        .limit(3);

      if (connections) {
        connections.forEach(conn => {
          const isRequester = conn.requester_id === userId;
          const otherCompany = isRequester ? conn.recipient.name : conn.requester.name;
          
          activities.push({
            id: conn.id,
            type: 'connection',
            title: `Connection ${conn.status}`,
            description: `${conn.status === 'accepted' ? 'Connected with' : 'Connection request to'} ${otherCompany}`,
            timestamp: formatTimeAgo(conn.created_at),
            status: conn.status === 'accepted' ? 'completed' : 'pending'
          });
        });
      }

      // Load recent material listings
      const { data: materials } = await supabase
        .from('materials')
        .select('*')
        .eq('company_id', userId)
        .order('created_at', { ascending: false })
        .limit(2);

      if (materials) {
        materials.forEach(material => {
          activities.push({
            id: material.id,
            type: 'listing',
            title: 'Material Listed',
            description: `Listed ${material.material_name} (${material.quantity} ${material.unit})`,
            timestamp: formatTimeAgo(material.created_at),
            status: 'active'
          });
        });
      }

      setRecentActivity(activities.sort((a, b) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      ).slice(0, 5));
    } catch (error) {
      console.error('Error loading recent activity:', error);
    }
  }

  function formatTimeAgo(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours} hours ago`;
    const diffInDays = Math.floor(diffInHours / 24);
    if (diffInDays < 7) return `${diffInDays} days ago`;
    return date.toLocaleDateString();
  }

  async function handleRecommendationAction(recommendationId: string) {
    try {
      await supabase
        .from('ai_recommendations')
        .update({ status: 'acted' })
        .eq('id', recommendationId);
      // Reload recommendations
      if (userProfile && userProfile.subscription && userProfile.subscription.tier !== 'free') {
        await loadAIRecommendations(userProfile.id);
      }
    } catch (error) {
      console.error('Error updating recommendation:', error);
    }
  }

  async function handleConnectToMatch(matchId: string) {
    if (userProfile?.subscription?.tier === 'free') {
      setShowUpgradeModal(true);
      return;
    }

    try {
      // Create connection request logic here
      console.log('Connecting to match:', matchId);
      // This would create a connection request in the database
    } catch (error) {
      console.error('Error connecting to match:', error);
    }
  }

  async function loadMessages() {
    if (!userProfile) return;
    const { data } = await supabase
      .from('messages')
      .select('*')
      .or(`sender_id.eq.${userProfile.id},receiver_id.eq.${userProfile.id}`)
      .order('created_at', { ascending: false });
    setMessages(data || []);
  }

  async function sendMessage() {
    if (!userProfile || !recipientId || !newMessage) return;
    await supabase.from('messages').insert([
      {
        sender_id: userProfile.id,
        receiver_id: recipientId,
        content: newMessage,
      },
    ]);
    setNewMessage('');
    loadMessages();
  }

  async function sendAiMessage() {
    if (!aiInput) return;
    // Call your backend AI endpoint (replace with your actual endpoint)
    const res = await fetch('/api/ai-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: aiInput })
    });
    const data = await res.json();
    setAiResponse(data.response || 'No response');
  }

  async function handleUpgrade() {
    if (!userProfile) return;
    
    try {
      // Upsert subscription to 'pro'
      const { error } = await supabase.from('subscriptions').upsert({
        company_id: userProfile.id,
        tier: 'pro',
        status: 'active',
        created_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString() // 1 year
      }, { onConflict: 'company_id' });

      if (error) throw error;

      // Update local state
      setUserProfile(prev => prev ? {
        ...prev,
        subscription: {
          tier: 'pro',
          status: 'active',
          expires_at: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString()
        }
      } : null);

      setShowUpgradeModal(false);
      
      // Reload dashboard data to show AI features
      await loadDashboardData();
      
      // Show success message
      alert('Successfully upgraded to Pro! You now have access to AI-powered features.');
    } catch (error) {
      console.error('Error upgrading subscription:', error);
      alert('Failed to upgrade. Please try again.');
    }
  }

  async function handleConnectToCompany(companyId: string) {
    if (!userProfile) return;
    const { data, error } = await supabase.from('connections').insert({
      requester_id: userProfile.id,
      recipient_id: companyId,
      status: 'pending',
      created_at: new Date().toISOString()
    }).select().single();
    if (data && data.id) {
      navigate(`/transaction/${data.id}`);
    }
  }

  function handleMessageCompany(companyId: string) {
    setRecipientId(companyId);
    // Optionally scroll to the messages section or open a modal
  }

  async function loadSubscriptionData() {
    if (!userProfile?.id) return;
    
    try {
      const userSubscription = await subscriptionService.getUserSubscription(userProfile.id);
      const access = await subscriptionService.getFeatureAccess(userProfile.id);
      
      setSubscription(userSubscription);
      setFeatureAccess(access);
    } catch (error) {
      console.error('Error loading subscription data:', error);
    }
  }

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    setUserProfile(null);
    setSubscription(null);
    setFeatureAccess(null);
    setMessages([]);
    setNewMessage('');
    setRecipientId('');
    setAiInput('');
    setAiResponse('');
    setNdaAccepted(false);
    setOnboardingComplete(false);
    setShowUpgradeModal(false);
    setUpgradeFeature('');
    toast.success('You have signed out.');
    setTimeout(() => {
      window.location.href = '/home';
    }, 1000);
  };

  useEffect(() => {
    if (userProfile && userProfile.role === 'admin') loadAIDebugStatus();
  }, [userProfile]);

  useEffect(() => {
    if (showNotifications) markAllNotificationsRead();
    // eslint-disable-next-line
  }, [showNotifications]);

  async function loadNotifications() {
    if (!userProfile) return;
    const { data, error } = await supabase
      .from('notifications')
      .select('*')
      .eq('user_id', userProfile.id)
      .order('created_at', { ascending: false });
    if (!error && data) {
      setNotifications(data);
      setUnreadCount(data.filter(n => !n.read).length);
    }
  }

  async function markAllNotificationsRead() {
    if (!userProfile) return;
    await supabase
      .from('notifications')
      .update({ read: true })
      .eq('user_id', userProfile.id)
      .eq('read', false);
    await loadNotifications();
  }

  const isProOrAdmin = (userProfile?.subscription?.tier !== 'free') || (userProfile?.role === 'admin');

  // Navigation handlers
  const handleViewAnalytics = () => {
    setShowAnalytics(true);
  };

  const handleAccountSettings = () => {
    setShowAccountSettings(true);
  };

  const handleNotifications = () => {
    setShowNotifications(true);
  };

  const handleChats = () => {
    setShowChats(true);
  };

  const handleOnboarding = () => {
    setShowOnboarding(true);
  };

  const handleMarketplace = () => {
    setShowMarketplace(true);
  };

  const handleBackToDashboard = () => {
    setShowAnalytics(false);
    setShowAccountSettings(false);
    setShowNotifications(false);
    setShowChats(false);
    setShowOnboarding(false);
    setShowMarketplace(false);
    setShowMaterialForm(false);
  };

  async function loadProactiveOpportunities(userId: string) {
    try {
      // Get user's materials
      const { data: userMaterials } = await supabase
        .from('materials')
        .select('*')
        .eq('company_id', userId);
      
      if (!userMaterials || userMaterials.length === 0) {
        setProactiveOpportunities([]);
        return;
      }
      
      // Find high-value opportunities based on user's materials
      const opportunities = [];
      
      for (const material of userMaterials) {
        // Find potential matches for this material
        const { data: potentialMatches } = await supabase
          .from('materials')
          .select(`
            *,
            companies(name, industry, location)
          `)
          .eq('type', material.type === 'waste' ? 'requirement' : 'waste')
          .neq('company_id', userId)
          .limit(3);
        
        if (potentialMatches && potentialMatches.length > 0) {
          for (const match of potentialMatches) {
            const matchScore = calculateMatchScore(material, match, []);
            if (matchScore > 0.6) {
              opportunities.push({
                id: `opp-${material.id}-${match.id}`,
                title: `${material.material_name} → ${match.companies.name}`,
                description: `High-value ${material.type} exchange opportunity`,
                potential_savings: Math.round(matchScore * 50000),
                sustainability_impact: Math.round(matchScore * 100),
                confidence: Math.round(matchScore * 100),
                company_name: match.companies.name,
                material_name: match.material_name
              });
            }
          }
        }
      }
      
      setProactiveOpportunities(opportunities.slice(0, 5));
    } catch (error) {
      console.error('Error loading proactive opportunities:', error);
    }
  }

  // Calculate match score helper function
  function calculateMatchScore(material1: any, material2: any, allMaterials: any[]) {
    let score = 0;
    
    // Material compatibility (40% weight)
    if (material1.material_name.toLowerCase().includes(material2.material_name.toLowerCase()) ||
        material2.material_name.toLowerCase().includes(material1.material_name.toLowerCase())) {
      score += 0.4;
    }
    
    // Quantity compatibility (30% weight)
    const quantityRatio = Math.min(material1.quantity, material2.quantity) / Math.max(material1.quantity, material2.quantity);
    score += quantityRatio * 0.3;
    
    // Industry compatibility (20% weight)
    if (material1.companies?.industry === material2.companies?.industry) {
      score += 0.2;
    }
    
    // Location proximity (10% weight)
    if (material1.companies?.location === material2.companies?.location) {
      score += 0.1;
    }
    
    return Math.min(1, score);
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="flex flex-col items-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500 mb-4"></div>
          <div className="text-gray-500">Loading dashboard...</div>
          {loadError && (
            <div className="mt-4 text-red-600 bg-red-50 border border-red-200 px-4 py-2 rounded">
              {loadError}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (showNotifications) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-3xl font-bold text-gray-900">Notifications</h1>
            <button
              onClick={handleBackToDashboard}
              className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600"
            >
              ← Back to Dashboard
            </button>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6">
            {notifications.length === 0 ? (
              <div className="text-center py-8 text-gray-500">No notifications yet.</div>
            ) : (
              <div className="space-y-4">
                {notifications.map((n) => (
                  <div key={n.id} className={`flex items-start space-x-3 p-4 border rounded-lg ${!n.read ? 'bg-emerald-50' : ''}`}>
                    <div className="pt-1">
                      {n.type === 'match' && <Zap className="h-5 w-5 text-emerald-500" />}
                      {n.type === 'connection' && <Users className="h-5 w-5 text-blue-500" />}
                      {n.type === 'message' && <MessageSquare className="h-5 w-5 text-purple-500" />}
                      {n.type === 'material' && <Factory className="h-5 w-5 text-orange-500" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <h3 className="font-semibold text-gray-900">{n.title}</h3>
                        {!n.read && <span className="text-xs bg-emerald-200 text-emerald-800 px-2 py-1 rounded-full">New</span>}
                      </div>
                      <p className="text-gray-600 text-sm">{n.body}</p>
                      <p className="text-xs text-gray-400 mt-1">{new Date(n.created_at).toLocaleString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (showChats) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-3xl font-bold text-gray-900">Chats</h1>
            <button
              onClick={handleBackToDashboard}
              className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600"
            >
              ← Back to Dashboard
            </button>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="text-center py-8">
              <h2 className="text-xl font-bold mb-2">No Active Chats</h2>
              <p className="text-gray-600 mb-4">Start connecting with companies to begin chatting</p>
              <button
                onClick={handleMarketplace}
                className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600"
              >
                Browse Marketplace
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (showOnboarding) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-3xl font-bold text-gray-900">
              {onboardingComplete ? 'Manage Onboarding' : 'AI Onboarding'}
            </h1>
            <button
              onClick={handleBackToDashboard}
              className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600"
            >
              ← Back to Dashboard
            </button>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6">
            {onboardingComplete ? (
              <div className="space-y-4">
                <div className="border-b pb-4">
                  <h2 className="text-xl font-bold mb-2">Your Current Setup</h2>
                  <p className="text-gray-600">Manage your company profile and material listings</p>
                </div>
                <button 
                  onClick={() => navigate('/onboarding')}
                  className="w-full p-4 border rounded-lg text-left hover:bg-gray-50"
                >
                  <h3 className="font-semibold">Edit Company Profile</h3>
                  <p className="text-gray-600">Update your company information and preferences</p>
                </button>
                <button 
                  onClick={() => setShowMaterialForm(true)}
                  className="w-full p-4 border rounded-lg text-left hover:bg-gray-50"
                >
                  <h3 className="font-semibold">Manage Materials</h3>
                  <p className="text-gray-600">Add, edit, or remove your material listings</p>
                </button>
                <button 
                  onClick={() => {
                    setOnboardingComplete(false);
                    handleBackToDashboard();
                  }}
                  className="w-full p-4 border rounded-lg text-left hover:bg-gray-50 text-red-600"
                >
                  <h3 className="font-semibold">Reset Onboarding</h3>
                  <p className="text-gray-600">Start fresh with a new onboarding process</p>
                </button>
              </div>
            ) : (
              <div className="text-center py-8">
                <h2 className="text-xl font-bold mb-4">Complete Your AI Onboarding</h2>
                <p className="text-gray-600 mb-6">Set up your company profile to get personalized AI recommendations</p>
                <button
                  onClick={() => navigate('/onboarding')}
                  className="bg-emerald-500 text-white px-6 py-3 rounded-lg hover:bg-emerald-600 font-medium"
                >
                  Start Onboarding
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (showMarketplace) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-3xl font-bold text-gray-900">Marketplace</h1>
            <button
              onClick={handleBackToDashboard}
              className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600"
            >
              ← Back to Dashboard
            </button>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="text-center py-8">
              <h2 className="text-xl font-bold mb-2">Marketplace Coming Soon</h2>
              <p className="text-gray-600">Advanced marketplace features will be available here.</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Show MaterialForm if needed
  if (showMaterialForm) {
    return (
      <MaterialForm 
        onClose={() => setShowMaterialForm(false)} 
        type="waste"
      />
    );
  }

  async function loadAIDebugStatus() {
    setAiDebug(a => ({ ...a, loading: true, error: '' }));
    try {
      const { data: matches, error } = await supabase
        .from('material_matches')
        .select('id, material_id, matched_material_id, match_score, created_at')
        .order('created_at', { ascending: false })
        .limit(10);
      if (error) throw error;
      const matchCount = matches?.length || 0;
      const lastRun = matches && matches.length > 0 ? matches[0].created_at : '';
      setAiDebug(a => ({
        ...a,
        lastRun,
        matchCount,
        recentMatches: matches || [],
        loading: false,
        error: ''
      }));
    } catch (err) {
      setAiDebug(a => ({ ...a, loading: false, error: (err as any).message }));
    }
  }

  async function forceAIMatchGeneration() {
    setAiDebug(a => ({ ...a, loading: true, error: '' }));
    try {
      await fetch('/api/generate-matches', { method: 'POST' });
      await loadAIDebugStatus();
    } catch (err) {
      setAiDebug(a => ({ ...a, loading: false, error: (err as any).message }));
    }
  }

  useEffect(() => {
    if (userProfile && userProfile.role === 'admin') loadAIDebugStatus();
  }, [userProfile]);

  useEffect(() => {
    if (showNotifications) markAllNotificationsRead();
    // eslint-disable-next-line
  }, [showNotifications]);

  async function loadNotifications() {
    if (!userProfile) return;
    const { data, error } = await supabase
      .from('notifications')
      .select('*')
      .eq('user_id', userProfile.id)
      .order('created_at', { ascending: false });
    if (!error && data) {
      setNotifications(data);
      setUnreadCount(data.filter(n => !n.read).length);
    }
  }

  async function markAllNotificationsRead() {
    if (!userProfile) return;
    await supabase
      .from('notifications')
      .update({ read: true })
      .eq('user_id', userProfile.id)
      .eq('read', false);
    await loadNotifications();
  }

  async function handleExplainMatch(matchId: string) {
    setExplainMatchId(matchId);
    setExplainLoading(true);
    setExplainMatchText('');
    try {
      const res = await fetch(`/api/explain-match/${matchId}`);
      const data = await res.json();
      setExplainMatchText(data.explanation || 'No explanation available.');
    } catch (err) {
      setExplainMatchText('Failed to load explanation.');
    }
    setExplainLoading(false);
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <div className="flex items-center space-x-2">
                <Workflow className="h-8 w-8 text-emerald-500" />
                <span className="text-2xl font-bold text-gray-900">SymbioFlow</span>
              </div>
              <div className="hidden md:ml-10 md:flex md:space-x-8">
                <button
                  onClick={() => navigate('/dashboard')}
                  className="text-emerald-600 border-b-2 border-emerald-600 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Dashboard
                </button>
                <button
                  onClick={() => navigate('/marketplace')}
                  className="text-gray-500 hover:text-gray-700 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Marketplace
                </button>
                <button
                  onClick={() => navigate('/home')}
                  className="text-gray-500 hover:text-gray-700 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Home
                </button>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/marketplace')}
                className="text-gray-600 hover:text-gray-900 transition"
              >
                Marketplace
              </button>
              <button
                onClick={handleNotifications}
                className="text-gray-600 hover:text-gray-900 transition relative"
              >
                <Bell className="h-5 w-5" />
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">
                    {unreadCount}
                  </span>
                )}
              </button>
              <button
                onClick={handleChats}
                className="text-gray-600 hover:text-gray-900 transition"
              >
                <MessageSquare className="h-5 w-5" />
              </button>
              <button
                onClick={handleSignOut}
                className="text-gray-600 hover:text-gray-900 transition"
              >
                Sign Out
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Welcome back, {userProfile?.name || ''}
              </h1>
              <p className="text-gray-600 mt-1">
                Level {userProfile?.level || 1} • {userProfile?.xp || 0} XP • {userProfile?.streak_days || 0} day streak
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Crown className={`h-5 w-5 ${isProOrAdmin ? 'text-yellow-500' : 'text-gray-400'}`} />
                <span className="text-sm font-medium text-gray-700">
                  {userProfile?.subscription?.tier?.toUpperCase() || 'FREE'} Plan
                </span>
              </div>
              {!isProOrAdmin && (
                <button
                  onClick={() => setShowUpgradeModal(true)}
                  className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition text-sm font-medium"
                >
                  Upgrade to Pro
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="mb-4">
          {!onboardingComplete ? (
            <button
              onClick={handleOnboarding}
              className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition font-medium"
            >
              Complete AI Onboarding
            </button>
          ) : (
            <div className="flex space-x-2">
              <button
                onClick={handleOnboarding}
                className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition font-medium"
              >
                Manage Onboarding
              </button>
              <button
                onClick={() => setShowMaterialForm(true)}
                className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition font-medium"
              >
                Add New Material
              </button>
            </div>
          )}
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Users className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Connections</p>
                <p className="text-2xl font-bold text-gray-900">{stats.connections || 0}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <Factory className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Materials Listed</p>
                <p className="text-2xl font-bold text-gray-900">{stats.materials_listed || 0}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Zap className="h-6 w-6 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">AI Matches</p>
                <p className="text-2xl font-bold text-gray-900">{stats.matches_found || 0}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6 relative">
            <div className="flex items-center">
              <div className="p-2 bg-emerald-100 rounded-lg">
                <TrendingUp className="h-6 w-6 text-emerald-600" />
              </div>
              <div className="ml-4 flex items-center">
                <p className="text-sm font-medium text-gray-600">Sustainability Score</p>
                <button
                  className="ml-2 text-emerald-600 hover:text-emerald-800"
                  onClick={() => setShowSustainabilityInfo(true)}
                  aria-label="Show sustainability score breakdown"
                >
                  <Info className="h-4 w-4" />
                </button>
              </div>
            </div>
            <p className="text-2xl font-bold text-gray-900 mt-2">{stats.sustainability_score?.toFixed(2) || '0.00'}%</p>

            {/* Sustainability Score Breakdown Modal */}
            {showSustainabilityInfo && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
                <div className="bg-white rounded-xl shadow-lg p-6 w-full max-w-md relative">
                  <button
                    className="absolute top-3 right-3 text-gray-400 hover:text-gray-700"
                    onClick={() => setShowSustainabilityInfo(false)}
                  >
                    <X className="h-5 w-5" />
                  </button>
                  <h2 className="text-xl font-bold mb-4 text-emerald-700">Sustainability Score Breakdown</h2>
                  <ul className="space-y-2 text-gray-700 text-sm mb-4">
                    <li><b>+20</b> for listing at least one material</li>
                    <li><b>+15</b> for having both waste and requirement materials</li>
                    <li><b>+10</b> for total material quantity &gt; 100 (or +5 for &gt; 50)</li>
                    <li><b>+5</b> per accepted connection (up to 25)</li>
                    <li><b>+3</b> per AI match (up to 30)</li>
                    <li><b>Max score:</b> 100</li>
                  </ul>
                  <div className="text-xs text-gray-500 mb-2">How to improve:</div>
                  <ul className="text-xs text-gray-600 space-y-1">
                    <li>• List more materials and diversify (waste + requirements)</li>
                    <li>• Increase your material quantities</li>
                    <li>• Connect with more companies</li>
                    <li>• Enable AI matches by listing compatible materials</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* AI Panels: Proactive Opportunities, GNN, Multi-Hop */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <div className="col-span-1">
            <div className="bg-white rounded-xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900">Proactive Opportunities</h2>
                <Zap className="h-5 w-5 text-emerald-500" />
              </div>
              {proactiveOpportunities.length > 0 ? (
                <div className="space-y-3">
                  {proactiveOpportunities.slice(0, 3).map((opp) => (
                    <div key={opp.id} className="border border-gray-200 rounded-lg p-3">
                      <h3 className="font-semibold text-sm text-gray-900">{opp.title}</h3>
                      <p className="text-xs text-gray-600 mb-2">{opp.description}</p>
                      <div className="flex justify-between text-xs">
                        <span className="text-emerald-600">€{opp.potential_savings}</span>
                        <span className="text-blue-600">{opp.confidence}% match</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4">
                  <p className="text-gray-500 text-sm">No opportunities yet</p>
                  <p className="text-gray-400 text-xs">Add materials to find matches</p>
                </div>
              )}
            </div>
          </div>
          <div className="col-span-1">
            <GnnSymbiosisPanel />
          </div>
          <div className="col-span-1">
            {/* Multi-Hop Symbiosis Panel */}
            {userProfile?.id && (
              <MultiHopSymbiosisPanel userId={userProfile.id} />
            )}
          </div>
        </div>

        {/* Global Impact Panel */}
        <div className="mb-8">
          <GlobalImpactPanel userId={userProfile?.id} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* AI Recommendations */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-2">
                  <Brain className="h-6 w-6 text-emerald-500" />
                  <h2 className="text-xl font-bold text-gray-900">
                    {isProOrAdmin ? 'AI Recommendations' : 'AI Recommendations (Pro Feature)'}
                  </h2>
                </div>
                {isProOrAdmin && (
                  <button 
                    onClick={() => loadAIRecommendations(userProfile!.id)}
                    className="text-emerald-600 hover:text-emerald-700 text-sm font-medium"
                  >
                    Refresh
                  </button>
                )}
              </div>

              {!isProOrAdmin ? (
                <div className="text-center py-12 border-2 border-dashed border-gray-200 rounded-lg">
                  <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Unlock AI-Powered Matching
                  </h3>
                  <p className="text-gray-600 mb-6">
                    Get personalized recommendations, smart material matching, and priority connections with our Pro plan.
                  </p>
                  <button
                    onClick={() => setShowUpgradeModal(true)}
                    className="bg-emerald-500 text-white px-6 py-3 rounded-lg hover:bg-emerald-600 transition font-medium"
                  >
                    Upgrade to Pro
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  {aiRecommendations.length === 0 ? (
                    <div className="text-center py-8">
                      <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
                      <p className="text-gray-600">No new recommendations at the moment. Check back later!</p>
                    </div>
                  ) : (
                    aiRecommendations.map((rec) => (
                      <div key={rec.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-2">
                              {rec.type === 'connection' && <Users className="h-4 w-4 text-blue-500" />}
                              {rec.type === 'material' && <Factory className="h-4 w-4 text-green-500" />}
                              {rec.type === 'opportunity' && <Star className="h-4 w-4 text-yellow-500" />}
                              <span className="text-sm font-medium text-gray-600 capitalize">
                                {rec.type}
                              </span>
                              <span className="text-xs bg-emerald-100 text-emerald-800 px-2 py-1 rounded-full">
                                {rec.confidence}% match
                              </span>
                            </div>
                            <h3 className="font-semibold text-gray-900 mb-1">{rec.title}</h3>
                            <p className="text-sm text-gray-600 mb-3">{rec.description}</p>
                            <button 
                              onClick={() => handleRecommendationAction(rec.id)}
                              className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition text-sm font-medium"
                            >
                              Take Action
                            </button>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>

            {/* Material Matches Section */}
            <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
              <h2 className="text-xl font-bold mb-4">Material Matches</h2>
              {materialMatches.length === 0 ? (
                <div>No matches found.</div>
              ) : (
                <div>
                  {materialMatches.map(match => (
                    <div key={match.id} className="mb-4 p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                          match.type === 'waste' ? 'bg-orange-100 text-orange-800' : 'bg-blue-100 text-blue-800'
                        }`}>
                          {match.type === 'waste' ? 'Available' : 'Needed'}
                        </span>
                        <span className="text-sm font-medium text-emerald-600">
                          {match.match_score}% match
                        </span>
                      </div>
                      <h4 className="font-semibold text-gray-900 mb-1">{match.material_name}</h4>
                      <p className="text-sm text-gray-600 mb-2">
                        {match.quantity} {match.unit} • {match.distance}
                      </p>
                      <p className="text-sm text-gray-500 mb-3">{match.company_name}</p>
                      <button
                        onClick={() => handleConnectToCompany(match.company_name)}
                        className="w-full bg-emerald-500 text-white py-2 px-4 rounded-lg hover:bg-emerald-600 transition text-sm font-medium"
                      >
                        Connect
                      </button>
                      <button
                        className="ml-2 px-2 py-1 text-xs bg-emerald-100 text-emerald-800 rounded hover:bg-emerald-200"
                        onClick={() => handleExplainMatch(match.id)}
                      >
                        Why this match?
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Quick Actions & Recent Activity */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h2>
              <div className="space-y-3">
                <button 
                  onClick={() => setShowMaterialForm(true)}
                  className="w-full flex items-center space-x-3 p-3 text-left hover:bg-gray-50 rounded-lg transition"
                >
                  <Plus className="h-5 w-5 text-emerald-500" />
                  <span className="font-medium">List New Material</span>
                </button>
                <button 
                  onClick={handleMarketplace}
                  className="w-full flex items-center space-x-3 p-3 text-left hover:bg-gray-50 rounded-lg transition"
                >
                  <Globe className="h-5 w-5 text-blue-500" />
                  <span className="font-medium">Find Partners</span>
                </button>
                <button
                  onClick={handleViewAnalytics}
                  className="w-full flex items-center space-x-3 p-3 text-left hover:bg-gray-50 rounded-lg transition"
                >
                  <BarChart3 className="h-5 w-5 text-purple-500" />
                  <span className="font-medium">View Analytics</span>
                </button>
                <button
                  onClick={handleAccountSettings}
                  className="w-full flex items-center space-x-3 p-3 text-left hover:bg-gray-50 rounded-lg transition"
                >
                  <Settings className="h-5 w-5 text-gray-500" />
                  <span className="font-medium">Account Settings</span>
                </button>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Activity</h2>
              <div className="space-y-4">
                {recentActivity.length === 0 ? (
                  <div className="text-center py-6">
                    <Clock className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-500 text-sm">No recent activity</p>
                  </div>
                ) : (
                  recentActivity.map((activity) => (
                    <div key={activity.id} className="flex items-start space-x-3">
                      <div className={`p-2 rounded-full ${
                        activity.type === 'match' ? 'bg-green-100' :
                        activity.type === 'message' ? 'bg-blue-100' :
                        activity.type === 'listing' ? 'bg-purple-100' :
                        'bg-gray-100'
                      }`}>
                        {activity.type === 'match' && <Zap className="h-4 w-4 text-green-600" />}
                        {activity.type === 'message' && <MessageSquare className="h-4 w-4 text-blue-600" />}
                        {activity.type === 'listing' && <Factory className="h-4 w-4 text-purple-600" />}
                        {activity.type === 'connection' && <Users className="h-4 w-4 text-gray-600" />}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <p className="text-sm font-medium text-gray-900">{activity.title}</p>
                          {activity.status === 'pending' && (
                            <AlertCircle className="h-3 w-3 text-yellow-500" />
                          )}
                          {activity.status === 'completed' && (
                            <CheckCircle className="h-3 w-3 text-green-500" />
                          )}
                        </div>
                        <p className="text-sm text-gray-600">{activity.description}</p>
                        <p className="text-xs text-gray-500 mt-1">{activity.timestamp}</p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        {/* AI-Powered Recommendations */}
        {userProfile && (
          <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
            <h2 className="text-xl font-bold mb-4">AI-Powered Recommendations</h2>
            {aiRecommendations.length === 0 ? (
              <div>No recommendations yet. List materials or update your profile to get matches.</div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {aiRecommendations.map(rec => (
                  <div key={rec.id} className="border rounded-lg p-4 shadow">
                    <h3 className="font-semibold text-lg mb-2">{rec.title}</h3>
                    <p className="mb-1">{rec.description}</p>
                    <div className="mb-1">Confidence: <span className="font-bold">{rec.confidence?.toFixed(1) || rec.confidence}%</span></div>
                    <button onClick={() => handleRecommendationAction(rec.id)} className="mt-2 bg-emerald-500 text-white px-4 py-1 rounded">Mark as Acted</button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Subscription Status */}
        {subscription && (
          <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Crown className="h-6 w-6 text-yellow-500" />
                <div>
                  <h3 className="font-semibold text-gray-900">
                    {subscription.tier === 'pro' ? 'Pro Plan' : 
                     subscription.tier === 'enterprise' ? 'Enterprise Plan' : 'Free Plan'}
                  </h3>
                  <p className="text-sm text-gray-600">
                    {subscription.tier === 'free' ? 'Upgrade to unlock AI features' : 
                     'Active subscription - All features unlocked'}
                  </p>
                </div>
              </div>
              {subscription.tier === 'free' && (
                <button
                  onClick={() => {
                    setUpgradeFeature('AI features');
                    setShowUpgradeModal(true);
                  }}
                  className="bg-purple-500 text-white px-4 py-2 rounded-lg hover:bg-purple-600 transition"
                >
                  Upgrade to Pro
                </button>
              )}
            </div>
          </div>
        )}

        {/* AI Recommendations Section */}
        <div className="bg-white rounded-xl shadow-sm p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <Brain className="h-6 w-6 text-blue-500" />
              <h3 className="text-lg font-semibold text-gray-900">AI Recommendations</h3>
            </div>
            {featureAccess?.aiRecommendations ? (
              <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-semibold rounded-full">
                PRO FEATURE
              </span>
            ) : (
              <button
                onClick={() => {
                  setUpgradeFeature('AI recommendations');
                  setShowUpgradeModal(true);
                }}
                className="text-purple-600 hover:text-purple-700 text-sm font-medium"
              >
                Upgrade to unlock
              </button>
            )}
          </div>
          
          {featureAccess?.aiRecommendations ? (
            <div className="space-y-3">
              {aiRecommendations.map((rec) => (
                <div key={rec.id} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-medium text-gray-900">{rec.title}</h4>
                      <p className="text-sm text-gray-600 mt-1">{rec.description}</p>
                      <div className="flex items-center space-x-2 mt-2">
                        <span className="text-xs text-gray-500">Confidence: {rec.confidence}%</span>
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          rec.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                          rec.status === 'completed' ? 'bg-green-100 text-green-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {rec.status}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={() => handleRecommendationAction(rec.id)}
                      className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                    >
                      View Details
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 border-2 border-dashed border-gray-200 rounded-lg">
              <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h4 className="text-lg font-medium text-gray-900 mb-2">AI Recommendations Locked</h4>
              <p className="text-gray-600 mb-4">
                Upgrade to Pro to get personalized AI recommendations for connections and opportunities.
              </p>
              <button
                onClick={() => {
                  setUpgradeFeature('AI recommendations');
                  setShowUpgradeModal(true);
                }}
                className="bg-purple-500 text-white px-6 py-2 rounded-lg hover:bg-purple-600 transition"
              >
                Upgrade to Pro
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Admin AI Debug/Status Panel */}
      {userProfile?.role === 'admin' && (
        <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-gray-900">AI Debug / Status Panel</h2>
            <button
              onClick={forceAIMatchGeneration}
              className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600"
              disabled={aiDebug.loading}
            >
              {aiDebug.loading ? 'Running...' : 'Force AI Match Generation'}
            </button>
          </div>
          {aiDebug.error && <div className="text-red-500 mb-2">{aiDebug.error}</div>}
          <div className="mb-2 text-sm text-gray-700">Last match generation: {aiDebug.lastRun ? new Date(aiDebug.lastRun).toLocaleString() : 'Never'}</div>
          <div className="mb-4 text-sm text-gray-700">Recent matches: {aiDebug.matchCount}</div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead>
                <tr className="text-gray-500">
                  <th className="px-2 py-1">Material ID</th>
                  <th className="px-2 py-1">Matched Material ID</th>
                  <th className="px-2 py-1">Score</th>
                  <th className="px-2 py-1">Created</th>
                </tr>
              </thead>
              <tbody>
                {aiDebug.recentMatches.map((m: any) => (
                  <tr key={m.id} className="border-b">
                    <td className="px-2 py-1">{m.material_id}</td>
                    <td className="px-2 py-1">{m.matched_material_id}</td>
                    <td className="px-2 py-1">{(m.match_score * 100).toFixed(1)}%</td>
                    <td className="px-2 py-1">{new Date(m.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Subscription Upgrade Modal */}
      <SubscriptionUpgradeModal
        isOpen={showUpgradeModal}
        onClose={() => setShowUpgradeModal(false)}
        requiredFeature={upgradeFeature}
        currentTier={subscription?.tier || 'free'}
      />

      <ToastContainer position="top-center" autoClose={2000} />

      {/* Modal for match explanation */}
      {explainMatchId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-white rounded-lg shadow-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-bold mb-2">Why this match?</h3>
            {explainLoading ? (
              <div className="text-gray-500">Loading...</div>
            ) : (
              <div className="text-gray-700 whitespace-pre-line">{explainMatchText}</div>
            )}
            <button
              className="mt-4 px-4 py-2 bg-emerald-500 text-white rounded hover:bg-emerald-600"
              onClick={() => setExplainMatchId(null)}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}