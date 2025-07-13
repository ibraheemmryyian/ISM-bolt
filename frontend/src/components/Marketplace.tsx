import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  Plus, 
  MapPin, 
  Calendar, 
  DollarSign, 
  Users, 
  Factory, 
  Recycle, 
  Zap, 
  Leaf, 
  Target,
  Workflow,
  Eye,
  MessageSquare,
  Star,
  TrendingUp,
  CheckCircle,
  AlertCircle,
  Clock,
  ArrowRight,
  Filter as FilterIcon,
  SortAsc,
  SortDesc,
  X,
  Heart,
  Check
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { messagingService } from '../lib/messagingService';
import { MaterialForm } from './MaterialForm';
import { MaterialApproval } from './MaterialApproval';
import { useNavigate, Link } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';

interface MarketplaceProps {
  onSignOut?: () => void;
}

interface Material {
  id: string;
  material_name: string;
  quantity: number;
  unit: string;
  description: string;
  type: 'waste' | 'requirement';
  created_at: string;
  company_id: string;
  status?: 'pending' | 'approved' | 'rejected';
  marketplace_posted?: boolean;
  ai_confidence?: number;
  company?: {
    name: string;
    location?: string;
  };
  distance?: string;
  match_score?: number;
}

interface Company {
  id: string;
  name: string;
  role: string;
  location?: string;
  organization_type?: string;
  materials_of_interest?: string;
  sustainability_score?: number;
}

export function Marketplace({ onSignOut }: MarketplaceProps) {
  const { isDark } = useTheme();
  const [activeTab, setActiveTab] = useState<'materials' | 'companies' | 'pending'>('materials');
  const [materials, setMaterials] = useState<Material[]>([]);
  const [companies, setCompanies] = useState<Company[]>([]);
  const [filteredMaterials, setFilteredMaterials] = useState<Material[]>([]);
  const [filteredCompanies, setFilteredCompanies] = useState<Company[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'waste' | 'requirement'>('all');
  const [showMaterialForm, setShowMaterialForm] = useState<'waste' | 'requirement' | null>(null);
  const [loading, setLoading] = useState(true);
  const [matchResults, setMatchResults] = useState<{ [materialId: string]: any }>({});
  const [loadingMatch, setLoadingMatch] = useState<string | null>(null);
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);
  const [connections, setConnections] = useState<Set<string>>(new Set());
  const [favorites, setFavorites] = useState<Set<string>>(new Set());
  const [loadingActions, setLoadingActions] = useState<{ [key: string]: boolean }>({});
  const [showFilters, setShowFilters] = useState(false);
  const [locationFilter, setLocationFilter] = useState('');
  const [quantityFilter, setQuantityFilter] = useState('');
  const [dateFilter, setDateFilter] = useState('');
  const [pendingMaterials, setPendingMaterials] = useState<Material[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    loadUserAndData();
  }, []);

  useEffect(() => {
    filterMaterials();
  }, [materials, searchQuery, filterType, locationFilter, quantityFilter, dateFilter]);

  useEffect(() => {
    filterCompanies();
  }, [companies, searchQuery]);

  async function loadUserAndData() {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        setCurrentUserId(user.id);
        await loadConnections(user.id);
        await loadFavorites(user.id);
      }
      await loadMarketplaceData();
      setLoading(false);
    } catch (error) {
      console.error('Error loading user data:', error);
      setLoading(false);
    }
  }

  async function loadConnections(userId: string) {
    try {
      const { data: connectionsData } = await supabase
        .from('connections')
        .select('connected_company_id')
        .eq('user_id', userId);

      if (connectionsData) {
        const connectionIds = new Set(connectionsData.map(c => c.connected_company_id));
        setConnections(connectionIds);
      }
    } catch (error) {
      console.error('Error loading connections:', error);
    }
  }

  async function loadFavorites(userId: string) {
    try {
      const { data: favoritesData } = await supabase
        .from('favorites')
        .select('material_id')
        .eq('user_id', userId);

      if (favoritesData) {
        const favoriteIds = new Set(favoritesData.map(f => f.material_id));
        setFavorites(favoriteIds);
      }
    } catch (error) {
      console.error('Error loading favorites:', error);
    }
  }

  async function loadMarketplaceData() {
    try {
      // Load approved materials for marketplace
      const { data: materialsData, error: materialsError } = await supabase
        .from('materials')
        .select('*')
        .eq('status', 'approved')
        .eq('marketplace_posted', true);

      if (materialsError) throw materialsError;

      const enhancedMaterials = materialsData?.map(material => ({
        ...material,
        company: { 
          name: material.companies?.name || 'Unknown Company', 
          location: material.companies?.location || 'Unknown Location'
        },
        distance: 'Calculating...',
        match_score: 0
      })) || [];

      setMaterials(enhancedMaterials);

      // Load pending materials for approval
      const { data: pendingData, error: pendingError } = await supabase
        .from('materials')
        .select('*')
        .eq('status', 'pending');

      if (pendingError) throw pendingError;

      const enhancedPending = pendingData?.map(material => ({
        ...material,
        company: { 
          name: material.companies?.name || 'Unknown Company', 
          location: material.companies?.location || 'Unknown Location'
        },
        ai_confidence: material.ai_confidence || 75
      })) || [];

      setPendingMaterials(enhancedPending);

      // Load companies
      const { data: companiesData, error: companiesError } = await supabase
        .from('companies')
        .select('*');

      if (companiesError) throw companiesError;

      const enhancedCompanies = companiesData?.map(company => ({
        id: company.id,
        name: company.name,
        role: company.role || 'user',
        location: company.location,
        organization_type: company.industry,
        materials_of_interest: company.process_description,
        sustainability_score: 0
      })) || [];

      setCompanies(enhancedCompanies);

    } catch (error) {
      console.error('Error loading marketplace data:', error);
    }
  }

  function filterMaterials() {
    let filtered = materials;

    if (searchQuery) {
      filtered = filtered.filter(material =>
        material.material_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        material.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        material.company?.name.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    if (filterType !== 'all') {
      filtered = filtered.filter(material => material.type === filterType);
    }

    if (locationFilter) {
      filtered = filtered.filter(material =>
        material.company?.location?.toLowerCase().includes(locationFilter.toLowerCase())
      );
    }

    if (quantityFilter) {
      const quantity = parseFloat(quantityFilter);
      if (!isNaN(quantity)) {
        filtered = filtered.filter(material => material.quantity >= quantity);
      }
    }

    if (dateFilter) {
      const filterDate = new Date(dateFilter);
      filtered = filtered.filter(material => 
        new Date(material.created_at) >= filterDate
      );
    }

    setFilteredMaterials(filtered);
  }

  function filterCompanies() {
    let filtered = companies;

    if (searchQuery) {
      filtered = filtered.filter(company =>
        company.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        company.location?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        company.organization_type?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    setFilteredCompanies(filtered);
  }

  async function handleConnect(companyId: string) {
    if (!currentUserId) {
      alert('Please log in to connect');
      return;
    }

    setLoadingActions(prev => ({ ...prev, [`connect-${companyId}`]: true }));

    try {
      const { error } = await supabase
        .from('connections')
        .insert({
          user_id: currentUserId,
          connected_company_id: companyId,
          status: 'connected'
        });

      if (error) throw error;

      setConnections(prev => new Set([...prev, companyId]));
      
      const successMessage = document.createElement('div');
      successMessage.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
      successMessage.textContent = 'Connection established!';
      document.body.appendChild(successMessage);
      setTimeout(() => document.body.removeChild(successMessage), 3000);
    } catch (error) {
      console.error('Error connecting:', error);
      const errorMessage = document.createElement('div');
      errorMessage.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
      errorMessage.textContent = 'Failed to connect. Please try again.';
      document.body.appendChild(errorMessage);
      setTimeout(() => document.body.removeChild(errorMessage), 3000);
    } finally {
      setLoadingActions(prev => ({ ...prev, [`connect-${companyId}`]: false }));
    }
  }

  async function handleChat(companyId: string) {
    if (!currentUserId) {
      alert('Please log in to start a chat');
      return;
    }

    setLoadingActions(prev => ({ ...prev, [`chat-${companyId}`]: true }));

    try {
      const { data: existingChat } = await supabase
        .from('chats')
        .select('id')
        .or(`user1_id.eq.${currentUserId},user2_id.eq.${currentUserId}`)
        .or(`user1_id.eq.${companyId},user2_id.eq.${companyId}`)
        .single();

      if (existingChat) {
        navigate(`/chats?chat=${existingChat.id}`);
      } else {
        const { data: newChat, error } = await supabase
          .from('chats')
          .insert({
            user1_id: currentUserId,
            user2_id: companyId,
            created_at: new Date().toISOString()
          })
          .select()
          .single();

        if (error) throw error;
        navigate(`/chats?chat=${newChat.id}`);
      }
    } catch (error) {
      console.error('Error starting chat:', error);
      const errorMessage = document.createElement('div');
      errorMessage.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
      errorMessage.textContent = 'Failed to start chat. Please try again.';
      document.body.appendChild(errorMessage);
      setTimeout(() => document.body.removeChild(errorMessage), 3000);
    } finally {
      setLoadingActions(prev => ({ ...prev, [`chat-${companyId}`]: false }));
    }
  }

  async function handleFavorite(materialId: string) {
    if (!currentUserId) {
      alert('Please log in to favorite materials');
      return;
    }

    setLoadingActions(prev => ({ ...prev, [`favorite-${materialId}`]: true }));

    try {
      if (favorites.has(materialId)) {
        const { error } = await supabase
          .from('favorites')
          .delete()
          .eq('user_id', currentUserId)
          .eq('material_id', materialId);

        if (error) throw error;
        setFavorites(prev => {
          const newFavorites = new Set(prev);
          newFavorites.delete(materialId);
          return newFavorites;
        });
      } else {
        const { error } = await supabase
          .from('favorites')
          .insert({
            user_id: currentUserId,
            material_id: materialId
          });

        if (error) throw error;
        setFavorites(prev => new Set([...prev, materialId]));
      }
    } catch (error) {
      console.error('Error handling favorite:', error);
    } finally {
      setLoadingActions(prev => ({ ...prev, [`favorite-${materialId}`]: false }));
    }
  }

  async function handleFindMatches(material: Material) {
    if (!currentUserId) {
      alert('Please log in to find matches');
      return;
    }

    setLoadingMatch(material.id);

    try {
      const response = await fetch('/api/generate-matches', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          materialId: material.id,
          companyId: currentUserId 
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        setMatchResults(prev => ({ 
          ...prev, 
          [material.id]: result
        }));
        
        const successMessage = document.createElement('div');
        successMessage.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
        successMessage.textContent = `Found ${result.total || 0} new matches!`;
        document.body.appendChild(successMessage);
        setTimeout(() => document.body.removeChild(successMessage), 3000);
      } else {
        throw new Error('Failed to find matches');
      }
    } catch (error) {
      console.error('Error finding matches:', error);
      const errorMessage = document.createElement('div');
      errorMessage.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
      errorMessage.textContent = 'Failed to find matches. Please try again.';
      document.body.appendChild(errorMessage);
      setTimeout(() => document.body.removeChild(errorMessage), 3000);
    } finally {
      setLoadingMatch(null);
    }
  }

  const handleMaterialApproved = async (materialId: string) => {
    // Remove from pending and add to approved
    setPendingMaterials(prev => prev.filter(m => m.id !== materialId));
    await loadMarketplaceData(); // Refresh data
  };

  const handleMaterialRejected = async (materialId: string) => {
    // Remove from pending
    setPendingMaterials(prev => prev.filter(m => m.id !== materialId));
  };

  const handleFeedbackSubmitted = async () => {
    // Refresh data after feedback
    await loadMarketplaceData();
  };

  function calculateRating(material: Material): number {
    let rating = 70;
    
    if (material.quantity > 100) rating += 10;
    if (material.description.length > 50) rating += 5;
    if (material.match_score && material.match_score > 80) rating += 10;
    
    const companyAge = new Date().getTime() - new Date(material.created_at).getTime();
    const daysSinceCreation = companyAge / (1000 * 60 * 60 * 24);
    if (daysSinceCreation > 30) rating += 5;
    
    return Math.min(100, rating);
  }

  if (loading) {
    return (
      <div className={`min-h-screen ${isDark ? 'bg-slate-900' : 'bg-gray-50'} flex items-center justify-center`}>
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Main Content */}
      <div className="container mx-auto px-6 py-8">
        {/* Header Section */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Industrial Symbiosis Marketplace</h1>
              <p className="text-gray-400">Connect with companies worldwide to exchange waste, resources, and create circular economies</p>
            </div>
            <button
              onClick={() => navigate('/marketplace/add-material')}
              className="bg-emerald-500 text-white px-6 py-3 rounded-lg hover:bg-emerald-600 transition flex items-center space-x-2"
            >
              <Plus className="w-5 h-5" />
              <span>Add Material</span>
            </button>
          </div>

          {/* Search and Filters */}
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  placeholder="Search materials..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                />
              </div>
              
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value as any)}
                className="px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              >
                <option value="all">All Types</option>
                <option value="waste">Waste</option>
                <option value="requirement">Requirement</option>
              </select>
              
              <input
                type="text"
                placeholder="Enter location..."
                value={locationFilter}
                onChange={(e) => setLocationFilter(e.target.value)}
                className="px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              />
              
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center justify-center space-x-2 px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white hover:bg-slate-600 transition"
              >
                <FilterIcon className="w-5 h-5" />
                <span>Filters</span>
              </button>
            </div>

            {/* Advanced Filters */}
            {showFilters && (
              <div className="mt-4 pt-4 border-t border-slate-700">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Quantity</label>
                    <input
                      type="number"
                      placeholder="Min Quantity..."
                      value={quantityFilter}
                      onChange={(e) => setQuantityFilter(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Date From</label>
                    <input
                      type="date"
                      value={dateFilter}
                      onChange={(e) => setDateFilter(e.target.value)}
                      className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                  </div>
                  
                  <button
                    onClick={() => setShowFilters(false)}
                    className="flex items-center justify-center space-x-2 px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg text-white hover:bg-slate-600 transition"
                  >
                    <X className="w-5 h-5" />
                    <span>Close Filters</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <div className="flex items-center space-x-3">
              <Factory className="w-8 h-8 text-emerald-400" />
              <div>
                <p className="text-sm text-gray-400">Total Materials</p>
                <p className="text-2xl font-bold text-white">{filteredMaterials.length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <div className="flex items-center space-x-3">
              <Users className="w-8 h-8 text-blue-400" />
              <div>
                <p className="text-sm text-gray-400">Active Companies</p>
                <p className="text-2xl font-bold text-white">{companies.length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <div className="flex items-center space-x-3">
              <Recycle className="w-8 h-8 text-green-400" />
              <div>
                <p className="text-sm text-gray-400">Waste Exchanges</p>
                <p className="text-2xl font-bold text-white">{materials.filter(m => m.type === 'waste').length}</p>
              </div>
            </div>
          </div>
          
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <div className="flex items-center space-x-3">
              <TrendingUp className="w-8 h-8 text-purple-400" />
              <div>
                <p className="text-sm text-gray-400">This Month</p>
                <p className="text-2xl font-bold text-white">{materials.filter(m => new Date(m.created_at).getMonth() === new Date().getMonth()).length}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Materials Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredMaterials.map((material) => (
            <div key={material.id} className="bg-slate-800 rounded-lg shadow-lg border border-slate-700 hover:shadow-xl transition-shadow">
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-2">
                    {material.type === 'waste' ? (
                      <Factory className="w-5 h-5 text-orange-500" />
                    ) : (
                      <Recycle className="w-5 h-5 text-blue-500" />
                    )}
                    <h3 className="text-lg font-semibold text-white">{material.material_name}</h3>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    material.status === 'pending'
                      ? 'bg-yellow-500 text-white'
                      : material.status === 'approved'
                        ? 'bg-green-500 text-white'
                        : 'bg-red-500 text-white'
                  }`}>
                    {material.status}
                  </span>
                </div>
                
                <p className="text-gray-300 mb-4 line-clamp-2">{material.description}</p>
                
                <div className="space-y-3 mb-4">
                  <div className="flex items-center space-x-2 text-sm">
                    <MapPin className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-300">{material.company?.location || 'Unknown Location'}</span>
                  </div>
                  
                  <div className="flex items-center space-x-2 text-sm">
                    <DollarSign className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-300">{material.quantity} {material.unit}</span>
                  </div>
                  
                  <div className="flex items-center space-x-2 text-sm">
                    <Calendar className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-300">{new Date(material.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <img 
                      src={material.company?.name === 'Unknown Company' ? 'https://via.placeholder.com/32x32' : `https://ui-avatars.com/api/?name=${material.company?.name}&background=0D8ABC&color=fff`} 
                      alt={material.company?.name || 'Unknown Company'}
                      className="w-8 h-8 rounded-full"
                    />
                    <span className="text-sm text-gray-300">{material.company?.name || 'Unknown Company'}</span>
                  </div>
                  
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleFavorite(material.id)}
                      disabled={loadingActions[`favorite-${material.id}`]}
                      className={`flex items-center space-x-1 px-3 py-1 rounded-md text-sm transition ${
                        favorites.has(material.id) 
                          ? 'bg-red-500 text-white hover:bg-red-600' 
                          : 'bg-slate-700 text-white hover:bg-slate-600'
                      }`}
                    >
                      <Heart className="w-4 h-4" />
                      <span>{favorites.has(material.id) ? 'Favorited' : 'Favorite'}</span>
                    </button>
                    <button
                      onClick={() => handleConnect(material.company_id)}
                      disabled={loadingActions[`connect-${material.company_id}`]}
                      className={`flex items-center space-x-1 px-3 py-1 rounded-md text-sm transition ${
                        connections.has(material.company_id)
                          ? 'bg-green-500 text-white hover:bg-green-600'
                          : 'bg-emerald-500 text-white hover:bg-emerald-600'
                      }`}
                    >
                      {loadingActions[`connect-${material.company_id}`] ? (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mx-auto"></div>
                      ) : connections.has(material.company_id) ? (
                        <Check className="w-4 h-4" />
                      ) : (
                        <Plus className="w-4 h-4" />
                      )}
                      <span>{connections.has(material.company_id) ? 'Connected' : 'Connect'}</span>
                    </button>
                    <button
                      onClick={() => handleChat(material.company_id)}
                      disabled={loadingActions[`chat-${material.company_id}`]}
                      className={`flex items-center space-x-1 px-3 py-1 rounded-md text-sm transition ${isDark ? 'bg-slate-700 text-white hover:bg-slate-600' : 'bg-emerald-500 text-white hover:bg-emerald-600'}`}
                    >
                      {loadingActions[`chat-${material.company_id}`] ? (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mx-auto"></div>
                      ) : (
                        <MessageSquare className="w-4 h-4" />
                      )}
                      <span>Chat</span>
                    </button>
                    <button
                      onClick={() => handleFindMatches(material)}
                      disabled={loadingMatch === material.id}
                      className="flex items-center space-x-1 px-3 py-1 rounded-md text-sm transition bg-purple-500 text-white hover:bg-purple-600"
                    >
                      {loadingMatch === material.id ? 'Matching...' : 'Find Matches'}
                      <Zap className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {Array.isArray(matchResults[material.id]) ? (
                  <div className="mt-2 p-2 bg-emerald-50 rounded text-sm">
                    <div className="font-semibold mb-1">Top Matches:</div>
                    {matchResults[material.id].length === 0 ? (
                      <div>No matches found.</div>
                    ) : (
                      matchResults[material.id].slice(0, 3).map((match: any, idx: number) => (
                        <div key={match.counterpart.id} className="mb-2">
                          <div>Counterpart: <b>{match.counterpart.company_id}</b></div>
                          <div>Score: <b>{match.result.revolutionary_score}</b></div>
                          <div>Quality: <b>{match.result.match_quality}</b></div>
                          <div>Sustainability: <b>{match.result.sustainability_score}</b></div>
                          <div>Trust: <b>{match.result.trust_status}</b></div>
                        </div>
                      ))
                    )}
                  </div>
                ) : matchResults[material.id]?.error ? (
                  <div className="mt-2 p-2 bg-red-50 rounded text-sm text-red-500">{matchResults[material.id].error}</div>
                ) : null}
              </div>
            </div>
          ))}
        </div>

        {/* Empty State */}
        {filteredMaterials.length === 0 && (
          <div className="text-center py-12">
            <Factory className="w-16 h-16 text-gray-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No materials found</h3>
            <p className="text-gray-400 mb-6">Try adjusting your search criteria or filters</p>
            <button
              onClick={() => {
                setSearchQuery('');
                setFilterType('all');
                setLocationFilter('');
                setQuantityFilter('');
                setDateFilter('');
              }}
              className="bg-emerald-500 text-white px-6 py-3 rounded-lg hover:bg-emerald-600 transition"
            >
              Clear Filters
            </button>
          </div>
        )}
      </div>

      {/* Material Form Modal */}
      {showMaterialForm && (
        <MaterialForm 
          type={showMaterialForm} 
          onClose={() => {
            setShowMaterialForm(null);
            loadMarketplaceData(); // Refresh data after adding material
          }} 
        />
      )}
    </div>
  );
}
