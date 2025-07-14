import React, { useState, useEffect } from 'react';
import { 
  Bell, 
  Filter, 
  Heart, 
  MapPin, 
  MessageSquare, 
  Plus, 
  Search, 
  Star, 
  Workflow,
  Factory,
  Recycle,
  Users,
  Calendar,
  TrendingUp,
  Eye,
  Check
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { messagingService } from '../lib/messagingService';
import { MaterialForm } from './MaterialForm';
import { useNavigate, Link } from 'react-router-dom';

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
  const [activeTab, setActiveTab] = useState<'materials' | 'companies'>('materials');
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
        await Promise.all([
          loadMarketplaceData(),
          loadUserConnections(user.id),
          loadUserFavorites(user.id)
        ]);
      } else {
        await loadMarketplaceData();
      }
    } catch (error) {
      console.error('Error loading user and data:', error);
    } finally {
      setLoading(false);
    }
  }

  async function loadUserConnections(userId: string) {
    try {
      const following = await messagingService.getFollowing(userId);
      const connectionSet = new Set(following.map(conn => conn.following_id));
      setConnections(connectionSet);
    } catch (error) {
      console.error('Error loading connections:', error);
    }
  }

  async function loadUserFavorites(userId: string) {
    try {
      const userFavorites = await messagingService.getFavorites(userId);
      const favoriteSet = new Set(userFavorites.map(fav => fav.material_id));
      setFavorites(favoriteSet);
    } catch (error) {
      console.error('Error loading favorites:', error);
    }
  }

  async function loadMarketplaceData() {
    try {
      // Show ALL materials in the marketplace
      const { data: materialsData, error: materialsError } = await supabase
        .from('materials')
        .select('*');

      if (materialsError) throw materialsError;

      // Add mock data for demonstration
      const enhancedMaterials = materialsData?.map(material => ({
        ...material,
        company: { 
          name: material.companies?.name || 'Unknown Company', 
          location: 'San Francisco, CA' 
        },
        distance: `${Math.floor(Math.random() * 50) + 1}km`,
        match_score: Math.floor(Math.random() * 30) + 70
      })) || [];

      setMaterials(enhancedMaterials);

      // Show ALL companies in the marketplace
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
        sustainability_score: Math.floor(Math.random() * 30) + 70
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
      alert('Please log in to connect with companies');
      return;
    }

    // Prevent connecting to yourself
    if (companyId === currentUserId) {
      alert('You cannot connect to yourself');
      return;
    }

    // Check if already connected
      if (connections.has(companyId)) {
      alert('You are already connected to this company');
      return;
    }

    setLoadingActions(prev => ({ ...prev, [`connect-${companyId}`]: true }));

    try {
      // Create connection request
      const { error } = await supabase
        .from('connections')
        .insert({
          requester_id: currentUserId,
          recipient_id: companyId,
          status: 'pending'
        });

      if (error) throw error;

      // Update local state
        setConnections(prev => new Set([...prev, companyId]));
      
      // Show success message in UI instead of alert
      const successMessage = document.createElement('div');
      successMessage.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
      successMessage.textContent = 'Connection request sent successfully!';
      document.body.appendChild(successMessage);
      setTimeout(() => document.body.removeChild(successMessage), 3000);
      
    } catch (error) {
      console.error('Error connecting:', error);
      const errorMessage = document.createElement('div');
      errorMessage.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
      errorMessage.textContent = 'Failed to send connection request';
      document.body.appendChild(errorMessage);
      setTimeout(() => document.body.removeChild(errorMessage), 3000);
    } finally {
      setLoadingActions(prev => ({ ...prev, [`connect-${companyId}`]: false }));
    }
  }

  async function handleChat(companyId: string) {
    if (!currentUserId) {
      alert('Please log in to start chatting');
      return;
    }

    // Prevent chatting with yourself
    if (companyId === currentUserId) {
      alert('You cannot chat with yourself');
      return;
    }

    setLoadingActions(prev => ({ ...prev, [`chat-${companyId}`]: true }));

    try {
      // Create or get existing chat conversation
      const { data: existingChat, error: chatError } = await supabase
        .from('conversations')
        .select('id')
        .or(`participant1.eq.${currentUserId},participant2.eq.${currentUserId}`)
        .or(`participant1.eq.${companyId},participant2.eq.${companyId}`)
        .single();

      let conversationId;
      
      if (existingChat) {
        conversationId = existingChat.id;
      } else {
        // Create new conversation
        const { data: newChat, error: createError } = await supabase
          .from('conversations')
          .insert({
            participant1: currentUserId,
            participant2: companyId,
            created_at: new Date().toISOString()
          })
          .select('id')
          .single();
        
        if (createError) throw createError;
        conversationId = newChat.id;
      }

      // Navigate to chat with conversation ID
      navigate(`/chats?conversation=${conversationId}`);
      
    } catch (error) {
      console.error('Error starting chat:', error);
      const errorMessage = document.createElement('div');
      errorMessage.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50';
      errorMessage.textContent = 'Failed to start chat';
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
      const isFavorited = favorites.has(materialId);
      
      if (isFavorited) {
        // Remove from favorites
        await supabase
          .from('favorites')
          .delete()
          .eq('company_id', currentUserId)
          .eq('material_id', materialId);
        
        setFavorites(prev => {
          const newSet = new Set(prev);
          newSet.delete(materialId);
          return newSet;
        });
      } else {
        // Add to favorites
        await supabase
          .from('favorites')
          .insert({
            company_id: currentUserId,
            material_id: materialId
          });
        
        setFavorites(prev => new Set([...prev, materialId]));
      }
    } catch (error) {
      console.error('Error toggling favorite:', error);
      alert('Failed to update favorite');
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
      // Call AI matching endpoint with specific material
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

  // Calculate rating based on material quality and company reputation
  function calculateRating(material: Material): number {
    let rating = 70; // Base rating
    
    // Material quality factors
    if (material.quantity > 100) rating += 10;
    if (material.description.length > 50) rating += 5;
    if (material.match_score && material.match_score > 80) rating += 10;
    
    // Company reputation (mock)
    const companyAge = new Date().getTime() - new Date(material.created_at).getTime();
    const daysSinceCreation = companyAge / (1000 * 60 * 60 * 24);
    if (daysSinceCreation > 30) rating += 5;
    
    return Math.min(100, rating);
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Nav Bar */}
      <nav className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between h-16">
          <div className="flex items-center space-x-8">
            <button onClick={() => navigate('/dashboard')} className="text-gray-700 hover:text-emerald-600 font-semibold transition">Dashboard</button>
            <button className="text-emerald-600 font-bold border-b-2 border-emerald-600 pb-1">Marketplace</button>
            <button onClick={() => navigate('/onboarding')} className="text-gray-700 hover:text-emerald-600 font-semibold transition">AI Onboarding</button>
          </div>
          {onSignOut && (
            <button onClick={onSignOut} className="text-gray-500 hover:text-red-600 transition">Sign Out</button>
          )}
        </div>
      </nav>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Marketplace</h1>
            <p className="text-gray-600 mt-1">
              Discover materials, connect with partners, and build sustainable relationships
            </p>
          </div>
          <button
            onClick={() => setShowMaterialForm('waste')}
            className="bg-emerald-500 text-white px-6 py-3 rounded-lg hover:bg-emerald-600 transition flex items-center space-x-2"
          >
            <Plus className="h-5 w-5" />
            <span>List Material</span>
          </button>
        </div>

        {/* Filters */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
              <input
                type="text"
                  placeholder="Search materials..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              />
            </div>
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value as 'all' | 'waste' | 'requirement')}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              >
                <option value="all">All Types</option>
                <option value="waste">Waste</option>
                <option value="requirement">Requirement</option>
              </select>
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              <Filter className="h-4 w-4" />
              <span>More Filters</span>
            </button>
          </div>

          {showFilters && (
            <div className="bg-gray-50 p-4 rounded-lg space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
                  <input
                    type="text"
                    placeholder="Filter by location..."
                    value={locationFilter}
                    onChange={(e) => setLocationFilter(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Min Quantity</label>
                  <input
                    type="number"
                    placeholder="Minimum quantity..."
                    value={quantityFilter}
                    onChange={(e) => setQuantityFilter(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Date From</label>
                  <input
                    type="date"
                    value={dateFilter}
                    onChange={(e) => setDateFilter(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                  />
                </div>
              </div>
              <div className="flex justify-end space-x-2">
                <button
                  onClick={() => {
                    setLocationFilter('');
                    setQuantityFilter('');
                    setDateFilter('');
                  }}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800"
                >
                  Clear Filters
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 mb-8">
          <button
            onClick={() => setActiveTab('materials')}
            className={`px-6 py-3 rounded-lg font-medium transition ${
              activeTab === 'materials'
                ? 'bg-emerald-500 text-white'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            Materials ({filteredMaterials.length})
          </button>
          <button
            onClick={() => setActiveTab('companies')}
            className={`px-6 py-3 rounded-lg font-medium transition ${
              activeTab === 'companies'
                ? 'bg-emerald-500 text-white'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            Companies ({filteredCompanies.length})
          </button>
        </div>

        {/* Content */}
        {activeTab === 'materials' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredMaterials.map((material) => (
              <div key={material.id} className="bg-white rounded-xl shadow-sm p-6 hover:shadow-md transition">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-2">
                    {material.type === 'waste' ? (
                      <Factory className="h-5 w-5 text-orange-500" />
                    ) : (
                      <Recycle className="h-5 w-5 text-blue-500" />
                    )}
                    <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                      material.type === 'waste'
                        ? 'bg-orange-100 text-orange-800'
                        : 'bg-blue-100 text-blue-800'
                    }`}>
                      {material.type === 'waste' ? 'Available' : 'Needed'}
                    </span>
                  </div>
                  <button 
                    onClick={() => handleFavorite(material.id)}
                    disabled={loadingActions[`favorite-${material.id}`]}
                    className={`transition-colors ${
                      favorites.has(material.id) 
                        ? 'text-red-500 hover:text-red-600' 
                        : 'text-gray-400 hover:text-red-500'
                    }`}
                  >
                    <Heart className={`h-5 w-5 ${favorites.has(material.id) ? 'fill-current' : ''}`} />
                  </button>
                </div>

                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {material.material_name}
                </h3>
                
                <div className="flex items-center space-x-4 text-sm text-gray-600 mb-3">
                  <span className="font-medium">
                    {material.quantity} {material.unit}
                  </span>
                  <span className="flex items-center space-x-1">
                    <MapPin className="h-4 w-4" />
                    <span>{material.distance}</span>
                  </span>
                </div>

                <p className="text-gray-600 text-sm mb-4 line-clamp-2">
                  {material.description}
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center">
                      <span className="text-xs font-medium text-emerald-800">
                        {material.company?.name.charAt(0)}
                      </span>
                    </div>
                    <span className="text-sm text-gray-600">{material.company?.name}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Star className="h-4 w-4 text-yellow-500" />
                    <span className="text-sm text-gray-600">{material.match_score}%</span>
                  </div>
                </div>

                <div className="flex space-x-2 mt-4">
                  <button 
                    onClick={() => handleConnect(material.company_id)}
                    disabled={loadingActions[`connect-${material.company_id}`]}
                    className={`flex-1 py-2 px-4 rounded-lg transition text-sm font-medium ${
                      connections.has(material.company_id)
                        ? 'bg-green-100 text-green-700 hover:bg-green-200'
                        : 'bg-emerald-500 text-white hover:bg-emerald-600'
                    }`}
                  >
                    {loadingActions[`connect-${material.company_id}`] ? (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mx-auto"></div>
                    ) : connections.has(material.company_id) ? (
                      <div className="flex items-center justify-center space-x-1">
                        <Check className="h-4 w-4" />
                        <span>Connected</span>
                      </div>
                    ) : (
                      'Connect'
                    )}
                  </button>
                  <button 
                    onClick={() => handleChat(material.company_id)}
                    disabled={loadingActions[`chat-${material.company_id}`]}
                    className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition"
                  >
                    {loadingActions[`chat-${material.company_id}`] ? (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-emerald-500"></div>
                    ) : (
                      <MessageSquare className="h-4 w-4" />
                    )}
                  </button>
                  <button
                    className="px-4 py-2 border border-emerald-500 text-emerald-600 rounded-lg hover:bg-emerald-50 transition text-sm font-medium"
                    onClick={() => handleFindMatches(material)}
                    disabled={loadingMatch === material.id}
                  >
                    {loadingMatch === material.id ? 'Matching...' : 'Find Matches'}
                  </button>
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
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredCompanies.map((company) => (
              <div key={company.id} className="bg-white rounded-xl shadow-sm p-6 hover:shadow-md transition">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-12 h-12 bg-emerald-100 rounded-full flex items-center justify-center">
                      <span className="text-lg font-bold text-emerald-800">
                        {company.name.charAt(0)}
                      </span>
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{company.name}</h3>
                      <p className="text-sm text-gray-600 capitalize">{company.organization_type}</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex items-center space-x-2 text-sm text-gray-600">
                    <MapPin className="h-4 w-4" />
                    <span>{company.location || 'Location not specified'}</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-600">
                    <Users className="h-4 w-4" />
                    <span className="capitalize">{company.role}</span>
                  </div>
                </div>

                <div className="mb-4">
                  <p className="text-sm text-gray-600">
                    <span className="font-medium">Interests:</span> {company.materials_of_interest || 'Various materials'}
                  </p>
                </div>

                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-1">
                    <TrendingUp className="h-4 w-4 text-green-500" />
                    <span className="text-sm text-gray-600">
                      {company.sustainability_score}% sustainability score
                    </span>
                  </div>
                </div>

                <div className="flex space-x-2">
                  <button 
                    onClick={() => handleConnect(company.id)}
                    disabled={loadingActions[`connect-${company.id}`]}
                    className={`flex-1 py-2 px-4 rounded-lg transition text-sm font-medium ${
                      connections.has(company.id)
                        ? 'bg-green-100 text-green-700 hover:bg-green-200'
                        : 'bg-emerald-500 text-white hover:bg-emerald-600'
                    }`}
                  >
                    {loadingActions[`connect-${company.id}`] ? (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mx-auto"></div>
                    ) : connections.has(company.id) ? (
                      <div className="flex items-center justify-center space-x-1">
                        <Check className="h-4 w-4" />
                        <span>Connected</span>
                      </div>
                    ) : (
                      'Connect'
                    )}
                  </button>
                  <button 
                    onClick={() => handleChat(company.id)}
                    disabled={loadingActions[`chat-${company.id}`]}
                    className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition"
                  >
                    {loadingActions[`chat-${company.id}`] ? (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-emerald-500"></div>
                    ) : (
                    <MessageSquare className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Empty State */}
        {((activeTab === 'materials' && filteredMaterials.length === 0) ||
          (activeTab === 'companies' && filteredCompanies.length === 0)) && (
          <div className="text-center py-12">
            <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Search className="h-12 w-12 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No {activeTab} found
            </h3>
            <p className="text-gray-600 mb-6">
              Try adjusting your search criteria or filters
            </p>
            <button
              onClick={() => setShowMaterialForm('waste')}
              className="bg-emerald-500 text-white px-6 py-3 rounded-lg hover:bg-emerald-600 transition"
            >
              List Your First Material
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
