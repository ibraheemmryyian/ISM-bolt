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
  Eye
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { MaterialForm } from './MaterialForm';

interface MarketplaceProps {
  onNavigate: (view: 'home' | 'dashboard' | 'marketplace' | 'admin') => void;
  onSignOut: () => void;
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

export function Marketplace({ onNavigate, onSignOut }: MarketplaceProps) {
  const [activeTab, setActiveTab] = useState<'materials' | 'companies'>('materials');
  const [materials, setMaterials] = useState<Material[]>([]);
  const [companies, setCompanies] = useState<Company[]>([]);
  const [filteredMaterials, setFilteredMaterials] = useState<Material[]>([]);
  const [filteredCompanies, setFilteredCompanies] = useState<Company[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'waste' | 'requirement'>('all');
  const [showMaterialForm, setShowMaterialForm] = useState<'waste' | 'requirement' | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadMarketplaceData();
  }, []);

  useEffect(() => {
    filterMaterials();
  }, [materials, searchQuery, filterType]);

  useEffect(() => {
    filterCompanies();
  }, [companies, searchQuery]);

  async function loadMarketplaceData() {
    try {
      // Load materials with company info
      const { data: materialsData, error: materialsError } = await supabase
        .from('materials')
        .select(`
          *,
          companies!inner(name)
        `)
        .order('created_at', { ascending: false });

      if (materialsError) throw materialsError;

      // Add mock data for demonstration
      const enhancedMaterials = materialsData?.map(material => ({
        ...material,
        company: { name: material.companies.name, location: 'San Francisco, CA' },
        distance: `${Math.floor(Math.random() * 50) + 1}km`,
        match_score: Math.floor(Math.random() * 30) + 70
      })) || [];

      setMaterials(enhancedMaterials);

      // Load companies with profiles
      const { data: companiesData, error: companiesError } = await supabase
        .from('companies')
        .select(`
          *,
          company_profiles(role, location, organization_type, materials_of_interest)
        `)
        .order('created_at', { ascending: false });

      if (companiesError) throw companiesError;

      const enhancedCompanies = companiesData?.map(company => ({
        id: company.id,
        name: company.name,
        role: company.company_profiles?.[0]?.role || 'user',
        location: company.company_profiles?.[0]?.location,
        organization_type: company.company_profiles?.[0]?.organization_type,
        materials_of_interest: company.company_profiles?.[0]?.materials_of_interest,
        sustainability_score: Math.floor(Math.random() * 30) + 70
      })) || [];

      setCompanies(enhancedCompanies);

    } catch (error) {
      console.error('Error loading marketplace data:', error);
    } finally {
      setLoading(false);
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

    setFilteredMaterials(filtered);
  }

  function filterCompanies() {
    let filtered = companies;

    if (searchQuery) {
      filtered = filtered.filter(company =>
        company.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        company.organization_type?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        company.materials_of_interest?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    setFilteredCompanies(filtered);
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
                  onClick={() => onNavigate('dashboard')}
                  className="text-gray-500 hover:text-gray-700 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Dashboard
                </button>
                <button
                  onClick={() => onNavigate('marketplace')}
                  className="text-emerald-600 border-b-2 border-emerald-600 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Marketplace
                </button>
                <button
                  onClick={() => onNavigate('home')}
                  className="text-gray-500 hover:text-gray-700 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Home
                </button>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button className="text-gray-400 hover:text-gray-500">
                <Bell className="h-6 w-6" />
              </button>
              <button className="text-gray-400 hover:text-gray-500">
                <MessageSquare className="h-6 w-6" />
              </button>
              <button
                onClick={onSignOut}
                className="text-gray-500 hover:text-gray-700 text-sm font-medium"
              >
                Sign Out
              </button>
            </div>
          </div>
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

        {/* Search and Filters */}
        <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search materials, companies, or keywords..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <div className="flex space-x-4">
              <select
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                value={filterType}
                onChange={(e) => setFilterType(e.target.value as 'all' | 'waste' | 'requirement')}
              >
                <option value="all">All Types</option>
                <option value="waste">Available Materials</option>
                <option value="requirement">Material Needs</option>
              </select>
              <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
                <Filter className="h-5 w-5" />
                <span>More Filters</span>
              </button>
            </div>
          </div>
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
                  <button className="text-gray-400 hover:text-red-500">
                    <Heart className="h-5 w-5" />
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
                  <button className="flex-1 bg-emerald-500 text-white py-2 px-4 rounded-lg hover:bg-emerald-600 transition text-sm font-medium">
                    Connect
                  </button>
                  <button className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition">
                    <Eye className="h-4 w-4" />
                  </button>
                </div>
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
                  <button className="text-gray-400 hover:text-red-500">
                    <Heart className="h-5 w-5" />
                  </button>
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
                  <button className="flex-1 bg-emerald-500 text-white py-2 px-4 rounded-lg hover:bg-emerald-600 transition text-sm font-medium">
                    Connect
                  </button>
                  <button className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition">
                    <MessageSquare className="h-4 w-4" />
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
