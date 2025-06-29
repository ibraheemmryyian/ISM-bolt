import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { 
  CheckCircle, 
  Edit, 
  Trash2, 
  Plus, 
  Brain, 
  AlertCircle, 
  ArrowRight,
  Loader2,
  Sparkles,
  Package,
  Recycle,
  Eye,
  Save
} from 'lucide-react';

interface Listing {
  id?: string;
  name: string;
  type: 'waste' | 'requirement';
  quantity: string | number;
  unit: string;
  description: string;
  ai_generated?: boolean;
  confidence_score?: number;
}

interface ReviewAIListingsProps {
  onConfirm: () => void;
}

export function ReviewAIListings({ onConfirm }: ReviewAIListingsProps) {
  const location = useLocation();
  const navigate = useNavigate();
  
  // Handle the backend response structure
  const responseData = location.state && (location.state as any);
  const aiListings = responseData?.listings || responseData?.aiListings || [];
  const matches = responseData?.matches || [];
  
  // Fallback sample data if no AI listings received
  const fallbackListings: Listing[] = [
    {
      id: 'sample-1',
      name: 'Concrete Debris',
      type: 'waste',
      quantity: '500',
      unit: 'tons',
      description: 'Crushed concrete from demolition projects, suitable for recycling into new construction materials',
      ai_generated: true,
      confidence_score: 0.85
    },
    {
      id: 'sample-2',
      name: 'Steel Scraps',
      type: 'waste',
      quantity: '200',
      unit: 'tons',
      description: 'Recyclable steel materials from manufacturing processes, high-grade material for steel production',
      ai_generated: true,
      confidence_score: 0.92
    },
    {
      id: 'sample-3',
      name: 'Raw Materials',
      type: 'requirement',
      quantity: '1000',
      unit: 'tons',
      description: 'General raw materials for production processes, seeking sustainable sourcing options',
      ai_generated: true,
      confidence_score: 0.78
    }
  ];
  
  const initialListings = aiListings.length > 0 ? aiListings : fallbackListings;
  const [listings, setListings] = useState<Listing[]>(initialListings);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [showAIInfo, setShowAIInfo] = useState(true);

  console.log('ReviewAIListings component mounted');
  console.log('Response data received:', responseData);
  console.log('AI Listings extracted:', aiListings);
  console.log('Current listings state:', listings);

  useEffect(() => {
    console.log('ReviewAIListings useEffect - component mounted');
    console.log('Location state:', location.state);
    console.log('Listings count:', listings.length);
  }, []);

  function handleEdit(index: number, field: keyof Listing, value: string) {
    const updated = [...listings];
    if (field === 'type') {
      (updated[index] as any)[field] = value as Listing['type'];
    } else {
      (updated[index] as any)[field] = value;
    }
    setListings(updated);
  }

  function handleDelete(index: number) {
    const updated = listings.filter((_, i) => i !== index);
    setListings(updated);
  }

  function handleAdd() {
    const newListing: Listing = {
      name: '',
      type: 'waste',
      quantity: '',
      unit: 'tons',
      description: '',
      ai_generated: false
    };
    setListings([...listings, newListing]);
    setEditingIndex(listings.length);
  }

  function startEditing(index: number) {
    setEditingIndex(index);
  }

  function stopEditing() {
    setEditingIndex(null);
  }

  async function handleConfirm() {
    setLoading(true);
    setError('');
    try {
      // Filter out empty listings
      const validListings = listings.filter(listing => 
        listing.name.trim() && 
        String(listing.quantity).trim() && 
        listing.description.trim()
      );

      if (validListings.length === 0) {
        throw new Error('Please add at least one valid listing');
      }

      const res = await fetch('/api/save-listings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          listings: validListings,
          userId: 'current-user-id' // This should come from auth context
        })
      });
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to save listings');
      }
      
      setLoading(false);
      onConfirm();
      navigate('/dashboard');
    } catch (err: any) {
      setLoading(false);
      setError(err.message || 'Failed to save listings.');
    }
  }

  function handleSkip() {
    navigate('/dashboard');
  }

  const aiGeneratedCount = listings.filter(l => l.ai_generated).length;
  const userAddedCount = listings.filter(l => !l.ai_generated).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-8">
      <div className="max-w-6xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <Brain className="h-8 w-8 text-emerald-500" />
            <h1 className="text-3xl font-bold text-gray-900">AI-Generated Material Listings</h1>
          </div>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Our AI has analyzed your company information and generated personalized material listings. 
            Review, edit, or add more before continuing to your dashboard.
          </p>
        </div>

        {/* --- INSTANT MATCHES SECTION --- */}
        {matches.length > 0 && (
          <div className="mb-8 bg-gradient-to-r from-emerald-50 to-blue-50 border border-emerald-200 rounded-xl p-6">
            <h2 className="text-xl font-bold text-emerald-900 mb-2">Your Instant AI Matches</h2>
            <p className="text-emerald-800 text-sm mb-4">
              These are your best matches, generated instantly by our AI. Match scores indicate compatibility.
            </p>
            <ul className="divide-y divide-emerald-200">
              {matches.map((match: any, idx: number) => (
                <li key={match.id || idx} className="py-3 flex flex-col md:flex-row md:items-center md:justify-between">
                  <div>
                    <span className="font-semibold text-emerald-900">Material ID:</span> {match.material_id} <br/>
                    <span className="font-semibold text-emerald-900">Matched Material ID:</span> {match.matched_material_id} <br/>
                    <span className="font-semibold text-emerald-900">Score:</span> <span className="font-bold text-emerald-700">{(match.match_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="mt-2 md:mt-0">
                    {/* Add more match details or explanations here if available */}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
        {/* --- END INSTANT MATCHES SECTION --- */}

        {/* AI Info Banner */}
        {showAIInfo && aiListings.length === 0 && (
          <div className="mb-6 bg-gradient-to-r from-emerald-50 to-blue-50 border border-emerald-200 rounded-xl p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3">
                <Sparkles className="h-6 w-6 text-emerald-600 mt-1" />
                <div>
                  <h3 className="font-semibold text-emerald-900 mb-1">AI Analysis Complete</h3>
                  <p className="text-emerald-800 text-sm">
                    Based on your company profile, we've generated {fallbackListings.length} material listings. 
                    These are intelligent suggestions that you can customize or expand upon.
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowAIInfo(false)}
                className="text-emerald-600 hover:text-emerald-800"
              >
                <Eye className="h-5 w-5" />
              </button>
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white rounded-xl shadow-sm p-6 text-center">
            <div className="flex items-center justify-center space-x-2 mb-2">
              <Brain className="h-5 w-5 text-emerald-500" />
              <span className="text-2xl font-bold text-gray-900">{aiGeneratedCount}</span>
            </div>
            <p className="text-sm text-gray-600">AI-Generated</p>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6 text-center">
            <div className="flex items-center justify-center space-x-2 mb-2">
              <Plus className="h-5 w-5 text-blue-500" />
              <span className="text-2xl font-bold text-gray-900">{userAddedCount}</span>
            </div>
            <p className="text-sm text-gray-600">User-Added</p>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6 text-center">
            <div className="flex items-center justify-center space-x-2 mb-2">
              <Package className="h-5 w-5 text-purple-500" />
              <span className="text-2xl font-bold text-gray-900">{listings.length}</span>
            </div>
            <p className="text-sm text-gray-600">Total Listings</p>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-600 mr-2" />
              <p className="text-red-800">{error}</p>
            </div>
          </div>
        )}
        
        {/* Listings */}
        <div className="bg-white rounded-xl shadow-sm p-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">Material Listings</h2>
            <button
              onClick={handleAdd}
              className="flex items-center space-x-2 bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition"
            >
              <Plus className="h-4 w-4" />
              <span>Add Listing</span>
            </button>
          </div>

          {listings.length === 0 ? (
            <div className="text-center py-12 border-2 border-dashed border-gray-200 rounded-lg">
              <Package className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Listings Available</h3>
              <p className="text-gray-600 mb-6">
                Start by adding your first material listing. You can add waste materials you produce 
                or requirements you need from other companies.
              </p>
              <button
                onClick={handleAdd}
                className="bg-emerald-500 text-white px-6 py-3 rounded-lg hover:bg-emerald-600 transition flex items-center space-x-2 mx-auto"
              >
                <Plus className="h-4 w-4" />
                <span>Add First Listing</span>
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {listings.map((listing, i) => (
                <div key={i} className="border border-gray-200 rounded-lg p-6 hover:border-gray-300 transition">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                        listing.type === 'waste' 
                          ? 'bg-orange-100 text-orange-800' 
                          : 'bg-blue-100 text-blue-800'
                      }`}>
                        {listing.type === 'waste' ? 'Waste' : 'Requirement'}
                      </span>
                      {listing.ai_generated && (
                        <span className="flex items-center space-x-1 px-2 py-1 bg-emerald-100 text-emerald-800 rounded-full text-xs">
                          <Brain className="h-3 w-3" />
                          <span>AI Generated</span>
                        </span>
                      )}
                      {listing.confidence_score && (
                        <span className="text-xs text-gray-500">
                          Confidence: {Math.round(listing.confidence_score * 100)}%
                        </span>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => startEditing(i)}
                        className="p-2 text-gray-400 hover:text-gray-600 transition"
                      >
                        <Edit className="h-4 w-4" />
                      </button>
                    <button
                      onClick={() => handleDelete(i)}
                        className="p-2 text-gray-400 hover:text-red-600 transition"
                    >
                        <Trash2 className="h-4 w-4" />
                    </button>
                    </div>
                  </div>
                  
                  {editingIndex === i ? (
                    <div className="space-y-4">
                      <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Material Name</label>
                      <input 
                            type="text"
                        value={listing.name} 
                            onChange={(e) => handleEdit(i, 'name', e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                            placeholder="Enter material name"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
                      <select 
                        value={listing.type} 
                            onChange={(e) => handleEdit(i, 'type', e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      >
                        <option value="waste">Waste</option>
                        <option value="requirement">Requirement</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Quantity</label>
                      <input 
                            type="text"
                        value={listing.quantity} 
                            onChange={(e) => handleEdit(i, 'quantity', e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                        placeholder="e.g., 500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Unit</label>
                      <input 
                            type="text"
                        value={listing.unit} 
                            onChange={(e) => handleEdit(i, 'unit', e.target.value)}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                            placeholder="e.g., tons, liters"
                          />
                        </div>
                      </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                    <textarea 
                      value={listing.description} 
                          onChange={(e) => handleEdit(i, 'description', e.target.value)}
                      rows={3}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      placeholder="Describe the material, its properties, and potential uses..."
                    />
                  </div>
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={stopEditing}
                          className="px-4 py-2 text-gray-600 hover:text-gray-800 transition"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={stopEditing}
                          className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition"
                        >
                          Save Changes
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-2">{listing.name}</h3>
                      <p className="text-gray-600 mb-3">{listing.description}</p>
                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <span>{listing.quantity} {listing.unit}</span>
                        <span>•</span>
                        <span className="capitalize">{listing.type}</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-between mt-8">
          <button
            onClick={handleSkip}
            className="px-6 py-3 text-gray-600 hover:text-gray-800 transition"
          >
            Skip for Now
          </button>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={handleAdd}
              className="flex items-center space-x-2 px-6 py-3 border border-emerald-500 text-emerald-600 rounded-lg hover:bg-emerald-50 transition"
            >
              <Plus className="h-4 w-4" />
              <span>Add More</span>
            </button>
            
              <button
                onClick={handleConfirm}
                disabled={loading || listings.length === 0}
              className="flex items-center space-x-2 px-8 py-3 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span>Saving...</span>
                </>
              ) : (
                <>
                  <Save className="h-5 w-5" />
                  <span>Save & Continue</span>
                  <ArrowRight className="h-5 w-5" />
                </>
              )}
              </button>
          </div>
        </div>
      </div>
    </div>
  );
} 