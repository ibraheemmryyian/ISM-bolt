import React, { useState } from 'react';
import { supabase } from '../lib/supabase';
import { X, Loader2, Trash2, FlaskConical } from 'lucide-react';

interface OnboardingFormProps {
  onClose: () => void;
}

export function OnboardingForm({ onClose }: OnboardingFormProps) {
  const [userType, setUserType] = useState<'business' | 'researcher' | ''>('');
  const [formData, setFormData] = useState({
    company_name: '',
    location: '',
    // Business fields
    current_waste: '',
    waste_quantity: '',
    waste_unit: '',
    waste_frequency: '',
    // Researcher fields
    research_focus: '',
    research_institution: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('Not authenticated');

      // Save to companies table instead of company_profiles
      const { error: err } = await supabase.from('companies').insert([
        {
          user_id: user.id,
          name: formData.company_name || formData.research_institution,
          location: formData.location,
          user_type: userType,
          current_waste_management: userType === 'business' ? formData.current_waste : null,
          waste_quantity: userType === 'business' ? formData.waste_quantity : null,
          waste_unit: userType === 'business' ? formData.waste_unit : null,
          waste_frequency: userType === 'business' ? formData.waste_frequency : null,
          process_description: userType === 'business' ? 
            `Waste: ${formData.current_waste}, Quantity: ${formData.waste_quantity} ${formData.waste_unit} per ${formData.waste_frequency}` :
            `Research Focus: ${formData.research_focus}`,
          onboarding_completed: true
        },
      ]);

      if (err) throw err;
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to save profile. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (!userType) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-xl p-6 w-full max-w-md relative">
          <button
            onClick={onClose}
            className="absolute right-4 top-4 text-gray-500 hover:text-gray-700"
          >
            <X className="h-5 w-5" />
          </button>

          <h2 className="text-2xl font-bold mb-6 text-center">Welcome to Industrial Symbiosis</h2>
          <p className="text-gray-600 mb-6 text-center">What brings you here today?</p>

          <div className="space-y-4">
            <button
              onClick={() => setUserType('business')}
              className="w-full p-4 border-2 border-gray-200 rounded-lg hover:border-emerald-500 hover:bg-emerald-50 transition-colors text-left"
            >
              <div className="flex items-center space-x-3">
                <Trash2 className="h-8 w-8 text-emerald-600" />
                <div>
                  <h3 className="font-semibold text-lg">I'm a Business</h3>
                  <p className="text-sm text-gray-600">I have waste materials to exchange or need materials</p>
                </div>
              </div>
            </button>

            <button
              onClick={() => setUserType('researcher')}
              className="w-full p-4 border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors text-left"
            >
              <div className="flex items-center space-x-3">
                <FlaskConical className="h-8 w-8 text-blue-600" />
                <div>
                  <h3 className="font-semibold text-lg">I'm a Researcher</h3>
                  <p className="text-sm text-gray-600">I'm researching circular economy and industrial symbiosis</p>
                </div>
              </div>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-xl p-6 w-full max-w-md relative">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 text-gray-500 hover:text-gray-700"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="flex items-center space-x-2 mb-6">
          {userType === 'business' ? (
            <Trash2 className="h-6 w-6 text-emerald-600" />
          ) : (
            <FlaskConical className="h-6 w-6 text-blue-600" />
          )}
          <h2 className="text-2xl font-bold">
            {userType === 'business' ? 'Business Profile' : 'Researcher Profile'}
          </h2>
        </div>

        {error && (
          <div className="bg-red-100 text-red-700 p-3 rounded mb-4 text-center text-sm">{error}</div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {userType === 'business' ? 'Company Name' : 'Institution Name'}
            </label>
            <input
              type="text"
              required
              placeholder={userType === 'business' ? 'Your company name' : 'Your research institution'}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              value={userType === 'business' ? formData.company_name : formData.research_institution}
              onChange={(e) => setFormData({ 
                ...formData, 
                [userType === 'business' ? 'company_name' : 'research_institution']: e.target.value 
              })}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Location
            </label>
            <input
              type="text"
              required
              placeholder="City, Country"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              value={formData.location}
              onChange={(e) => setFormData({ ...formData, location: e.target.value })}
            />
          </div>

          {userType === 'business' ? (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  What waste materials do you currently have?
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g., Plastic scraps, metal shavings, organic waste"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                  value={formData.current_waste}
                  onChange={(e) => setFormData({ ...formData, current_waste: e.target.value })}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Quantity & Frequency
                </label>
                <div className="flex space-x-3">
                  <input
                    type="text"
                    required
                    placeholder="e.g., 1000"
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                    value={formData.waste_quantity}
                    onChange={(e) => setFormData({ ...formData, waste_quantity: e.target.value })}
                  />
                  <select
                    required
                    className="w-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                    value={formData.waste_unit}
                    onChange={(e) => setFormData({ ...formData, waste_unit: e.target.value })}
                  >
                    <option value="">Unit</option>
                    <option value="kg">kg</option>
                    <option value="tons">tons</option>
                    <option value="liters">liters</option>
                    <option value="cubic meters">m³</option>
                    <option value="pieces">pieces</option>
                  </select>
                  <select
                    required
                    className="w-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                    value={formData.waste_frequency}
                    onChange={(e) => setFormData({ ...formData, waste_frequency: e.target.value })}
                  >
                    <option value="">Per</option>
                    <option value="daily">Day</option>
                    <option value="weekly">Week</option>
                    <option value="monthly">Month</option>
                    <option value="quarterly">Quarter</option>
                    <option value="yearly">Year</option>
                  </select>
                </div>
                <p className="text-sm text-gray-500 mt-1">
                  Example: 1000 kg per month
                </p>
              </div>
            </>
          ) : (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Research Focus
              </label>
              <textarea
                required
                rows={3}
                placeholder="What aspects of industrial symbiosis are you researching?"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                value={formData.research_focus}
                onChange={(e) => setFormData({ ...formData, research_focus: e.target.value })}
              />
            </div>
          )}

          <div className="flex space-x-3 pt-4">
            <button
              type="button"
              onClick={() => setUserType('')}
              className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
            >
              Back
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 bg-emerald-500 text-white py-2 px-4 rounded-lg hover:bg-emerald-600 transition disabled:opacity-50"
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Saving...
                </div>
              ) : (
                'Complete Setup'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}