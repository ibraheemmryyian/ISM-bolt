import React from 'react';
import { Crown, Check, X, Zap, Brain, BarChart3, Download, Users } from 'lucide-react';
import { subscriptionService } from '../lib/subscriptionService';

interface SubscriptionUpgradeModalProps {
  isOpen: boolean;
  onClose: () => void;
  requiredFeature?: string;
  currentTier?: string;
}

export function SubscriptionUpgradeModal({ 
  isOpen, 
  onClose, 
  requiredFeature = 'AI features',
  currentTier = 'free'
}: SubscriptionUpgradeModalProps) {
  if (!isOpen) return null;

  const tiers = subscriptionService.getSubscriptionTiers();

  const getFeatureIcon = (feature: string) => {
    switch (feature.toLowerCase()) {
      case 'ai':
      case 'recommendations':
      case 'matching':
        return <Brain className="h-5 w-5" />;
      case 'analytics':
        return <BarChart3 className="h-5 w-5" />;
      case 'export':
        return <Download className="h-5 w-5" />;
      case 'bulk':
        return <Users className="h-5 w-5" />;
      default:
        return <Zap className="h-5 w-5" />;
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-2xl p-8 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="text-center mb-8">
          <Crown className="h-16 w-16 text-yellow-500 mx-auto mb-4" />
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Upgrade Your Plan</h2>
          <p className="text-gray-600">
            {requiredFeature} requires a Pro or Enterprise subscription
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {Object.entries(tiers).map(([key, tier]) => (
            <div
              key={key}
              className={`relative rounded-xl p-6 border-2 transition-all ${
                key === 'pro' 
                  ? 'border-purple-500 bg-purple-50 scale-105' 
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              {key === 'pro' && (
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <span className="bg-purple-500 text-white px-3 py-1 rounded-full text-sm font-semibold">
                    RECOMMENDED
                  </span>
                </div>
              )}
              
              <div className="text-center mb-4">
                <h3 className="text-xl font-bold text-gray-900 mb-1">{tier.name}</h3>
                <p className="text-2xl font-bold text-purple-600 mb-1">{tier.price}</p>
                {key === 'free' && <p className="text-sm text-gray-500">Forever free</p>}
              </div>

              <div className="space-y-3 mb-6">
                {tier.features.map((feature, index) => (
                  <div key={index} className="flex items-center space-x-3">
                    <Check className="h-5 w-5 text-green-500 flex-shrink-0" />
                    <span className="text-sm text-gray-700">{feature}</span>
                  </div>
                ))}
                
                {'limitations' in tier && tier.limitations?.map((limitation: string, index: number) => (
                  <div key={index} className="flex items-center space-x-3">
                    <X className="h-5 w-5 text-red-500 flex-shrink-0" />
                    <span className="text-sm text-gray-500">{limitation}</span>
                  </div>
                ))}
              </div>

              <button
                onClick={() => {
                  if (key === 'free') {
                    onClose();
                  } else {
                    // Handle upgrade logic
                    alert(`Upgrade to ${tier.name} - This would integrate with your payment processor`);
                  }
                }}
                className={`w-full py-3 px-4 rounded-lg font-semibold transition ${
                  key === 'pro'
                    ? 'bg-purple-500 text-white hover:bg-purple-600'
                    : key === 'enterprise'
                    ? 'bg-gray-800 text-white hover:bg-gray-900'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {key === 'free' ? 'Stay Free' : `Upgrade to ${tier.name}`}
              </button>
            </div>
          ))}
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <div className="flex items-start space-x-3">
            <Brain className="h-6 w-6 text-blue-500 mt-1 flex-shrink-0" />
            <div>
              <h4 className="font-semibold text-blue-900 mb-1">Enterprise-Grade AI Features</h4>
              <p className="text-blue-700 text-sm">
                Pro and Enterprise plans include advanced AI matching, recommendations, and analytics 
                designed specifically for industrial companies and large-scale operations.
              </p>
            </div>
          </div>
        </div>

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <div className="flex items-start space-x-3">
            <Crown className="h-6 w-6 text-yellow-600 mt-1 flex-shrink-0" />
            <div>
              <h4 className="font-semibold text-yellow-900 mb-1">Enterprise Plan Benefits</h4>
              <p className="text-yellow-700 text-sm">
                Perfect for large companies with multiple locations, custom integrations, 
                dedicated account management, and 24/7 support.
              </p>
            </div>
          </div>
        </div>

        <div className="flex justify-center space-x-4">
          <button
            onClick={onClose}
            className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition"
          >
            Maybe Later
          </button>
          <button
            onClick={() => {
              // Handle Pro upgrade
              alert('Pro upgrade - This would integrate with your payment processor');
            }}
            className="px-6 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition"
          >
            Upgrade to Pro
          </button>
        </div>
      </div>
    </div>
  );
} 