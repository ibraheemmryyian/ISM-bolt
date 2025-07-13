import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';

interface Plugin {
  id: string;
  manifest: {
    name: string;
    version: string;
    description: string;
    author: string;
    plugin_type: string;
    permissions: string[];
    tags: string[];
  };
  status: string;
  loaded_at: string;
  last_used: string;
  usage_count: number;
  error_message?: string;
}

interface MarketplacePlugin {
  name: string;
  version: string;
  description: string;
  author: string;
  plugin_type: string;
  downloads: number;
  rating: number;
  tags: string[];
  price?: number;
}

const PluginManager: React.FC = () => {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [marketplacePlugins, setMarketplacePlugins] = useState<MarketplacePlugin[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'installed' | 'marketplace'>('installed');
  const [selectedPlugin, setSelectedPlugin] = useState<Plugin | null>(null);
  const [executionResult, setExecutionResult] = useState<any>(null);
  const [executionMethod, setExecutionMethod] = useState('');
  const [executionArgs, setExecutionArgs] = useState('');
  const [executionKwargs, setExecutionKwargs] = useState('');

  useEffect(() => {
    loadPlugins();
    loadMarketplacePlugins();
  }, []);

  const loadPlugins = async () => {
    try {
      const response = await fetch('/api/plugins');
      const data = await response.json();
      setPlugins(data.plugins || []);
    } catch (error) {
      console.error('Failed to load plugins:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadMarketplacePlugins = async () => {
    try {
      const params = new URLSearchParams();
      if (searchQuery) params.append('query', searchQuery);
      if (selectedType) params.append('type', selectedType);
      if (selectedTags.length > 0) params.append('tags', selectedTags.join(','));

      const response = await fetch(`/api/marketplace/search?${params}`);
      const data = await response.json();
      setMarketplacePlugins(data.plugins || []);
    } catch (error) {
      console.error('Failed to load marketplace plugins:', error);
    }
  };

  const installPlugin = async (pluginName: string, version?: string) => {
    try {
      const response = await fetch(`/api/marketplace/plugins/${pluginName}/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ version })
      });

      const data = await response.json();
      if (data.success) {
        await loadPlugins();
        alert('Plugin installed successfully!');
      } else {
        alert(`Failed to install plugin: ${data.error}`);
      }
    } catch (error) {
      console.error('Failed to install plugin:', error);
      alert('Failed to install plugin');
    }
  };

  const unloadPlugin = async (pluginId: string) => {
    if (!confirm('Are you sure you want to unload this plugin?')) return;

    try {
      const response = await fetch(`/api/plugins/${pluginId}`, {
        method: 'DELETE'
      });

      const data = await response.json();
      if (data.success) {
        await loadPlugins();
        alert('Plugin unloaded successfully!');
      } else {
        alert(`Failed to unload plugin: ${data.error}`);
      }
    } catch (error) {
      console.error('Failed to unload plugin:', error);
      alert('Failed to unload plugin');
    }
  };

  const executePlugin = async () => {
    if (!selectedPlugin || !executionMethod) return;

    try {
      const args = executionArgs ? JSON.parse(executionArgs) : [];
      const kwargs = executionKwargs ? JSON.parse(executionKwargs) : {};

      const response = await fetch(`/api/plugins/${selectedPlugin.id}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method: executionMethod, args, kwargs })
      });

      const data = await response.json();
      if (data.success) {
        setExecutionResult(data.result);
      } else {
        alert(`Execution failed: ${data.error}`);
      }
    } catch (error) {
      console.error('Failed to execute plugin:', error);
      alert('Failed to execute plugin');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-600';
      case 'inactive': return 'text-gray-600';
      case 'error': return 'text-red-600';
      case 'loading': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'finance': return 'bg-blue-100 text-blue-800';
      case 'carbon': return 'bg-green-100 text-green-800';
      case 'logistics': return 'bg-purple-100 text-purple-800';
      case 'analytics': return 'bg-orange-100 text-orange-800';
      case 'integration': return 'bg-indigo-100 text-indigo-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Plugin Ecosystem</h1>
        <p className="text-gray-600">
          Manage plugins, discover new ones in the marketplace, and extend platform functionality
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('installed')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'installed'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Installed Plugins ({plugins.length})
          </button>
          <button
            onClick={() => setActiveTab('marketplace')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'marketplace'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Marketplace
          </button>
        </nav>
      </div>

      {activeTab === 'installed' && (
        <div className="space-y-6">
          {/* Installed Plugins */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {plugins.map((plugin) => (
              <div key={plugin.id} className="bg-white rounded-lg shadow-md p-6 border">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{plugin.manifest.name}</h3>
                    <p className="text-sm text-gray-500">v{plugin.manifest.version}</p>
                  </div>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(plugin.status)}`}>
                    {plugin.status}
                  </span>
                </div>

                <p className="text-gray-600 text-sm mb-4">{plugin.manifest.description}</p>

                <div className="flex flex-wrap gap-2 mb-4">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTypeColor(plugin.manifest.plugin_type)}`}>
                    {plugin.manifest.plugin_type}
                  </span>
                  {plugin.manifest.tags?.map((tag) => (
                    <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs">
                      {tag}
                    </span>
                  ))}
                </div>

                <div className="text-xs text-gray-500 mb-4">
                  <p>Author: {plugin.manifest.author}</p>
                  <p>Loaded: {new Date(plugin.loaded_at).toLocaleDateString()}</p>
                  <p>Usage: {plugin.usage_count} times</p>
                </div>

                {plugin.error_message && (
                  <div className="bg-red-50 border border-red-200 rounded-md p-3 mb-4">
                    <p className="text-red-800 text-sm">{plugin.error_message}</p>
                  </div>
                )}

                <div className="flex space-x-2">
                  <button
                    onClick={() => setSelectedPlugin(plugin)}
                    className="flex-1 bg-indigo-600 text-white px-3 py-2 rounded-md text-sm font-medium hover:bg-indigo-700"
                  >
                    Execute
                  </button>
                  <button
                    onClick={() => unloadPlugin(plugin.id)}
                    className="flex-1 bg-red-600 text-white px-3 py-2 rounded-md text-sm font-medium hover:bg-red-700"
                  >
                    Unload
                  </button>
                </div>
              </div>
            ))}
          </div>

          {plugins.length === 0 && !loading && (
            <div className="text-center py-12">
              <div className="text-gray-400 text-6xl mb-4">🔌</div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No plugins installed</h3>
              <p className="text-gray-600 mb-4">Install plugins from the marketplace to extend functionality</p>
              <button
                onClick={() => setActiveTab('marketplace')}
                className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700"
              >
                Browse Marketplace
              </button>
            </div>
          )}
        </div>
      )}

      {activeTab === 'marketplace' && (
        <div className="space-y-6">
          {/* Search and Filters */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Search</label>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search plugins..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Type</label>
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="">All Types</option>
                  <option value="finance">Finance</option>
                  <option value="carbon">Carbon</option>
                  <option value="logistics">Logistics</option>
                  <option value="analytics">Analytics</option>
                  <option value="integration">Integration</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Tags</label>
                <input
                  type="text"
                  value={selectedTags.join(', ')}
                  onChange={(e) => setSelectedTags(e.target.value.split(',').map(t => t.trim()).filter(t => t))}
                  placeholder="Enter tags separated by commas"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>
            <div className="mt-4">
              <button
                onClick={loadMarketplacePlugins}
                className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700"
              >
                Search
              </button>
            </div>
          </div>

          {/* Marketplace Plugins */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {marketplacePlugins.map((plugin) => (
              <div key={plugin.name} className="bg-white rounded-lg shadow-md p-6 border">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{plugin.name}</h3>
                    <p className="text-sm text-gray-500">v{plugin.version}</p>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center text-yellow-400">
                      {'★'.repeat(Math.floor(plugin.rating))}
                      <span className="text-gray-600 text-sm ml-1">({plugin.rating})</span>
                    </div>
                    <p className="text-xs text-gray-500">{plugin.downloads} downloads</p>
                  </div>
                </div>

                <p className="text-gray-600 text-sm mb-4">{plugin.description}</p>

                <div className="flex flex-wrap gap-2 mb-4">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTypeColor(plugin.plugin_type)}`}>
                    {plugin.plugin_type}
                  </span>
                  {plugin.tags?.map((tag) => (
                    <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-xs">
                      {tag}
                    </span>
                  ))}
                </div>

                <div className="text-xs text-gray-500 mb-4">
                  <p>Author: {plugin.author}</p>
                  {plugin.price && <p>Price: ${plugin.price}</p>}
                </div>

                <button
                  onClick={() => installPlugin(plugin.name)}
                  className="w-full bg-green-600 text-white px-3 py-2 rounded-md text-sm font-medium hover:bg-green-700"
                >
                  Install
                </button>
              </div>
            ))}
          </div>

          {marketplacePlugins.length === 0 && !loading && (
            <div className="text-center py-12">
              <div className="text-gray-400 text-6xl mb-4">🏪</div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No plugins found</h3>
              <p className="text-gray-600">Try adjusting your search criteria</p>
            </div>
          )}
        </div>
      )}

      {/* Plugin Execution Modal */}
      {selectedPlugin && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Execute Plugin: {selectedPlugin.manifest.name}</h2>
              <button
                onClick={() => setSelectedPlugin(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Method Name</label>
                <input
                  type="text"
                  value={executionMethod}
                  onChange={(e) => setExecutionMethod(e.target.value)}
                  placeholder="e.g., calculate_roi, analyze_data"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Arguments (JSON array)</label>
                <textarea
                  value={executionArgs}
                  onChange={(e) => setExecutionArgs(e.target.value)}
                  placeholder='["arg1", "arg2"]'
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Keyword Arguments (JSON object)</label>
                <textarea
                  value={executionKwargs}
                  onChange={(e) => setExecutionKwargs(e.target.value)}
                  placeholder='{"key1": "value1", "key2": "value2"}'
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={executePlugin}
                  className="flex-1 bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700"
                >
                  Execute
                </button>
                <button
                  onClick={() => setSelectedPlugin(null)}
                  className="flex-1 bg-gray-300 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-400"
                >
                  Cancel
                </button>
              </div>

              {executionResult && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Execution Result:</h3>
                  <pre className="bg-gray-100 p-4 rounded-md text-sm overflow-x-auto">
                    {JSON.stringify(executionResult, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PluginManager; 