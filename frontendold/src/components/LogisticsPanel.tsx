import React, { useState, useEffect } from 'react';
import { 
  Truck, 
  Train, 
  Ship, 
  Plane, 
  MapPin, 
  Calculator, 
  Leaf, 
  Clock, 
  Euro,
  TrendingUp,
  Loader2,
  AlertCircle,
  CheckCircle,
  Info
} from 'lucide-react';

interface Location {
  name: string;
  latitude: number;
  longitude: number;
  country: string;
  city: string;
  postal_code?: string;
}

interface RouteSegment {
  mode: string;
  origin: string;
  destination: string;
  distance_km: number;
  duration_hours: number;
  cost_eur: number;
  carbon_kg: number;
}

interface Route {
  route_id: string;
  summary: {
    total_distance_km: number;
    total_duration_hours: number;
    total_cost_eur: number;
    total_carbon_kg: number;
    reliability_score: number;
  };
  segments: RouteSegment[];
  carbon_impact: {
    total_carbon_kg: number;
    carbon_tax_eur: number;
    carbon_intensity_kg_per_ton_km: number;
    carbon_savings_vs_truck: number;
    carbon_equivalent: {
      trees_needed: number;
      car_km_equivalent: number;
      flight_km_equivalent: number;
    };
  };
  recommendations: string[];
}

interface LogisticsPanelProps {
  userId?: string;
}

export function LogisticsPanel({ userId }: LogisticsPanelProps) {
  const [origin, setOrigin] = useState<Location>({
    name: '',
    latitude: 0,
    longitude: 0,
    country: '',
    city: ''
  });
  
  const [destination, setDestination] = useState<Location>({
    name: '',
    latitude: 0,
    longitude: 0,
    country: '',
    city: ''
  });
  
  const [cargoWeight, setCargoWeight] = useState<number>(1);
  const [cargoValue, setCargoValue] = useState<number>(1000);
  const [urgency, setUrgency] = useState<string>('normal');
  const [routes, setRoutes] = useState<Route[]>([]);
  const [selectedRoute, setSelectedRoute] = useState<Route | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [optimizationCriteria, setOptimizationCriteria] = useState<string>('cost');
  const [showSimulation, setShowSimulation] = useState(false);
  const [simulationResults, setSimulationResults] = useState<any[]>([]);

  // Predefined locations for quick selection
  const predefinedLocations = [
    { name: 'Rotterdam Port', latitude: 51.9225, longitude: 4.4792, country: 'Netherlands', city: 'Rotterdam' },
    { name: 'Hamburg Port', latitude: 53.5511, longitude: 9.9937, country: 'Germany', city: 'Hamburg' },
    { name: 'Antwerp Port', latitude: 51.2194, longitude: 4.4025, country: 'Belgium', city: 'Antwerp' },
    { name: 'Barcelona Port', latitude: 41.3851, longitude: 2.1734, country: 'Spain', city: 'Barcelona' },
    { name: 'Milan Industrial Zone', latitude: 45.4642, longitude: 9.1900, country: 'Italy', city: 'Milan' },
    { name: 'Amsterdam', latitude: 52.3676, longitude: 4.9041, country: 'Netherlands', city: 'Amsterdam' },
    { name: 'Paris', latitude: 48.8566, longitude: 2.3522, country: 'France', city: 'Paris' },
    { name: 'Berlin', latitude: 52.5200, longitude: 13.4050, country: 'Germany', city: 'Berlin' }
  ];

  const calculateRoutes = async () => {
    if (!origin.name || !destination.name || !cargoWeight || !cargoValue) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/logistics/route-planning', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          origin,
          destination,
          cargo_weight: cargoWeight,
          cargo_value: cargoValue,
          urgency
        })
      });

      if (!response.ok) {
        throw new Error('Failed to calculate routes');
      }

      const data = await response.json();
      setRoutes(data.routes);
      setSelectedRoute(data.routes[0] || null);
    } catch (err: any) {
      setError(err.message || 'Failed to calculate routes');
    } finally {
      setLoading(false);
    }
  };

  const runSimulation = async () => {
    if (!origin.name || !destination.name || !cargoWeight || !cargoValue) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/logistics/simulation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          origin,
          destination,
          cargo_weight: cargoWeight,
          cargo_value: cargoValue,
          scenarios: ['normal', 'urgent', 'express', 'eco_friendly']
        })
      });

      if (!response.ok) {
        throw new Error('Failed to run simulation');
      }

      const data = await response.json();
      setSimulationResults(data.simulation_results);
      setShowSimulation(true);
    } catch (err: any) {
      setError(err.message || 'Failed to run simulation');
    } finally {
      setLoading(false);
    }
  };

  const optimizeRoute = async () => {
    if (routes.length === 0) {
      setError('No routes to optimize');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/logistics/optimize-route', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          routes,
          optimization_criteria: optimizationCriteria,
          cargo_weight: cargoWeight,
          cargo_value: cargoValue
        })
      });

      if (!response.ok) {
        throw new Error('Failed to optimize route');
      }

      const data = await response.json();
      setSelectedRoute(data.optimal_route);
    } catch (err: any) {
      setError(err.message || 'Failed to optimize route');
    } finally {
      setLoading(false);
    }
  };

  const getTransportIcon = (mode: string) => {
    switch (mode) {
      case 'truck': return <Truck className="h-4 w-4" />;
      case 'train': return <Train className="h-4 w-4" />;
      case 'ship': return <Ship className="h-4 w-4" />;
      case 'air': return <Plane className="h-4 w-4" />;
      default: return <Truck className="h-4 w-4" />;
    }
  };

  const formatDuration = (hours: number) => {
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;
    if (days > 0) {
      return `${days}d ${remainingHours}h`;
    }
    return `${remainingHours}h`;
  };

  const formatCost = (cost: number) => {
    return new Intl.NumberFormat('de-DE', {
      style: 'currency',
      currency: 'EUR'
    }).format(cost);
  };

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Truck className="h-6 w-6 text-blue-500" />
          <h2 className="text-xl font-bold text-gray-900">Logistics & Cost Engine</h2>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={runSimulation}
            disabled={loading}
            className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition disabled:opacity-50"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Run Simulation'}
          </button>
        </div>
      </div>

      {/* Input Form */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Origin */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Origin</h3>
          <div className="space-y-3">
            <input
              type="text"
              placeholder="Location name"
              value={origin.name}
              onChange={(e) => setOrigin({ ...origin, name: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="grid grid-cols-2 gap-2">
              <input
                type="number"
                placeholder="Latitude"
                value={origin.latitude || ''}
                onChange={(e) => setOrigin({ ...origin, latitude: parseFloat(e.target.value) || 0 })}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <input
                type="number"
                placeholder="Longitude"
                value={origin.longitude || ''}
                onChange={(e) => setOrigin({ ...origin, longitude: parseFloat(e.target.value) || 0 })}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <input
              type="text"
              placeholder="Country"
              value={origin.country}
              onChange={(e) => setOrigin({ ...origin, country: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="text"
              placeholder="City"
              value={origin.city}
              onChange={(e) => setOrigin({ ...origin, city: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Destination */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Destination</h3>
          <div className="space-y-3">
            <input
              type="text"
              placeholder="Location name"
              value={destination.name}
              onChange={(e) => setDestination({ ...destination, name: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="grid grid-cols-2 gap-2">
              <input
                type="number"
                placeholder="Latitude"
                value={destination.latitude || ''}
                onChange={(e) => setDestination({ ...destination, latitude: parseFloat(e.target.value) || 0 })}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <input
                type="number"
                placeholder="Longitude"
                value={destination.longitude || ''}
                onChange={(e) => setDestination({ ...destination, longitude: parseFloat(e.target.value) || 0 })}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <input
              type="text"
              placeholder="Country"
              value={destination.country}
              onChange={(e) => setDestination({ ...destination, country: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="text"
              placeholder="City"
              value={destination.city}
              onChange={(e) => setDestination({ ...destination, city: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {/* Cargo Details */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Cargo Weight (tons)</label>
          <input
            type="number"
            value={cargoWeight}
            onChange={(e) => setCargoWeight(parseFloat(e.target.value) || 0)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Cargo Value (EUR)</label>
          <input
            type="number"
            value={cargoValue}
            onChange={(e) => setCargoValue(parseFloat(e.target.value) || 0)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Urgency</label>
          <select
            value={urgency}
            onChange={(e) => setUrgency(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="normal">Normal</option>
            <option value="urgent">Urgent</option>
            <option value="express">Express</option>
          </select>
        </div>
      </div>

      {/* Quick Location Selection */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Quick Location Selection</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {predefinedLocations.map((location, index) => (
            <button
              key={index}
              onClick={() => {
                if (index % 2 === 0) {
                  setOrigin(location);
                } else {
                  setDestination(location);
                }
              }}
              className="px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition"
            >
              {location.name}
            </button>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-4 mb-6">
        <button
          onClick={calculateRoutes}
          disabled={loading}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition disabled:opacity-50 flex items-center space-x-2"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Calculator className="h-4 w-4" />}
          <span>Calculate Routes</span>
        </button>
        
        <select
          value={optimizationCriteria}
          onChange={(e) => setOptimizationCriteria(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="cost">Optimize by Cost</option>
          <option value="carbon">Optimize by Carbon</option>
          <option value="time">Optimize by Time</option>
          <option value="reliability">Optimize by Reliability</option>
          <option value="balanced">Balanced Optimization</option>
        </select>
        
        <button
          onClick={optimizeRoute}
          disabled={routes.length === 0 || loading}
          className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition disabled:opacity-50"
        >
          Optimize
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* Routes Display */}
      {routes.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Available Routes</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {routes.map((route, index) => (
              <div
                key={route.route_id}
                className={`p-4 border rounded-lg cursor-pointer transition ${
                  selectedRoute?.route_id === route.route_id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedRoute(route)}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">Route {index + 1}</span>
                  <div className="flex items-center space-x-1">
                    {route.segments.map((segment, segIndex) => (
                      <div key={segIndex} className="flex items-center space-x-1">
                        {getTransportIcon(segment.mode)}
                        {segIndex < route.segments.length - 1 && <span>→</span>}
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Distance:</span>
                    <span className="font-medium">{route.summary.total_distance_km.toFixed(0)} km</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Duration:</span>
                    <span className="font-medium">{formatDuration(route.summary.total_duration_hours)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Cost:</span>
                    <span className="font-medium text-green-600">{formatCost(route.summary.total_cost_eur)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Carbon:</span>
                    <span className="font-medium text-orange-600">{route.summary.total_carbon_kg.toFixed(1)} kg CO2</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Selected Route Details */}
      {selectedRoute && (
        <div className="mt-6 p-6 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Route Details</h3>
          
          {/* Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{selectedRoute.summary.total_distance_km.toFixed(0)}</div>
              <div className="text-sm text-gray-600">km</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{formatCost(selectedRoute.summary.total_cost_eur)}</div>
              <div className="text-sm text-gray-600">Total Cost</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{selectedRoute.summary.total_carbon_kg.toFixed(1)}</div>
              <div className="text-sm text-gray-600">kg CO2</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{(selectedRoute.summary.reliability_score * 100).toFixed(0)}%</div>
              <div className="text-sm text-gray-600">Reliability</div>
            </div>
          </div>

          {/* Segments */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-gray-900 mb-3">Route Segments</h4>
            <div className="space-y-3">
              {selectedRoute.segments.map((segment, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-white rounded-lg">
                  <div className="flex items-center space-x-3">
                    {getTransportIcon(segment.mode)}
                    <div>
                      <div className="font-medium">{segment.origin} → {segment.destination}</div>
                      <div className="text-sm text-gray-600">{segment.distance_km.toFixed(0)} km</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-medium">{formatCost(segment.cost_eur)}</div>
                    <div className="text-sm text-gray-600">{formatDuration(segment.duration_hours)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Carbon Impact */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-gray-900 mb-3">Carbon Impact</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-3 bg-white rounded-lg">
                <div className="text-lg font-bold text-green-600">
                  {selectedRoute.carbon_impact.carbon_savings_vs_truck > 0 ? '+' : ''}
                  {selectedRoute.carbon_impact.carbon_savings_vs_truck.toFixed(1)} kg CO2
                </div>
                <div className="text-sm text-gray-600">vs Truck-only</div>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="text-lg font-bold text-blue-600">
                  {selectedRoute.carbon_impact.carbon_equivalent.trees_needed.toFixed(0)}
                </div>
                <div className="text-sm text-gray-600">Trees needed</div>
              </div>
              <div className="p-3 bg-white rounded-lg">
                <div className="text-lg font-bold text-orange-600">
                  {formatCost(selectedRoute.carbon_impact.carbon_tax_eur)}
                </div>
                <div className="text-sm text-gray-600">Carbon Tax</div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          {selectedRoute.recommendations.length > 0 && (
            <div>
              <h4 className="text-md font-semibold text-gray-900 mb-3">Recommendations</h4>
              <div className="space-y-2">
                {selectedRoute.recommendations.map((rec, index) => (
                  <div key={index} className="flex items-start space-x-2 p-3 bg-blue-50 rounded-lg">
                    <Info className="h-4 w-4 text-blue-500 mt-0.5" />
                    <span className="text-sm text-blue-700">{rec}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Simulation Results */}
      {showSimulation && simulationResults.length > 0 && (
        <div className="mt-6 p-6 bg-purple-50 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Simulation Results</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {simulationResults.map((result, index) => (
              <div key={index} className="p-4 bg-white rounded-lg">
                <div className="text-lg font-bold text-purple-600 capitalize mb-2">{result.scenario}</div>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span>Cost:</span>
                    <span className="font-medium">{formatCost(result.route.summary.total_cost_eur)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Carbon:</span>
                    <span className="font-medium">{result.route.summary.total_carbon_kg.toFixed(1)} kg</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Time:</span>
                    <span className="font-medium">{formatDuration(result.route.summary.total_duration_hours)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 