import React, { useEffect, useState } from 'react';

interface ImpactData {
  total_carbon_reduction: number;
  total_waste_reduction: number;
  total_cost_savings: number;
  avg_carbon_reduction: number;
  avg_waste_reduction: number;
  avg_cost_savings: number;
  matches_counted: number;
}

interface GlobalImpactPanelProps {
  userId?: string;
}

const GlobalImpactPanel: React.FC<GlobalImpactPanelProps> = ({ userId }) => {
  const [impact, setImpact] = useState<ImpactData | null>(null);
  const [loading, setLoading] = useState(true);
  const [showUser, setShowUser] = useState(false);

  useEffect(() => {
    async function fetchImpact() {
      setLoading(true);
      const url = showUser && userId ? `/api/global-impact?userId=${userId}` : '/api/global-impact';
      const res = await fetch(url);
      const data = await res.json();
      setImpact(data.impact);
      setLoading(false);
    }
    fetchImpact();
  }, [showUser, userId]);

  return (
    <div className="bg-white rounded-xl shadow-sm p-6 mb-8">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-900">Global Impact Forecasting</h2>
        {userId && (
          <button
            onClick={() => setShowUser((v) => !v)}
            className="px-3 py-1 rounded bg-emerald-100 text-emerald-800 text-sm font-semibold hover:bg-emerald-200 transition"
          >
            {showUser ? 'Show Global' : 'Show My Impact'}
          </button>
        )}
      </div>
      {loading ? (
        <div>Loading impact data...</div>
      ) : impact ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-emerald-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-emerald-700">{impact.total_carbon_reduction} t</div>
            <div className="text-sm text-gray-600">Total CO₂ Reduction</div>
            <div className="text-xs text-gray-500 mt-1">Avg: {impact.avg_carbon_reduction} t/match</div>
          </div>
          <div className="bg-blue-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-blue-700">{impact.total_waste_reduction} t</div>
            <div className="text-sm text-gray-600">Total Waste Reduction</div>
            <div className="text-xs text-gray-500 mt-1">Avg: {impact.avg_waste_reduction} t/match</div>
          </div>
          <div className="bg-yellow-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-yellow-700">€{impact.total_cost_savings}</div>
            <div className="text-sm text-gray-600">Total Cost Savings</div>
            <div className="text-xs text-gray-500 mt-1">Avg: €{impact.avg_cost_savings}/match</div>
          </div>
        </div>
      ) : (
        <div>No impact data available.</div>
      )}
      <div className="text-xs text-gray-400 mt-4">Matches counted: {impact?.matches_counted || 0}</div>
    </div>
  );
};

export default GlobalImpactPanel; 