import React from 'react';
import { FaCheckCircle, FaExclamationTriangle } from 'react-icons/fa';

interface MatchExplanation {
  semantic_reason: string;
  trust_reason: string;
  sustainability_reason: string;
  forecast_reason: string;
  overall_reason: string;
  confidence_level: string;
}

interface AIExplanationModalProps {
  isOpen: boolean;
  onClose: () => void;
  explanation: MatchExplanation;
  scores: {
    semantic_score: number;
    trust_score: number;
    sustainability_score: number;
    forecast_score: number;
    external_score: number;
    revolutionary_score: number;
  };
  compliance?: { compliant: boolean; issues: string[]; suggestions: string[] };
}

export function AIExplanationModal({ isOpen, onClose, explanation, scores, compliance }: AIExplanationModalProps) {
  if (!isOpen) return null;

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    if (score >= 0.4) return 'text-orange-600';
    return 'text-red-600';
  };

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'Very High': return 'text-green-600 bg-green-100';
      case 'High': return 'text-blue-600 bg-blue-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'Low': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-900">AI Match Explanation</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl"
          >
            ×
          </button>
        </div>

        {/* Overall Score */}
        <div className="mb-6 p-4 bg-gradient-to-r from-emerald-50 to-blue-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Overall Match Score</h3>
              <p className="text-sm text-gray-600 mt-1">{explanation.overall_reason}</p>
            </div>
            <div className="text-right">
              <div className={`text-3xl font-bold ${getScoreColor(scores.revolutionary_score)}`}>
                {(scores.revolutionary_score * 100).toFixed(0)}%
              </div>
              <div className={`text-sm px-2 py-1 rounded-full ${getConfidenceColor(explanation.confidence_level)}`}>
                {explanation.confidence_level} Confidence
              </div>
            </div>
          </div>
        </div>

        {/* Individual Factors */}
        <div className="space-y-4">
          {/* Semantic Similarity */}
          <div className="border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-gray-900">Semantic Similarity</h4>
              <span className={`font-bold ${getScoreColor(scores.semantic_score)}`}>
                {(scores.semantic_score * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-sm text-gray-600">{explanation.semantic_reason}</p>
          </div>

          {/* Trust Score */}
          <div className="border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-gray-900">Trust & Reliability</h4>
              <span className={`font-bold ${getScoreColor(scores.trust_score)}`}>
                {(scores.trust_score * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-sm text-gray-600">{explanation.trust_reason}</p>
          </div>

          {/* Sustainability Impact */}
          <div className="border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-gray-900">Sustainability Impact</h4>
              <span className={`font-bold ${getScoreColor(scores.sustainability_score)}`}>
                {(scores.sustainability_score * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-sm text-gray-600">{explanation.sustainability_reason}</p>
          </div>

          {/* Future Compatibility */}
          <div className="border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-gray-900">Future Compatibility</h4>
              <span className={`font-bold ${getScoreColor(scores.forecast_score)}`}>
                {(scores.forecast_score * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-sm text-gray-600">{explanation.forecast_reason}</p>
          </div>

          {/* External Data */}
          <div className="border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-gray-900">Market Conditions</h4>
              <span className={`font-bold ${getScoreColor(scores.external_score)}`}>
                {(scores.external_score * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-sm text-gray-600">Based on current market prices, regulations, and logistics availability</p>
          </div>
        </div>

        {/* Compliance Section */}
        {compliance && (
          <div className="compliance-section" style={{ transition: 'all 0.3s', background: '#f8f9fa', borderRadius: 8, padding: 16, marginTop: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              Regulatory Compliance
              {compliance.compliant ? (
                <FaCheckCircle color="green" title="Compliant" />
              ) : (
                <FaExclamationTriangle color="orange" title="Non-Compliant" />
              )}
            </h3>
            <div>Status: <strong style={{ color: compliance.compliant ? 'green' : 'red' }}>{compliance.compliant ? 'Compliant' : 'Non-Compliant'}</strong></div>
            {compliance.issues.length > 0 && (
              <div style={{ marginTop: 8 }}><strong>Issues:</strong>
                <ul>{compliance.issues.map((issue, i) => <li key={i}>{issue}</li>)}</ul>
              </div>
            )}
            {compliance.suggestions.length > 0 && (
              <div style={{ marginTop: 8 }}><strong>Suggestions:</strong>
                <ul>{compliance.suggestions.map((s, i) => <li key={i}>{s}</li>)}</ul>
              </div>
            )}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3 mt-6 pt-4 border-t">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            Close
          </button>
          <button
            onClick={() => {
              // TODO: Implement feedback collection
              console.log('User feedback collected');
            }}
            className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600"
          >
            Provide Feedback
          </button>
        </div>
      </div>
    </div>
  );
} 