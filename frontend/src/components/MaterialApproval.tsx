import React, { useState } from 'react';
import { 
  Check, 
  X, 
  MessageSquare,
  Send,
  Loader2,
  Globe,
  TrendingUp
} from 'lucide-react';
import { FederatedLearningService } from '../lib/federatedLearningService';
import { supabase } from '../lib/supabase';

interface MaterialApprovalProps {
  material: any;
  onApproved: () => void;
  onRejected: () => void;
  onFeedbackSubmitted: () => void;
}

export const MaterialApproval: React.FC<MaterialApprovalProps> = ({
  material,
  onApproved,
  onRejected,
  onFeedbackSubmitted
}) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showReasonInput, setShowReasonInput] = useState(false);
  const [reason, setReason] = useState('');
  const [action, setAction] = useState<string>('');

  const handleApproval = async (approved: boolean) => {
    try {
      setIsSubmitting(true);
      setAction(approved ? 'approve' : 'reject');

      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        console.error('User not authenticated');
        return;
      }

      const { data: company } = await supabase
        .from('companies')
        .select('id')
        .eq('user_id', user.id)
        .single();

      if (!company) {
        console.error('Company not found');
        return;
      }

      // If rejecting and no reason provided, ask for one
      if (!approved && !reason.trim()) {
        setShowReasonInput(true);
        setIsSubmitting(false);
        return;
      }

      // Submit federated learning feedback
      const feedback = {
        type: 'material' as const,
        action: approved ? 'approve' as const : 'reject' as const,
        reason: reason || undefined,
        user_id: user.id,
        company_id: company.id,
        item_id: material.id,
        item_data: material,
        confidence_score: material.ai_confidence || 75
      };

      await FederatedLearningService.submitFeedback(feedback);

      // If approved, post to marketplace
      if (approved) {
        await postToMarketplace(material, company.id);
        onApproved();
      } else {
        onRejected();
      }

      onFeedbackSubmitted();
    } catch (error) {
      console.error('Error handling material approval:', error);
    } finally {
      setIsSubmitting(false);
      setShowReasonInput(false);
      setReason('');
    }
  };

  const postToMarketplace = async (material: any, companyId: string) => {
    try {
      // Update material status to approved and post to marketplace
      const { error: updateError } = await supabase
        .from('materials')
        .update({
          status: 'approved',
          marketplace_posted: true,
          posted_at: new Date().toISOString()
        })
        .eq('id', material.id);

      if (updateError) {
        console.error('Error updating material status:', updateError);
        return;
      }

      // Trigger AI matching for this material
      await triggerAIMatching(material, companyId);

    } catch (error) {
      console.error('Error posting to marketplace:', error);
    }
  };

  const triggerAIMatching = async (material: any, companyId: string) => {
    try {
      const response = await fetch('/api/ai-matching/trigger', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          material_id: material.id,
          company_id: companyId,
          material_data: material
        })
      });

      if (!response.ok) {
        console.error('Error triggering AI matching:', response.statusText);
      }
    } catch (error) {
      console.error('Error triggering AI matching:', error);
    }
  };

  return (
    <div className="space-y-4">
      {/* Approval Buttons */}
      <div className="flex space-x-3">
        <button
          onClick={() => handleApproval(true)}
          disabled={isSubmitting}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
            isSubmitting && action === 'approve'
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-green-100 text-green-700 hover:bg-green-200 border border-green-300 hover:scale-105'
          }`}
        >
          {isSubmitting && action === 'approve' ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Check className="h-4 w-4" />
          )}
          <span>Approve & Post to Marketplace</span>
        </button>

        <button
          onClick={() => handleApproval(false)}
          disabled={isSubmitting}
          className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
            isSubmitting && action === 'reject'
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-red-100 text-red-700 hover:bg-red-200 border border-red-300 hover:scale-105'
          }`}
        >
          {isSubmitting && action === 'reject' ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <X className="h-4 w-4" />
          )}
          <span>Reject</span>
        </button>
      </div>

      {/* Reason Input for Rejection */}
      {showReasonInput && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <MessageSquare className="h-5 w-5 text-amber-600 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-amber-800 mb-2">
                Help us improve! Why is this material listing not suitable?
              </p>
              <textarea
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="Tell us why this material listing doesn't work for your company..."
                className="w-full p-2 border border-amber-300 rounded-md text-sm focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                rows={3}
              />
              <div className="flex space-x-2 mt-2">
                <button
                  onClick={() => handleApproval(false)}
                  disabled={isSubmitting}
                  className="flex items-center space-x-1 px-3 py-1 bg-amber-600 text-white rounded-md text-sm hover:bg-amber-700 disabled:opacity-50"
                >
                  <Send className="h-3 w-3" />
                  <span>Submit Feedback</span>
                </button>
                <button
                  onClick={() => {
                    setShowReasonInput(false);
                    setReason('');
                    setIsSubmitting(false);
                  }}
                  className="px-3 py-1 bg-gray-200 text-gray-700 rounded-md text-sm hover:bg-gray-300"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Approval Benefits */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <Globe className="h-5 w-5 text-blue-600 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-blue-800 mb-1">
              Benefits of Approval:
            </h4>
            <ul className="text-xs text-blue-700 space-y-1">
              <li>• Material becomes visible to potential partners</li>
              <li>• AI matching starts immediately</li>
              <li>• Receive real-time matching suggestions</li>
              <li>• Contribute to federated learning for better AI</li>
            </ul>
          </div>
        </div>
      </div>

      {/* AI Confidence Indicator */}
      {material.ai_confidence && (
        <div className="flex items-center space-x-2 text-xs text-gray-500">
          <div className="flex items-center space-x-1">
            <div className={`w-2 h-2 rounded-full ${
              material.ai_confidence >= 80 ? 'bg-green-500' :
              material.ai_confidence >= 60 ? 'bg-yellow-500' : 'bg-red-500'
            }`} />
            <span>AI Confidence: {material.ai_confidence}%</span>
          </div>
          {material.ai_confidence < 70 && (
            <span className="text-amber-600">• Low confidence - feedback helps improve AI</span>
          )}
        </div>
      )}
    </div>
  );
}; 