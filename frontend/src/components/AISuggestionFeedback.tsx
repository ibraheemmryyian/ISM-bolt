import React, { useState } from 'react';
import { 
  Check, 
  X, 
  Clock, 
  Zap, 
  ThumbsUp, 
  ThumbsDown, 
  MessageSquare,
  Send,
  Loader2
} from 'lucide-react';
import { FederatedLearningService, FeedbackData } from '../lib/federatedLearningService';
import { supabase } from '../lib/supabase';

interface AISuggestionFeedbackProps {
  suggestion: any;
  onFeedbackSubmitted: () => void;
  onApproved?: () => void;
  onRejected?: () => void;
}

export const AISuggestionFeedback: React.FC<AISuggestionFeedbackProps> = ({
  suggestion,
  onFeedbackSubmitted,
  onApproved,
  onRejected
}) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showReasonInput, setShowReasonInput] = useState(false);
  const [reason, setReason] = useState('');
  const [feedbackType, setFeedbackType] = useState<string>('');

  const handleFeedback = async (action: string) => {
    try {
      setIsSubmitting(true);
      setFeedbackType(action);

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

      let finalReason = reason;
      
      // If AI confidence is low and user is rejecting, ask for reason
      if (action === 'reject' && suggestion.confidence_score < 70 && !reason.trim()) {
        setShowReasonInput(true);
        setIsSubmitting(false);
        return;
      }

      const feedback: Omit<FeedbackData, 'id' | 'created_at'> = {
        type: 'suggestion',
        action: action as any,
        reason: finalReason || undefined,
        user_id: user.id,
        company_id: company.id,
        item_id: suggestion.id,
        item_data: suggestion,
        confidence_score: suggestion.confidence_score
      };

      const success = await FederatedLearningService.submitFeedback(feedback);
      
      if (success) {
        onFeedbackSubmitted();
        
        // Trigger specific callbacks
        if (action === 'approve' || action === 'implemented') {
          onApproved?.();
        } else if (action === 'reject' || action === 'not_relevant') {
          onRejected?.();
        }
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
    } finally {
      setIsSubmitting(false);
      setShowReasonInput(false);
      setReason('');
    }
  };

  const getActionButton = (action: string, icon: React.ReactNode, label: string, color: string) => (
    <button
      onClick={() => handleFeedback(action)}
      disabled={isSubmitting}
      className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${color} ${
        isSubmitting && feedbackType === action ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'
      }`}
    >
      {isSubmitting && feedbackType === action ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        icon
      )}
      <span>{label}</span>
    </button>
  );

  return (
    <div className="space-y-3">
      {/* Feedback Buttons */}
      <div className="flex flex-wrap gap-2">
        {getActionButton(
          'approve',
          <Check className="h-4 w-4" />,
          'Approve',
          'bg-green-100 text-green-700 hover:bg-green-200 border border-green-300'
        )}
        
        {getActionButton(
          'already_using',
          <Zap className="h-4 w-4" />,
          'Already Using',
          'bg-blue-100 text-blue-700 hover:bg-blue-200 border border-blue-300'
        )}
        
        {getActionButton(
          'possible',
          <ThumbsUp className="h-4 w-4" />,
          'Possible',
          'bg-yellow-100 text-yellow-700 hover:bg-yellow-200 border border-yellow-300'
        )}
        
        {getActionButton(
          'reject',
          <X className="h-4 w-4" />,
          'Reject',
          'bg-red-100 text-red-700 hover:bg-red-200 border border-red-300'
        )}
        
        {getActionButton(
          'not_relevant',
          <ThumbsDown className="h-4 w-4" />,
          'Not Relevant',
          'bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-300'
        )}
      </div>

      {/* Reason Input */}
      {showReasonInput && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <MessageSquare className="h-5 w-5 text-amber-600 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-amber-800 mb-2">
                Help us improve! Why is this suggestion not relevant?
              </p>
              <textarea
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="Tell us why this suggestion doesn't work for your company..."
                className="w-full p-2 border border-amber-300 rounded-md text-sm focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                rows={3}
              />
              <div className="flex space-x-2 mt-2">
                <button
                  onClick={() => handleFeedback('reject')}
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

      {/* AI Confidence Indicator */}
      <div className="flex items-center space-x-2 text-xs text-gray-500">
        <div className="flex items-center space-x-1">
          <div className={`w-2 h-2 rounded-full ${
            suggestion.confidence_score >= 80 ? 'bg-green-500' :
            suggestion.confidence_score >= 60 ? 'bg-yellow-500' : 'bg-red-500'
          }`} />
          <span>AI Confidence: {suggestion.confidence_score}%</span>
        </div>
        {suggestion.confidence_score < 70 && (
          <span className="text-amber-600">• Low confidence - feedback helps improve AI</span>
        )}
      </div>
    </div>
  );
}; 