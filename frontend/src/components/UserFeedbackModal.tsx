import React, { useState } from 'react';
import { Star, StarOff } from 'lucide-react';

interface UserFeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
  matchId: string;
  onSubmit: (feedback: FeedbackData) => void;
}

interface FeedbackData {
  matchId: string;
  rating: number;
  feedback: string;
  categories: string[];
  timestamp: string;
}

const feedbackCategories = [
  'Match was relevant',
  'Match was not relevant',
  'Company location too far',
  'Material type mismatch',
  'Quantity mismatch',
  'Price expectations',
  'Trust concerns',
  'Sustainability impact unclear',
  'Timing issues',
  'Other'
];

export function UserFeedbackModal({ isOpen, onClose, matchId, onSubmit }: UserFeedbackModalProps) {
  const [rating, setRating] = useState(0);
  const [feedback, setFeedback] = useState('');
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleCategoryToggle = (category: string) => {
    setSelectedCategories(prev => 
      prev.includes(category) 
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  const handleSubmit = async () => {
    if (rating === 0) {
      alert('Please provide a rating');
      return;
    }

    setIsSubmitting(true);
    
    const feedbackData: FeedbackData = {
      matchId,
      rating,
      feedback,
      categories: selectedCategories,
      timestamp: new Date().toISOString()
    };

    try {
      await onSubmit(feedbackData);
      // Reset form
      setRating(0);
      setFeedback('');
      setSelectedCategories([]);
      onClose();
    } catch (error) {
      console.error('Error submitting feedback:', error);
      alert('Failed to submit feedback. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (rating > 0 || feedback || selectedCategories.length > 0) {
      if (confirm('You have unsaved feedback. Are you sure you want to close?')) {
        onClose();
      }
    } else {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-gray-900">Rate This Match</h2>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600 text-2xl"
          >
            ×
          </button>
        </div>

        {/* Star Rating */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            How relevant was this match?
          </label>
          <div className="flex space-x-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => setRating(star)}
                className="focus:outline-none"
              >
                {star <= rating ? (
                  <Star className="w-8 h-8 text-yellow-400 fill-current" />
                ) : (
                  <StarOff className="w-8 h-8 text-gray-300" />
                )}
              </button>
            ))}
          </div>
          <div className="text-sm text-gray-500 mt-1">
            {rating === 1 && 'Very Poor'}
            {rating === 2 && 'Poor'}
            {rating === 3 && 'Fair'}
            {rating === 4 && 'Good'}
            {rating === 5 && 'Excellent'}
          </div>
        </div>

        {/* Feedback Categories */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            What aspects apply to this match? (Select all that apply)
          </label>
          <div className="grid grid-cols-1 gap-2">
            {feedbackCategories.map((category) => (
              <label key={category} className="flex items-center">
                <input
                  type="checkbox"
                  checked={selectedCategories.includes(category)}
                  onChange={() => handleCategoryToggle(category)}
                  className="rounded border-gray-300 text-emerald-600 focus:ring-emerald-500"
                />
                <span className="ml-2 text-sm text-gray-700">{category}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Text Feedback */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Additional comments (optional)
          </label>
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            placeholder="Tell us more about why this match was or wasn't helpful..."
          />
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3">
          <button
            onClick={handleClose}
            disabled={isSubmitting}
            className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isSubmitting || rating === 0}
            className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
          </button>
        </div>

        {/* Help Text */}
        <div className="mt-4 p-3 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-700">
            <strong>Why we collect feedback:</strong> Your feedback helps our AI learn and provide better matches in the future. 
            This improves the experience for everyone in the network.
          </p>
        </div>
      </div>
    </div>
  );
} 