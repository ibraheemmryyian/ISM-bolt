import React from 'react';
import { CheckCircle, Circle } from 'lucide-react';

interface Step {
  id: string;
  title: string;
  description?: string;
  status: 'completed' | 'current' | 'upcoming';
}

interface ProgressIndicatorProps {
  steps: Step[];
  currentStep: number;
  onStepClick?: (stepIndex: number) => void;
  className?: string;
  showStepNumbers?: boolean;
  orientation?: 'horizontal' | 'vertical';
}

export function ProgressIndicator({
  steps,
  currentStep,
  onStepClick,
  className = '',
  showStepNumbers = true,
  orientation = 'horizontal'
}: ProgressIndicatorProps) {
  const isHorizontal = orientation === 'horizontal';

  const getStepIcon = (step: Step, index: number) => {
    if (step.status === 'completed') {
      return <CheckCircle className="h-6 w-6 text-green-600" />;
    }
    
    if (step.status === 'current') {
      return (
        <div className="h-6 w-6 rounded-full border-2 border-blue-600 bg-blue-600 flex items-center justify-center">
          <span className="text-white text-xs font-medium">
            {showStepNumbers ? index + 1 : ''}
          </span>
        </div>
      );
    }
    
    return (
      <div className="h-6 w-6 rounded-full border-2 border-gray-300 bg-white flex items-center justify-center">
        <span className="text-gray-500 text-xs font-medium">
          {showStepNumbers ? index + 1 : ''}
        </span>
      </div>
    );
  };

  const getStepClasses = (step: Step) => {
    const baseClasses = 'flex items-center space-x-3';
    
    if (step.status === 'completed') {
      return `${baseClasses} text-green-600`;
    }
    
    if (step.status === 'current') {
      return `${baseClasses} text-blue-600`;
    }
    
    return `${baseClasses} text-gray-500`;
  };

  const getTitleClasses = (step: Step) => {
    const baseClasses = 'font-medium';
    
    if (step.status === 'completed') {
      return `${baseClasses} text-green-800`;
    }
    
    if (step.status === 'current') {
      return `${baseClasses} text-blue-800`;
    }
    
    return `${baseClasses} text-gray-700`;
  };

  return (
    <div className={`${isHorizontal ? 'flex' : 'space-y-4'} ${className}`}>
      {steps.map((step, index) => (
        <div key={step.id} className={`${isHorizontal ? 'flex-1' : ''}`}>
          <div className={`${isHorizontal ? 'flex flex-col items-center' : 'flex items-start'}`}>
            {/* Step Icon and Line */}
            <div className={`${isHorizontal ? 'flex items-center w-full' : 'flex flex-col items-center'}`}>
              {isHorizontal && index > 0 && (
                <div className={`flex-1 h-0.5 ${
                  steps[index - 1].status === 'completed' ? 'bg-green-600' : 'bg-gray-300'
                }`} />
              )}
              
              <button
                onClick={() => onStepClick?.(index)}
                disabled={!onStepClick}
                className={`${getStepClasses(step)} ${
                  onStepClick ? 'cursor-pointer hover:opacity-80' : 'cursor-default'
                }`}
              >
                {getStepIcon(step, index)}
              </button>
              
              {isHorizontal && index < steps.length - 1 && (
                <div className={`flex-1 h-0.5 ${
                  step.status === 'completed' ? 'bg-green-600' : 'bg-gray-300'
                }`} />
              )}
            </div>
            
            {/* Step Content */}
            <div className={`${isHorizontal ? 'mt-3 text-center' : 'ml-3 flex-1'}`}>
              <h3 className={getTitleClasses(step)}>
                {step.title}
              </h3>
              {step.description && (
                <p className="text-sm text-gray-500 mt-1">
                  {step.description}
                </p>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// Specialized progress indicators
export function OnboardingProgress({ currentStep }: { currentStep: number }) {
  const steps = [
    {
      id: 'company-info',
      title: 'Company Information',
      description: 'Basic company details',
      status: 'completed' as const
    },
    {
      id: 'industry',
      title: 'Industry & Processes',
      description: 'Your industry and operations',
      status: 'completed' as const
    },
    {
      id: 'waste-streams',
      title: 'Waste Streams',
      description: 'Materials you want to optimize',
      status: 'current' as const
    },
    {
      id: 'goals',
      title: 'Sustainability Goals',
      description: 'Your environmental objectives',
      status: 'upcoming' as const
    },
    {
      id: 'review',
      title: 'Review & Submit',
      description: 'Final review and submission',
      status: 'upcoming' as const
    }
  ];

  // Update step statuses based on current step
  const updatedSteps = steps.map((step, index) => ({
    ...step,
    status: index < currentStep ? 'completed' : index === currentStep ? 'current' : 'upcoming'
  }));

  return (
    <ProgressIndicator
      steps={updatedSteps}
      currentStep={currentStep}
      orientation="horizontal"
      className="mb-8"
    />
  );
}

export function ApplicationProgress({ status }: { status: 'pending' | 'reviewing' | 'approved' | 'rejected' }) {
  const steps: Step[] = [
    {
      id: 'submitted',
      title: 'Application Submitted',
      description: 'Your application has been received',
      status: 'completed'
    },
    {
      id: 'reviewing',
      title: 'Under Review',
      description: 'We\'re reviewing your information',
      status: 'current'
    },
    {
      id: 'approved',
      title: 'Approved',
      description: 'Welcome to SymbioFlows!',
      status: 'upcoming'
    }
  ];

  // Update step statuses based on application status
  const updatedSteps: Step[] = steps.map(step => {
    if (step.id === 'submitted') return { ...step, status: 'completed' as const };
    if (step.id === 'reviewing') {
      return { 
        ...step, 
        status: (status === 'reviewing' || status === 'approved' || status === 'rejected') ? 'completed' as const : 'current' as const 
      };
    }
    if (step.id === 'approved') {
      return { 
        ...step, 
        status: status === 'approved' ? 'completed' as const : 'upcoming' as const 
      };
    }
    return step;
  });

  return (
    <ProgressIndicator
      steps={updatedSteps}
      currentStep={status === 'pending' ? 1 : status === 'reviewing' ? 2 : 3}
      orientation="horizontal"
      className="mb-6"
    />
  );
} 