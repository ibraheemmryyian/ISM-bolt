import React, { useState } from 'react';
import { HelpCircle, Info } from 'lucide-react';

interface TooltipProps {
  content: string;
  children?: React.ReactNode;
  icon?: 'help' | 'info';
  position?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
}

export function Tooltip({ 
  content, 
  children, 
  icon = 'help', 
  position = 'top',
  className = ''
}: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  const getPositionClasses = () => {
    switch (position) {
      case 'top':
        return 'bottom-full left-1/2 transform -translate-x-1/2 mb-2';
      case 'bottom':
        return 'top-full left-1/2 transform -translate-x-1/2 mt-2';
      case 'left':
        return 'right-full top-1/2 transform -translate-y-1/2 mr-2';
      case 'right':
        return 'left-full top-1/2 transform -translate-y-1/2 ml-2';
    }
  };

  const getArrowClasses = () => {
    switch (position) {
      case 'top':
        return 'top-full left-1/2 transform -translate-x-1/2 border-t-gray-800';
      case 'bottom':
        return 'bottom-full left-1/2 transform -translate-x-1/2 border-b-gray-800';
      case 'left':
        return 'left-full top-1/2 transform -translate-y-1/2 border-l-gray-800';
      case 'right':
        return 'right-full top-1/2 transform -translate-y-1/2 border-r-gray-800';
    }
  };

  return (
    <div 
      className={`relative inline-block ${className}`}
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children || (
        <button className="inline-flex items-center justify-center w-5 h-5 text-gray-400 hover:text-gray-600 transition-colors">
          {icon === 'help' ? (
            <HelpCircle className="w-4 h-4" />
          ) : (
            <Info className="w-4 h-4" />
          )}
        </button>
      )}
      
      {isVisible && (
        <div className={`absolute z-50 ${getPositionClasses()}`}>
          <div className="bg-gray-800 text-white text-sm rounded-lg px-3 py-2 max-w-xs shadow-lg">
            {content}
            <div className={`absolute w-0 h-0 border-4 border-transparent ${getArrowClasses()}`}></div>
          </div>
        </div>
      )}
    </div>
  );
}

// Specialized tooltip components
export function HelpTooltip({ content, children, ...props }: Omit<TooltipProps, 'icon'>) {
  return <Tooltip content={content} icon="help" {...props}>{children}</Tooltip>;
}

export function InfoTooltip({ content, children, ...props }: Omit<TooltipProps, 'icon'>) {
  return <Tooltip content={content} icon="info" {...props}>{children}</Tooltip>;
}

// Form field with tooltip
interface FormFieldWithTooltipProps {
  label: string;
  tooltipContent: string;
  children: React.ReactNode;
  required?: boolean;
  className?: string;
}

export function FormFieldWithTooltip({ 
  label, 
  tooltipContent, 
  children, 
  required = false,
  className = ''
}: FormFieldWithTooltipProps) {
  return (
    <div className={`space-y-2 ${className}`}>
      <label className="flex items-center space-x-2 text-sm font-medium text-gray-700">
        <span>{label}</span>
        {required && <span className="text-red-500">*</span>}
        <HelpTooltip content={tooltipContent} />
      </label>
      {children}
    </div>
  );
} 