import React from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: boolean;
  errorMessage?: string;
  success?: boolean;
  successMessage?: string;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  label?: string;
  helperText?: string;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ 
    className = '', 
    error, 
    errorMessage, 
    success,
    successMessage,
    icon,
    iconPosition = 'left',
    label,
    helperText,
    id,
    ...props 
  }, ref) => {
    const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
    
    return (
      <div className="w-full">
        {label && (
          <label htmlFor={inputId} className="block text-sm font-medium text-gray-700 mb-1">
            {label}
          </label>
        )}
        <div className="relative">
          {icon && iconPosition === 'left' && (
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <span className="text-gray-400">{icon}</span>
            </div>
          )}
          <input
            id={inputId}
            ref={ref}
            className={`block w-full px-3 py-2 border rounded-md shadow-sm focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 sm:text-sm transition-all duration-200 ${
              icon && iconPosition === 'left' ? 'pl-10' : ''
            } ${
              icon && iconPosition === 'right' ? 'pr-10' : ''
            } ${
              error 
                ? 'border-red-300 focus:ring-red-500 focus:border-red-500' 
                : success
                ? 'border-green-300 focus:ring-green-500 focus:border-green-500'
                : 'border-gray-300'
            } ${className}`}
            {...props}
          />
          {icon && iconPosition === 'right' && (
            <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
              <span className="text-gray-400">{icon}</span>
            </div>
          )}
        </div>
        {error && errorMessage && (
          <p className="mt-1 text-sm text-red-600 flex items-center">
            <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            {errorMessage}
          </p>
        )}
        {success && successMessage && (
          <p className="mt-1 text-sm text-green-600 flex items-center">
            <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            {successMessage}
          </p>
        )}
        {helperText && !error && !success && (
          <p className="mt-1 text-sm text-gray-500">{helperText}</p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input'; 