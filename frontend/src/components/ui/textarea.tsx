import React from 'react';

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  error?: boolean;
  errorMessage?: string;
  success?: boolean;
  successMessage?: string;
  label?: string;
  helperText?: string;
  maxLength?: number;
  showCharacterCount?: boolean;
  autoResize?: boolean;
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ 
    className = '', 
    error, 
    errorMessage, 
    success,
    successMessage,
    label,
    helperText,
    maxLength,
    showCharacterCount = false,
    autoResize = false,
    id,
    onChange,
    ...props 
  }, ref) => {
    const inputId = id || `textarea-${Math.random().toString(36).substr(2, 9)}`;
    const [value, setValue] = React.useState(props.value || '');
    
    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      setValue(e.target.value);
      if (onChange) {
        onChange(e);
      }
      if (autoResize) {
        e.target.style.height = 'auto';
        e.target.style.height = `${e.target.scrollHeight}px`;
      }
    };

    const characterCount = typeof value === 'string' ? value.length : 0;
    
    return (
      <div className="w-full">
        {label && (
          <label htmlFor={inputId} className="block text-sm font-medium text-gray-700 mb-1">
            {label}
          </label>
        )}
        <textarea
          id={inputId}
          ref={ref}
          className={`block w-full px-3 py-2 border rounded-md shadow-sm focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 sm:text-sm transition-all duration-200 resize-vertical min-h-[80px] ${
            error 
              ? 'border-red-300 focus:ring-red-500 focus:border-red-500' 
              : success
              ? 'border-green-300 focus:ring-green-500 focus:border-green-500'
              : 'border-gray-300'
          } ${className}`}
          onChange={handleChange}
          value={value}
          maxLength={maxLength}
          {...props}
        />
        <div className="flex justify-between items-center mt-1">
          <div>
            {error && errorMessage && (
              <p className="text-sm text-red-600 flex items-center">
                <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                {errorMessage}
              </p>
            )}
            {success && successMessage && (
              <p className="text-sm text-green-600 flex items-center">
                <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                {successMessage}
              </p>
            )}
            {helperText && !error && !success && (
              <p className="text-sm text-gray-500">{helperText}</p>
            )}
          </div>
          {showCharacterCount && maxLength && (
            <span className={`text-xs ${
              characterCount > maxLength * 0.9 ? 'text-red-500' : 'text-gray-400'
            }`}>
              {characterCount}/{maxLength}
            </span>
          )}
        </div>
      </div>
    );
  }
);

Textarea.displayName = 'Textarea'; 