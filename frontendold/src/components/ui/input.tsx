import React from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  error?: boolean;
  errorMessage?: string;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className = '', error, errorMessage, ...props }, ref) => (
    <div className="w-full">
      <input
        ref={ref}
        className={`block w-full px-3 py-2 border rounded-md shadow-sm focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 sm:text-sm transition-colors ${
          error 
            ? 'border-red-300 focus:ring-red-500 focus:border-red-500' 
            : 'border-gray-300'
        } ${className}`}
        {...props}
      />
      {error && errorMessage && (
        <p className="mt-1 text-sm text-red-600">{errorMessage}</p>
      )}
    </div>
  )
);

Input.displayName = 'Input'; 