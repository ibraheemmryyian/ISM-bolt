import React from 'react';

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'default' | 'outline' | 'ghost' | 'destructive' | 'secondary';
  size?: 'sm' | 'md' | 'lg';
  asChild?: boolean;
};

export const Button: React.FC<ButtonProps> = ({ 
  variant = 'default', 
  size = 'md', 
  className = '', 
  disabled,
  ...props 
}) => {
  const base = 'inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none';
  
  const sizeStyles = {
    sm: 'h-8 px-3 text-xs',
    md: 'h-10 px-4 py-2',
    lg: 'h-12 px-8 text-lg'
  };
  
  const variantStyles = {
    default: 'bg-emerald-600 text-white hover:bg-emerald-700',
    outline: 'border border-emerald-600 text-emerald-600 bg-white hover:bg-emerald-50',
    ghost: 'text-emerald-600 hover:bg-emerald-50',
    destructive: 'bg-red-600 text-white hover:bg-red-700',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200'
  };
      
  return (
    <button 
      className={`${base} ${sizeStyles[size]} ${variantStyles[variant]} ${className}`} 
      disabled={disabled}
      {...props} 
    />
  );
}; 