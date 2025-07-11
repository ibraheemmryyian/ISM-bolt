import React from 'react';

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'default' | 'outline' | 'ghost' | 'destructive' | 'secondary' | 'premium' | 'gradient';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  asChild?: boolean;
  loading?: boolean;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
};

export const Button: React.FC<ButtonProps> = ({ 
  variant = 'default', 
  size = 'md', 
  className = '', 
  disabled,
  loading = false,
  icon,
  iconPosition = 'left',
  children,
  ...props 
}) => {
  const base = 'inline-flex items-center justify-center rounded-md font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none active:scale-95';
  
  const sizeStyles = {
    sm: 'h-8 px-3 text-xs',
    md: 'h-10 px-4 py-2',
    lg: 'h-12 px-6 py-3 text-base',
    xl: 'h-14 px-8 py-4 text-lg'
  };
  
  const variantStyles = {
    default: 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-md hover:shadow-lg',
    outline: 'border-2 border-emerald-600 text-emerald-600 bg-white hover:bg-emerald-50 hover:border-emerald-700',
    ghost: 'text-emerald-600 hover:bg-emerald-50 hover:text-emerald-700',
    destructive: 'bg-red-600 text-white hover:bg-red-700 shadow-md hover:shadow-lg',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 border border-gray-300',
    premium: 'bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:from-purple-700 hover:to-blue-700 shadow-lg hover:shadow-xl',
    gradient: 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white hover:from-emerald-600 hover:to-teal-600 shadow-md hover:shadow-lg'
  };

  const isDisabled = disabled || loading;
      
  return (
    <button 
      className={`${base} ${sizeStyles[size]} ${variantStyles[variant]} ${className}`} 
      disabled={isDisabled}
      {...props} 
    >
      {loading && (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      )}
      {!loading && icon && iconPosition === 'left' && (
        <span className="mr-2">{icon}</span>
      )}
      {children}
      {!loading && icon && iconPosition === 'right' && (
        <span className="ml-2">{icon}</span>
      )}
    </button>
  );
}; 