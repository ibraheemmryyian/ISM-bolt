import React from 'react';

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'default' | 'outline';
  size?: 'sm' | 'md' | 'lg';
};

export const Button: React.FC<ButtonProps> = ({ variant = 'default', size = 'md', className = '', ...props }) => {
  const base = 'px-4 py-2 rounded font-semibold transition focus:outline-none';
  
  const sizeStyles = {
    sm: 'px-2 py-1 text-sm',
    md: 'px-4 py-2',
    lg: 'px-6 py-3 text-lg'
  };
  
  const styles =
    variant === 'outline'
      ? 'border border-emerald-600 text-emerald-600 bg-white hover:bg-emerald-50'
      : 'bg-emerald-600 text-white hover:bg-emerald-700';
      
  return <button className={`${base} ${sizeStyles[size]} ${styles} ${className}`} {...props} />;
}; 