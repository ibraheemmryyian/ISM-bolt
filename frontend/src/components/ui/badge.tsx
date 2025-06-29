import React from 'react';

type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & {
  variant?: 'default' | 'outline';
};

export const Badge: React.FC<BadgeProps> = ({ variant = 'default', className = '', ...props }) => {
  const base = 'inline-block px-2 py-0.5 rounded text-xs font-medium';
  const styles = variant === 'outline' 
    ? 'border border-emerald-200 text-emerald-700 bg-emerald-50'
    : 'bg-emerald-100 text-emerald-700';
    
  return <span className={`${base} ${styles} ${className}`} {...props} />;
}; 