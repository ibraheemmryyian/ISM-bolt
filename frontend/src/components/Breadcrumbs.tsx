import React from 'react';
import { ChevronRight, Home } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

interface BreadcrumbItem {
  label: string;
  path?: string;
  icon?: React.ReactNode;
}

interface BreadcrumbsProps {
  items?: BreadcrumbItem[];
  className?: string;
}

export function Breadcrumbs({ items, className = '' }: BreadcrumbsProps) {
  const location = useLocation();
  
  // Auto-generate breadcrumbs from pathname if no items provided
  const breadcrumbItems = items || generateBreadcrumbsFromPath(location.pathname);

  return (
    <nav className={`flex items-center space-x-2 text-sm text-gray-600 ${className}`}>
      {breadcrumbItems.map((item, index) => (
        <React.Fragment key={index}>
          {index > 0 && (
            <ChevronRight className="h-4 w-4 text-gray-400" />
          )}
          
          {item.path && index < breadcrumbItems.length - 1 ? (
            <Link
              to={item.path}
              className="flex items-center space-x-1 hover:text-gray-900 transition-colors"
            >
              {item.icon && <span>{item.icon}</span>}
              <span>{item.label}</span>
            </Link>
          ) : (
            <span className="flex items-center space-x-1 text-gray-900 font-medium">
              {item.icon && <span>{item.icon}</span>}
              <span>{item.label}</span>
            </span>
          )}
        </React.Fragment>
      ))}
    </nav>
  );
}

function generateBreadcrumbsFromPath(pathname: string): BreadcrumbItem[] {
  const segments = pathname.split('/').filter(Boolean);
  const breadcrumbs: BreadcrumbItem[] = [
    { label: 'Home', path: '/', icon: <Home className="h-4 w-4" /> }
  ];

  let currentPath = '';
  
  segments.forEach((segment, index) => {
    currentPath += `/${segment}`;
    
    // Convert segment to readable label
    const label = segment
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
    
    breadcrumbs.push({
      label,
      path: index === segments.length - 1 ? undefined : currentPath
    });
  });

  return breadcrumbs;
}

// Predefined breadcrumb configurations for common routes
export const breadcrumbConfigs = {
  dashboard: [
    { label: 'Home', path: '/', icon: <Home className="h-4 w-4" /> },
    { label: 'Dashboard', path: '/dashboard' }
  ],
  
  marketplace: [
    { label: 'Home', path: '/', icon: <Home className="h-4 w-4" /> },
    { label: 'Marketplace', path: '/marketplace' }
  ],
  
  onboarding: [
    { label: 'Home', path: '/', icon: <Home className="h-4 w-4" /> },
    { label: 'Dashboard', path: '/dashboard' },
    { label: 'Onboarding', path: '/onboarding' }
  ],
  
  admin: [
    { label: 'Home', path: '/', icon: <Home className="h-4 w-4" /> },
    { label: 'Admin', path: '/admin' }
  ],
  
  profile: [
    { label: 'Home', path: '/', icon: <Home className="h-4 w-4" /> },
    { label: 'Dashboard', path: '/dashboard' },
    { label: 'Profile', path: '/profile' }
  ],
  
  notifications: [
    { label: 'Home', path: '/', icon: <Home className="h-4 w-4" /> },
    { label: 'Dashboard', path: '/dashboard' },
    { label: 'Notifications', path: '/notifications' }
  ],
  
  chats: [
    { label: 'Home', path: '/', icon: <Home className="h-4 w-4" /> },
    { label: 'Dashboard', path: '/dashboard' },
    { label: 'Chats', path: '/chats' }
  ]
}; 