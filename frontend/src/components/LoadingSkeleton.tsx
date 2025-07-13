import React from 'react';

interface SkeletonProps {
  className?: string;
  width?: string;
  height?: string;
  rounded?: boolean;
}

export function Skeleton({ 
  className = '', 
  width = 'w-full', 
  height = 'h-4', 
  rounded = true 
}: SkeletonProps) {
  return (
    <div 
      className={`
        bg-gray-200 animate-pulse
        ${width} ${height}
        ${rounded ? 'rounded' : ''}
        ${className}
      `}
    />
  );
}

export function CardSkeleton() {
  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <div className="flex items-center space-x-3">
        <Skeleton className="w-12 h-12 rounded-full" />
        <div className="flex-1 space-y-2">
          <Skeleton className="w-3/4" />
          <Skeleton className="w-1/2" />
        </div>
      </div>
      <Skeleton className="w-full h-20" />
      <div className="flex space-x-2">
        <Skeleton className="w-20 h-6" />
        <Skeleton className="w-16 h-6" />
      </div>
    </div>
  );
}

export function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow p-6">
        <Skeleton className="w-64 h-8 mb-4" />
        <Skeleton className="w-full h-4 mb-2" />
        <Skeleton className="w-3/4 h-4" />
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div className="space-y-2">
                <Skeleton className="w-24 h-4" />
                <Skeleton className="w-16 h-6" />
              </div>
              <Skeleton className="w-12 h-12 rounded-lg" />
            </div>
          </div>
        ))}
      </div>

      {/* Content Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {Array.from({ length: 2 }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    </div>
  );
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <Skeleton className="w-32 h-6" />
          <Skeleton className="w-24 h-8" />
        </div>
      </div>

      {/* Table */}
      <div className="divide-y divide-gray-200">
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="px-6 py-4">
            <div className="flex items-center space-x-4">
              <Skeleton className="w-12 h-12 rounded-full" />
              <div className="flex-1 space-y-2">
                <Skeleton className="w-1/3" />
                <Skeleton className="w-1/4" />
              </div>
              <div className="flex space-x-2">
                <Skeleton className="w-16 h-6" />
                <Skeleton className="w-20 h-6" />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function FormSkeleton() {
  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-6">
      <div className="space-y-2">
        <Skeleton className="w-32 h-5" />
        <Skeleton className="w-full h-10" />
      </div>
      
      <div className="space-y-2">
        <Skeleton className="w-24 h-5" />
        <Skeleton className="w-full h-10" />
      </div>
      
      <div className="space-y-2">
        <Skeleton className="w-28 h-5" />
        <Skeleton className="w-full h-20" />
      </div>
      
      <div className="flex space-x-3">
        <Skeleton className="w-24 h-10" />
        <Skeleton className="w-20 h-10" />
      </div>
    </div>
  );
}

export function MarketplaceSkeleton() {
  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex flex-wrap gap-4">
          <Skeleton className="w-32 h-10" />
          <Skeleton className="w-40 h-10" />
          <Skeleton className="w-28 h-10" />
          <Skeleton className="w-24 h-10" />
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    </div>
  );
}

export function ProfileSkeleton() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center space-x-4">
          <Skeleton className="w-20 h-20 rounded-full" />
          <div className="flex-1 space-y-2">
            <Skeleton className="w-48 h-6" />
            <Skeleton className="w-32 h-4" />
            <Skeleton className="w-64 h-4" />
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="bg-white rounded-lg shadow p-6 text-center">
            <Skeleton className="w-16 h-16 rounded-full mx-auto mb-4" />
            <Skeleton className="w-24 h-6 mx-auto mb-2" />
            <Skeleton className="w-20 h-4 mx-auto" />
          </div>
        ))}
      </div>

      {/* Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FormSkeleton />
        <CardSkeleton />
      </div>
    </div>
  );
}

// Default export for backward compatibility
export default function LoadingSkeleton() {
  return <DashboardSkeleton />;
} 