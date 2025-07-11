import React, { useState, useEffect } from 'react';
import { Search, Filter, X, ChevronDown, ChevronUp } from 'lucide-react';

interface FilterOption {
  value: string;
  label: string;
  count?: number;
}

interface FilterGroup {
  id: string;
  label: string;
  options: FilterOption[];
  multiSelect?: boolean;
}

interface SearchAndFilterProps {
  onSearch: (query: string) => void;
  onFilterChange: (filters: Record<string, string[]>) => void;
  filters?: FilterGroup[];
  placeholder?: string;
  className?: string;
  showFilters?: boolean;
}

export function SearchAndFilter({
  onSearch,
  onFilterChange,
  filters = [],
  placeholder = 'Search...',
  className = '',
  showFilters = true
}: SearchAndFilterProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilters, setActiveFilters] = useState<Record<string, string[]>>({});
  const [expandedFilters, setExpandedFilters] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      onSearch(searchQuery);
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery, onSearch]);

  useEffect(() => {
    onFilterChange(activeFilters);
  }, [activeFilters, onFilterChange]);

  const handleFilterToggle = (filterId: string, value: string, multiSelect = false) => {
    setActiveFilters(prev => {
      const currentValues = prev[filterId] || [];
      
      if (multiSelect) {
        const newValues = currentValues.includes(value)
          ? currentValues.filter(v => v !== value)
          : [...currentValues, value];
        
        return {
          ...prev,
          [filterId]: newValues.length > 0 ? newValues : []
        };
      } else {
        return {
          ...prev,
          [filterId]: currentValues.includes(value) ? [] : [value]
        };
      }
    });
  };

  const clearAllFilters = () => {
    setActiveFilters({});
  };

  const getActiveFilterCount = () => {
    return Object.values(activeFilters).reduce((total, values) => total + values.length, 0);
  };

  const toggleFilterGroup = (filterId: string) => {
    setExpandedFilters(prev => ({
      ...prev,
      [filterId]: !prev[filterId]
    }));
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder={placeholder}
          className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          data-search-input
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery('')}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Filters */}
      {showFilters && filters.length > 0 && (
        <div className="space-y-4">
          {/* Filter Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-gray-500" />
              <span className="text-sm font-medium text-gray-700">Filters</span>
              {getActiveFilterCount() > 0 && (
                <span className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                  {getActiveFilterCount()}
                </span>
              )}
            </div>
            {getActiveFilterCount() > 0 && (
              <button
                onClick={clearAllFilters}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Clear all
              </button>
            )}
          </div>

          {/* Filter Groups */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filters.map(filterGroup => (
              <div key={filterGroup.id} className="bg-white border border-gray-200 rounded-lg">
                <button
                  onClick={() => toggleFilterGroup(filterGroup.id)}
                  className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-50"
                >
                  <span className="font-medium text-gray-700">{filterGroup.label}</span>
                  {expandedFilters[filterGroup.id] ? (
                    <ChevronUp className="h-4 w-4 text-gray-500" />
                  ) : (
                    <ChevronDown className="h-4 w-4 text-gray-500" />
                  )}
                </button>
                
                {expandedFilters[filterGroup.id] && (
                  <div className="px-4 pb-3 space-y-2">
                    {filterGroup.options.map(option => {
                      const isSelected = (activeFilters[filterGroup.id] || []).includes(option.value);
                      return (
                        <label
                          key={option.value}
                          className="flex items-center space-x-2 cursor-pointer hover:bg-gray-50 p-2 rounded"
                        >
                          <input
                            type={filterGroup.multiSelect ? 'checkbox' : 'radio'}
                            checked={isSelected}
                            onChange={() => handleFilterToggle(filterGroup.id, option.value, filterGroup.multiSelect)}
                            className="text-blue-600 focus:ring-blue-500"
                          />
                          <span className="text-sm text-gray-700">{option.label}</span>
                          {option.count !== undefined && (
                            <span className="text-xs text-gray-500">({option.count})</span>
                          )}
                        </label>
                      );
                    })}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Specialized search components
export function MarketplaceSearch({ onSearch, onFilterChange }: {
  onSearch: (query: string) => void;
  onFilterChange: (filters: Record<string, string[]>) => void;
}) {
  const filters: FilterGroup[] = [
    {
      id: 'type',
      label: 'Material Type',
      options: [
        { value: 'waste', label: 'Waste Streams', count: 45 },
        { value: 'requirement', label: 'Requirements', count: 32 },
        { value: 'byproduct', label: 'Byproducts', count: 18 }
      ]
    },
    {
      id: 'industry',
      label: 'Industry',
      multiSelect: true,
      options: [
        { value: 'chemical', label: 'Chemical', count: 23 },
        { value: 'manufacturing', label: 'Manufacturing', count: 34 },
        { value: 'food', label: 'Food & Beverage', count: 12 },
        { value: 'textile', label: 'Textile', count: 8 },
        { value: 'metal', label: 'Metal Processing', count: 15 }
      ]
    },
    {
      id: 'location',
      label: 'Location',
      options: [
        { value: 'local', label: 'Local (50km)', count: 28 },
        { value: 'regional', label: 'Regional (200km)', count: 45 },
        { value: 'national', label: 'National', count: 67 }
      ]
    }
  ];

  return (
    <SearchAndFilter
      onSearch={onSearch}
      onFilterChange={onFilterChange}
      filters={filters}
      placeholder="Search materials, companies, or industries..."
    />
  );
}

export function CompanySearch({ onSearch, onFilterChange }: {
  onSearch: (query: string) => void;
  onFilterChange: (filters: Record<string, string[]>) => void;
}) {
  const filters: FilterGroup[] = [
    {
      id: 'size',
      label: 'Company Size',
      options: [
        { value: 'small', label: 'Small (<50 employees)', count: 23 },
        { value: 'medium', label: 'Medium (50-500)', count: 45 },
        { value: 'large', label: 'Large (>500)', count: 12 }
      ]
    },
    {
      id: 'sustainability',
      label: 'Sustainability Level',
      options: [
        { value: 'beginner', label: 'Beginner', count: 18 },
        { value: 'intermediate', label: 'Intermediate', count: 34 },
        { value: 'advanced', label: 'Advanced', count: 28 }
      ]
    },
    {
      id: 'partnerships',
      label: 'Partnership Status',
      multiSelect: true,
      options: [
        { value: 'active', label: 'Active Partnerships', count: 42 },
        { value: 'seeking', label: 'Seeking Partners', count: 31 },
        { value: 'open', label: 'Open to Partnerships', count: 56 }
      ]
    }
  ];

  return (
    <SearchAndFilter
      onSearch={onSearch}
      onFilterChange={onFilterChange}
      filters={filters}
      placeholder="Search companies by name, industry, or location..."
    />
  );
} 