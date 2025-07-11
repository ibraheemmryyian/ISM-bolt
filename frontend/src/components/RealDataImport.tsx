import React, { useState, useCallback, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { 
  Upload, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  TrendingUp, 
  Users, 
  DollarSign,
  Activity,
  Target,
  BarChart3,
  Network,
  FileText,
  Download,
  RefreshCw,
  Eye,
  Play
} from 'lucide-react';

interface CompanyData {
  id?: string;
  name: string;
  industry: string;
  location: string;
  size: string;
  contact_info: {
    email: string;
    phone?: string;
    website?: string;
  };
  business_description: string;
  waste_streams?: Array<{
    name: string;
    quantity: number;
    unit: string;
    type: string;
    description?: string;
  }>;
  resource_needs?: Array<{
    name: string;
    quantity: number;
    unit: string;
    type: string;
    description?: string;
  }>;
}

interface ImportResult {
  import_id: string;
  summary: {
    total_companies: number;
    successful: number;
    failed: number;
    success_rate: number;
    processing_time: number;
    total_potential_value: number;
  };
  insights: {
    market_analysis: any;
    symbiosis_network: any;
    high_value_targets: any[];
  };
  next_actions: any[];
}

interface HighValueTarget {
  id: string;
  name: string;
  industry: string;
  location: string;
  potential_savings: number;
  symbiosis_score: number;
  carbon_reduction: number;
  data_quality_score: number;
  contact_info: any;
}

const RealDataImport: React.FC = () => {
  const [companies, setCompanies] = useState<CompanyData[]>([]);
  const [importing, setImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(0);
  const [importResult, setImportResult] = useState<ImportResult | null>(null);
  const [highValueTargets, setHighValueTargets] = useState<HighValueTarget[]>([]);
  const [marketAnalysis, setMarketAnalysis] = useState<any>(null);
  const [symbiosisNetwork, setSymbiosisNetwork] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('import');
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [systemHealth, setSystemHealth] = useState<any>(null);

  // Load system health on component mount
  useEffect(() => {
    checkSystemHealth();
    loadHighValueTargets();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await fetch('/api/monitoring/health');
      const health = await response.json();
      setSystemHealth(health);
    } catch (error) {
      console.error('Failed to check system health:', error);
    }
  };

  const loadHighValueTargets = async () => {
    try {
      const response = await fetch('/api/real-data/high-value-targets?limit=10');
      const data = await response.json();
      if (data.success) {
        setHighValueTargets(data.data);
      }
    } catch (error) {
      console.error('Failed to load high-value targets:', error);
    }
  };

  const validateCompanyData = (company: CompanyData): string[] => {
    const errors: string[] = [];

    if (!company.name?.trim()) {
      errors.push('Company name is required');
    }

    if (!company.industry?.trim()) {
      errors.push('Industry is required');
    }

    if (!company.location?.trim()) {
      errors.push('Location is required');
    }

    if (!company.contact_info?.email?.trim()) {
      errors.push('Email is required');
    }

    if (company.waste_streams) {
      company.waste_streams.forEach((waste, index) => {
        if (!waste.name?.trim()) {
          errors.push(`Waste stream ${index + 1}: Name is required`);
        }
        if (!waste.quantity || waste.quantity <= 0) {
          errors.push(`Waste stream ${index + 1}: Valid quantity is required`);
        }
        if (!waste.unit?.trim()) {
          errors.push(`Waste stream ${index + 1}: Unit is required`);
        }
      });
    }

    if (company.resource_needs) {
      company.resource_needs.forEach((need, index) => {
        if (!need.name?.trim()) {
          errors.push(`Resource need ${index + 1}: Name is required`);
        }
        if (!need.quantity || need.quantity <= 0) {
          errors.push(`Resource need ${index + 1}: Valid quantity is required`);
        }
        if (!need.unit?.trim()) {
          errors.push(`Resource need ${index + 1}: Unit is required`);
        }
      });
    }

    return errors;
  };

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        let data: CompanyData[];

        if (file.name.endsWith('.json')) {
          data = JSON.parse(content);
        } else if (file.name.endsWith('.csv')) {
          // Simple CSV parsing - in production, use a proper CSV library
          const lines = content.split('\n');
          const headers = lines[0].split(',');
          data = lines.slice(1).map(line => {
            const values = line.split(',');
            const company: any = {};
            headers.forEach((header, index) => {
              company[header.trim()] = values[index]?.trim();
            });
            return company;
          });
        } else {
          throw new Error('Unsupported file format. Please use JSON or CSV.');
        }

        // Validate all companies
        const allErrors: string[] = [];
        data.forEach((company, index) => {
          const errors = validateCompanyData(company);
          errors.forEach(error => {
            allErrors.push(`Company ${index + 1} (${company.name}): ${error}`);
          });
        });

        setValidationErrors(allErrors);
        setCompanies(data);

        if (allErrors.length === 0) {
          console.log(`✅ Loaded ${data.length} valid companies`);
        } else {
          console.warn(`⚠️ Loaded ${data.length} companies with ${allErrors.length} validation errors`);
        }
      } catch (error) {
        console.error('Failed to parse file:', error);
        setValidationErrors([`Failed to parse file: ${error}`]);
      }
    };

    reader.readAsText(file);
  }, []);

  const startBulkImport = async () => {
    if (companies.length === 0) {
      alert('Please upload company data first');
      return;
    }

    if (validationErrors.length > 0) {
      alert('Please fix validation errors before importing');
      return;
    }

    setImporting(true);
    setImportProgress(0);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setImportProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + Math.random() * 10;
        });
      }, 1000);

      const response = await fetch('/api/real-data/bulk-import', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          companies: companies,
          options: {
            validate_data: true,
            enrich_with_apis: true,
            generate_insights: true,
            calculate_metrics: true
          }
        }),
      });

      clearInterval(progressInterval);
      setImportProgress(100);

      const result = await response.json();

      if (result.success) {
        setImportResult(result.data);
        setActiveTab('results');
        
        // Load updated data
        loadHighValueTargets();
        
        console.log('🎉 Bulk import completed successfully!');
        console.log(`Processed ${result.data.summary.successful} companies`);
        console.log(`Total potential value: $${result.data.summary.total_potential_value.toLocaleString()}`);
      } else {
        throw new Error(result.error || 'Import failed');
      }
    } catch (error) {
      console.error('Import failed:', error);
      alert(`Import failed: ${error}`);
    } finally {
      setImporting(false);
      setImportProgress(0);
    }
  };

  const downloadTemplate = () => {
    const template: CompanyData = {
      name: "Example Company",
      industry: "manufacturing",
      location: "City, Country",
      size: "medium",
      contact_info: {
        email: "contact@example.com",
        phone: "+1234567890",
        website: "https://example.com"
      },
      business_description: "Example company description",
      waste_streams: [
        {
          name: "Steel scrap",
          quantity: 1000,
          unit: "kg",
          type: "waste",
          description: "Steel manufacturing waste"
        }
      ],
      resource_needs: [
        {
          name: "Recycled steel",
          quantity: 500,
          unit: "kg",
          type: "need",
          description: "Need for recycled steel"
        }
      ]
    };

    const blob = new Blob([JSON.stringify([template], null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'company-data-template.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Real Data Import</h1>
          <p className="text-gray-600 mt-2">
            Import and process 50 real company profiles for maximum value extraction
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant={systemHealth?.status === 'healthy' ? 'default' : 'destructive'}>
            {systemHealth?.status === 'healthy' ? 'System Healthy' : 'System Issues'}
          </Badge>
          <Button variant="outline" size="sm" onClick={checkSystemHealth}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="import">Import Data</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
          <TabsTrigger value="targets">High-Value Targets</TabsTrigger>
          <TabsTrigger value="analysis">Market Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="import" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Upload className="h-5 w-5" />
                <span>Upload Company Data</span>
              </CardTitle>
              <CardDescription>
                Upload your 50 real company profiles in JSON or CSV format
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                <input
                  type="file"
                  accept=".json,.csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-lg font-medium text-gray-900">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-sm text-gray-500">
                    JSON or CSV files up to 10MB
                  </p>
                </label>
              </div>

              {companies.length > 0 && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-medium">Loaded Companies: {companies.length}</h3>
                    <Button variant="outline" size="sm" onClick={downloadTemplate}>
                      <Download className="h-4 w-4 mr-2" />
                      Download Template
                    </Button>
                  </div>

                  {validationErrors.length > 0 && (
                    <Alert>
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        <strong>Validation Errors ({validationErrors.length}):</strong>
                        <ul className="mt-2 space-y-1">
                          {validationErrors.slice(0, 5).map((error, index) => (
                            <li key={index} className="text-sm text-red-600">• {error}</li>
                          ))}
                          {validationErrors.length > 5 && (
                            <li className="text-sm text-red-600">
                              ... and {validationErrors.length - 5} more errors
                            </li>
                          )}
                        </ul>
                      </AlertDescription>
                    </Alert>
                  )}

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {companies.slice(0, 6).map((company, index) => (
                      <Card key={index} className="p-4">
                        <h4 className="font-medium truncate">{company.name}</h4>
                        <p className="text-sm text-gray-500">{company.industry}</p>
                        <p className="text-sm text-gray-500">{company.location}</p>
                        <div className="flex items-center space-x-2 mt-2">
                          <Badge variant="outline" className="text-xs">
                            {company.waste_streams?.length || 0} waste streams
                          </Badge>
                          <Badge variant="outline" className="text-xs">
                            {company.resource_needs?.length || 0} resource needs
                          </Badge>
                        </div>
                      </Card>
                    ))}
                    {companies.length > 6 && (
                      <Card className="p-4 flex items-center justify-center">
                        <p className="text-sm text-gray-500">
                          +{companies.length - 6} more companies
                        </p>
                      </Card>
                    )}
                  </div>

                  <Button 
                    onClick={startBulkImport} 
                    disabled={importing || validationErrors.length > 0}
                    className="w-full"
                  >
                    {importing ? (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                        Processing {companies.length} Companies...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Start Bulk Import
                      </>
                    )}
                  </Button>

                  {importing && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Processing companies...</span>
                        <span>{Math.round(importProgress)}%</span>
                      </div>
                      <Progress value={importProgress} />
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="results" className="space-y-6">
          {importResult ? (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    <span>Import Results</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {importResult.summary.successful}
                      </div>
                      <div className="text-sm text-gray-500">Successful</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-red-600">
                        {importResult.summary.failed}
                      </div>
                      <div className="text-sm text-gray-500">Failed</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {importResult.summary.success_rate.toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-500">Success Rate</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {formatCurrency(importResult.summary.total_potential_value)}
                      </div>
                      <div className="text-sm text-gray-500">Total Potential Value</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Next Actions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {importResult.next_actions.map((action, index) => (
                      <div key={index} className="flex items-start space-x-3 p-3 border rounded-lg">
                        <div className={`w-2 h-2 rounded-full mt-2 ${
                          action.priority === 'high' ? 'bg-red-500' : 
                          action.priority === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                        }`} />
                        <div className="flex-1">
                          <h4 className="font-medium">{action.action}</h4>
                          <p className="text-sm text-gray-600">{action.description}</p>
                          <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                            <span>Timeline: {action.timeline}</span>
                            {action.estimated_value && (
                              <span>Value: {formatCurrency(action.estimated_value)}</span>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : (
            <Card>
              <CardContent className="text-center py-8">
                <Clock className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                <p className="text-gray-500">No import results available. Start an import to see results.</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="targets" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Target className="h-5 w-5" />
                <span>High-Value Targets</span>
              </CardTitle>
              <CardDescription>
                Companies with the highest potential for symbiosis opportunities
              </CardDescription>
            </CardHeader>
            <CardContent>
              {highValueTargets.length > 0 ? (
                <div className="space-y-4">
                  {highValueTargets.map((target) => (
                    <Card key={target.id} className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h4 className="font-medium">{target.name}</h4>
                          <p className="text-sm text-gray-500">{target.industry} • {target.location}</p>
                          <div className="flex items-center space-x-4 mt-2">
                            <Badge variant="outline" className="text-xs">
                              <DollarSign className="h-3 w-3 mr-1" />
                              {formatCurrency(target.potential_savings)}
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              <Activity className="h-3 w-3 mr-1" />
                              {target.symbiosis_score.toFixed(2)}
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              <TrendingUp className="h-3 w-3 mr-1" />
                              {formatNumber(target.carbon_reduction)} kg CO2e
                            </Badge>
                          </div>
                        </div>
                        <div className="flex space-x-2">
                          <Button variant="outline" size="sm">
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button size="sm">
                            Contact
                          </Button>
                        </div>
                      </div>
                    </Card>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Target className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-500">No high-value targets found. Import company data to see targets.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5" />
                  <span>Market Analysis</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {marketAnalysis ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">
                          {formatCurrency(marketAnalysis.total_waste_value)}
                        </div>
                        <div className="text-sm text-gray-500">Total Waste Value</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                          {formatCurrency(marketAnalysis.total_potential_savings)}
                        </div>
                        <div className="text-sm text-gray-500">Potential Savings</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <BarChart3 className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                    <p className="text-gray-500">No market analysis available</p>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Network className="h-5 w-5" />
                  <span>Symbiosis Network</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {symbiosisNetwork ? (
                  <div className="space-y-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {symbiosisNetwork.potential_partnerships?.length || 0}
                      </div>
                      <div className="text-sm text-gray-500">Potential Partnerships</div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Network className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                    <p className="text-gray-500">No symbiosis network data available</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RealDataImport; 