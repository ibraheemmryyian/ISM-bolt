import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Alert, AlertDescription } from './ui/alert';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Truck, 
  Leaf, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Target,
  BarChart3,
  Calculator,
  MapPin,
  Scale,
  Zap
} from 'lucide-react';

interface CostBreakdown {
  waste_disposal_cost: number;
  transportation_cost: number;
  processing_cost: number;
  storage_cost: number;
  regulatory_fees: number;
  insurance_cost: number;
  opportunity_cost: number;
  carbon_tax: number;
  total_cost: number;
}

interface SymbiosisAnalysis {
  company_id: string;
  partner_id: string;
  material_id: string;
  traditional_cost: CostBreakdown;
  symbiosis_cost: CostBreakdown;
  net_savings: number;
  payback_period_months: number;
  roi_percentage: number;
  carbon_savings_kg: number;
  carbon_savings_value: number;
  risk_score: number;
  confidence_level: number;
  recommendations: string[];
}

interface FinancialAnalysisPanelProps {
  analysis?: SymbiosisAnalysis;
  loading?: boolean;
  onRefresh?: () => void;
}

const FinancialAnalysisPanel: React.FC<FinancialAnalysisPanelProps> = ({
  analysis,
  loading = false,
  onRefresh
}) => {
  const [activeTab, setActiveTab] = useState('overview');

  if (loading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calculator className="h-5 w-5" />
            Financial Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            <span className="ml-2">Calculating financial analysis...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!analysis) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calculator className="h-5 w-5" />
            Financial Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              No financial analysis available. Select a symbiosis opportunity to analyze.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(Math.round(num));
  };

  const getRiskColor = (risk: number) => {
    if (risk < 0.3) return 'text-green-600 bg-green-100';
    if (risk < 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-600 bg-green-100';
    if (confidence > 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getRecommendationIcon = (recommendation: string) => {
    if (recommendation.includes('RECOMMENDED') || recommendation.includes('✅')) {
      return <CheckCircle className="h-4 w-4 text-green-600" />;
    }
    if (recommendation.includes('NOT RECOMMENDED') || recommendation.includes('❌')) {
      return <XCircle className="h-4 w-4 text-red-600" />;
    }
    return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Calculator className="h-5 w-5" />
            Financial Analysis
          </CardTitle>
          {onRefresh && (
            <Button variant="outline" size="sm" onClick={onRefresh}>
              <BarChart3 className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="breakdown">Cost Breakdown</TabsTrigger>
            <TabsTrigger value="roi">ROI Analysis</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <DollarSign className="h-4 w-4 text-green-600" />
                    <span className="text-sm font-medium">Monthly Savings</span>
                  </div>
                  <div className="text-2xl font-bold text-green-600">
                    {formatCurrency(analysis.net_savings)}
                  </div>
                  <div className="text-xs text-muted-foreground">vs traditional disposal</div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <Target className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium">ROI</span>
                  </div>
                  <div className="text-2xl font-bold text-blue-600">
                    {analysis.roi_percentage.toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground">annual return</div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-orange-600" />
                    <span className="text-sm font-medium">Payback</span>
                  </div>
                  <div className="text-2xl font-bold text-orange-600">
                    {analysis.payback_period_months.toFixed(1)}m
                  </div>
                  <div className="text-xs text-muted-foreground">to break even</div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <Leaf className="h-4 w-4 text-emerald-600" />
                    <span className="text-sm font-medium">Carbon Saved</span>
                  </div>
                  <div className="text-2xl font-bold text-emerald-600">
                    {formatNumber(analysis.carbon_savings_kg)} kg
                  </div>
                  <div className="text-xs text-muted-foreground">CO2 equivalent</div>
                </CardContent>
              </Card>
            </div>

            {/* Cost Comparison Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Cost Comparison</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-red-500 rounded"></div>
                      <span className="font-medium">Traditional Disposal</span>
                    </div>
                    <span className="text-lg font-bold text-red-600">
                      {formatCurrency(analysis.traditional_cost.total_cost)}
                    </span>
                  </div>
                  <Progress 
                    value={(analysis.traditional_cost.total_cost / (analysis.traditional_cost.total_cost + analysis.symbiosis_cost.total_cost)) * 100} 
                    className="h-3"
                  />
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-green-500 rounded"></div>
                      <span className="font-medium">Symbiosis Partnership</span>
                    </div>
                    <span className="text-lg font-bold text-green-600">
                      {formatCurrency(analysis.symbiosis_cost.total_cost)}
                    </span>
                  </div>
                  <Progress 
                    value={(analysis.symbiosis_cost.total_cost / (analysis.traditional_cost.total_cost + analysis.symbiosis_cost.total_cost)) * 100} 
                    className="h-3 bg-green-100"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Risk and Confidence */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5" />
                    Risk Assessment
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span>Risk Score</span>
                      <Badge className={getRiskColor(analysis.risk_score)}>
                        {(analysis.risk_score * 100).toFixed(0)}%
                      </Badge>
                    </div>
                    <Progress value={analysis.risk_score * 100} className="h-2" />
                    <p className="text-sm text-muted-foreground">
                      {analysis.risk_score < 0.3 ? 'Low risk - proceed with confidence' :
                       analysis.risk_score < 0.6 ? 'Moderate risk - monitor closely' :
                       'High risk - consider mitigation strategies'}
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Scale className="h-5 w-5" />
                    Analysis Confidence
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span>Confidence Level</span>
                      <Badge className={getConfidenceColor(analysis.confidence_level)}>
                        {(analysis.confidence_level * 100).toFixed(0)}%
                      </Badge>
                    </div>
                    <Progress value={analysis.confidence_level * 100} className="h-2" />
                    <p className="text-sm text-muted-foreground">
                      {analysis.confidence_level > 0.8 ? 'High confidence in analysis' :
                       analysis.confidence_level > 0.6 ? 'Moderate confidence - verify key data' :
                       'Low confidence - manual review recommended'}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="breakdown" className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Traditional Disposal Breakdown */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2 text-red-600">
                    <XCircle className="h-5 w-5" />
                    Traditional Disposal Costs
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {Object.entries(analysis.traditional_cost).map(([key, value]) => {
                    if (key === 'total_cost') return null;
                    const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    return (
                      <div key={key} className="flex items-center justify-between">
                        <span className="text-sm">{label}</span>
                        <span className="font-medium">{formatCurrency(value)}</span>
                      </div>
                    );
                  })}
                  <div className="border-t pt-3 flex items-center justify-between">
                    <span className="font-semibold">Total Cost</span>
                    <span className="text-lg font-bold text-red-600">
                      {formatCurrency(analysis.traditional_cost.total_cost)}
                    </span>
                  </div>
                </CardContent>
              </Card>

              {/* Symbiosis Partnership Breakdown */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2 text-green-600">
                    <CheckCircle className="h-5 w-5" />
                    Symbiosis Partnership Costs
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {Object.entries(analysis.symbiosis_cost).map(([key, value]) => {
                    if (key === 'total_cost') return null;
                    const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    return (
                      <div key={key} className="flex items-center justify-between">
                        <span className="text-sm">{label}</span>
                        <span className={`font-medium ${value < 0 ? 'text-green-600' : ''}`}>
                          {formatCurrency(Math.abs(value))}
                          {value < 0 && ' (revenue)'}
                        </span>
                      </div>
                    );
                  })}
                  <div className="border-t pt-3 flex items-center justify-between">
                    <span className="font-semibold">Total Cost</span>
                    <span className="text-lg font-bold text-green-600">
                      {formatCurrency(analysis.symbiosis_cost.total_cost)}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="roi" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* ROI Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Return on Investment
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center">
                    <div className="text-4xl font-bold text-blue-600">
                      {analysis.roi_percentage.toFixed(1)}%
                    </div>
                    <div className="text-sm text-muted-foreground">Annual ROI</div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span>Monthly Savings</span>
                      <span className="font-medium">{formatCurrency(analysis.net_savings)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Annual Savings</span>
                      <span className="font-medium">{formatCurrency(analysis.net_savings * 12)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Setup Investment</span>
                      <span className="font-medium">{formatCurrency(5000)}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Payback Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Clock className="h-5 w-5" />
                    Payback Period
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="text-center">
                    <div className="text-4xl font-bold text-orange-600">
                      {analysis.payback_period_months.toFixed(1)}
                    </div>
                    <div className="text-sm text-muted-foreground">Months to Break Even</div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span>Investment</span>
                      <span className="font-medium">{formatCurrency(5000)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Monthly Savings</span>
                      <span className="font-medium">{formatCurrency(analysis.net_savings)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Payback Period</span>
                      <span className="font-medium">{analysis.payback_period_months.toFixed(1)} months</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Environmental Impact */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Leaf className="h-5 w-5" />
                  Environmental Impact
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-emerald-600">
                      {formatNumber(analysis.carbon_savings_kg)}
                    </div>
                    <div className="text-sm text-muted-foreground">kg CO2 Saved</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-emerald-600">
                      {formatCurrency(analysis.carbon_savings_value)}
                    </div>
                    <div className="text-sm text-muted-foreground">Carbon Tax Savings</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-emerald-600">
                      {(analysis.carbon_savings_kg / 1000).toFixed(1)}
                    </div>
                    <div className="text-sm text-muted-foreground">Tons CO2 Equivalent</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="recommendations" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {analysis.recommendations.map((recommendation, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 rounded-lg border">
                      {getRecommendationIcon(recommendation)}
                      <span className="text-sm">{recommendation}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Action Items */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Next Steps</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm">Contact potential partner to discuss terms</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm">Negotiate transportation and processing costs</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm">Set up legal agreements and insurance</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm">Plan logistics and delivery schedule</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm">Monitor performance and adjust as needed</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default FinancialAnalysisPanel; 