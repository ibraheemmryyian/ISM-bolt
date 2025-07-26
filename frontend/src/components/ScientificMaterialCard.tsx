import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { supabase } from '../lib/supabase';
import { 
  Beaker, 
  Leaf, 
  Recycle, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  Info,
  ExternalLink,
  Loader2
} from 'lucide-react';

interface ScientificMaterialData {
  id: string;
  name: string;
  category: string;
  scientific_properties: {
    density?: number;
    melting_point?: number;
    tensile_strength?: number;
    [key: string]: any;
  };
  sustainability_metrics: {
    recyclability_score: number;
    carbon_footprint: number;
    renewable_content: number;
    biodegradability_score: number;
    toxicity_level: string;
    energy_intensity: number;
    water_intensity: number;
  };
  environmental_impact: {
    carbon_savings: number;
    energy_savings: number;
    water_savings: number;
    waste_reduction: number;
  };
  circular_opportunities: Array<{
    type: string;
    description: string;
    potential_impact: number;
    implementation_steps: string[];
  }>;
  alternatives?: Array<{
    id: string;
    name: string;
    sustainability_improvement: number;
    cost_impact: string;
  }>;
}

interface ScientificMaterialCardProps {
  materialId: string;
  materialName: string;
  onAlternativeSelect?: (alternative: any) => void;
}

const ScientificMaterialCard: React.FC<ScientificMaterialCardProps> = ({
  materialId,
  materialName,
  onAlternativeSelect
}) => {
  const [materialData, setMaterialData] = useState<ScientificMaterialData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAlternatives, setShowAlternatives] = useState(false);

  useEffect(() => {
    loadMaterialData();
  }, [materialId]);

  const loadMaterialData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load material data from database first
      const { data: dbMaterial, error: dbError } = await supabase
        .from('materials')
        .select('*')
        .eq('id', materialId)
        .maybeSingle();

      if (dbError) throw dbError;

      // If we have scientific data, use it
      if (dbMaterial.scientific_properties) {
        setMaterialData({
          id: dbMaterial.id,
          name: dbMaterial.name,
          category: dbMaterial.category,
          scientific_properties: dbMaterial.scientific_properties,
          sustainability_metrics: dbMaterial.sustainability_metrics,
          environmental_impact: dbMaterial.environmental_impact,
          circular_opportunities: dbMaterial.circular_opportunities || []
        });
      } else {
        // Fetch from Next-Gen Materials API
        const response = await fetch(`/api/materials/${materialName}/data`);
        const data = await response.json();

        if (data.success) {
          setMaterialData(data.material);
          
          // Save to database for future use
          await supabase
            .from('materials')
            .update({
              scientific_properties: data.material.scientific_properties,
              sustainability_metrics: data.material.sustainability_metrics,
              environmental_impact: data.material.environmental_impact,
              circular_opportunities: data.material.circular_opportunities
            })
            .eq('id', materialId);
        } else {
          throw new Error(data.error || 'Failed to load material data');
        }
      }
    } catch (error) {
      console.error('Error loading material data:', error);
      setError(error instanceof Error ? error.message : 'Failed to load material data');
    } finally {
      setLoading(false);
    }
  };

  const loadAlternatives = async () => {
    try {
      const response = await fetch(`/api/materials/${materialId}/alternatives`);
      const data = await response.json();

      if (data.success) {
        setMaterialData(prev => prev ? { ...prev, alternatives: data.alternatives } : null);
      }
    } catch (error) {
      console.error('Error loading alternatives:', error);
    }
  };

  const getSustainabilityColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getToxicityColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'high': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  if (loading) {
    return (
      <Card className="w-full">
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
          <span className="ml-2">Loading scientific data...</span>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full">
        <CardContent className="flex items-center justify-center py-8">
          <AlertTriangle className="h-6 w-6 text-red-600" />
          <span className="ml-2 text-red-600">{error}</span>
        </CardContent>
      </Card>
    );
  }

  if (!materialData) {
    return (
      <Card className="w-full">
        <CardContent className="flex items-center justify-center py-8">
          <Info className="h-6 w-6 text-gray-600" />
          <span className="ml-2 text-gray-600">No scientific data available</span>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Beaker className="h-5 w-5 text-blue-600" />
          Scientific Material Analysis
        </CardTitle>
        <p className="text-sm text-gray-600">
          {materialData.name} - {materialData.category}
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Scientific Properties */}
        <div>
          <h3 className="font-medium text-gray-900 mb-3">Physical Properties</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {materialData.scientific_properties.density && (
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-lg font-semibold">{materialData.scientific_properties.density}</div>
                <div className="text-sm text-gray-600">Density (g/cm³)</div>
              </div>
            )}
            {materialData.scientific_properties.melting_point && (
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-lg font-semibold">{materialData.scientific_properties.melting_point}°C</div>
                <div className="text-sm text-gray-600">Melting Point</div>
              </div>
            )}
            {materialData.scientific_properties.tensile_strength && (
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-lg font-semibold">{materialData.scientific_properties.tensile_strength} MPa</div>
                <div className="text-sm text-gray-600">Tensile Strength</div>
              </div>
            )}
          </div>
        </div>

        {/* Sustainability Metrics */}
        <div>
          <h3 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
            <Leaf className="h-4 w-4 text-green-600" />
            Sustainability Profile
          </h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Recyclability</span>
                <span>{materialData.sustainability_metrics.recyclability_score}%</span>
              </div>
              <Progress value={materialData.sustainability_metrics.recyclability_score} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Renewable Content</span>
                <span>{materialData.sustainability_metrics.renewable_content}%</span>
              </div>
              <Progress value={materialData.sustainability_metrics.renewable_content} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Biodegradability</span>
                <span>{materialData.sustainability_metrics.biodegradability_score}%</span>
              </div>
              <Progress value={materialData.sustainability_metrics.biodegradability_score} className="h-2" />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Carbon Footprint</span>
              <Badge className={getSustainabilityColor(100 - materialData.sustainability_metrics.carbon_footprint * 10)}>
                {materialData.sustainability_metrics.carbon_footprint} kg CO2e/kg
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Toxicity Level</span>
              <Badge className={getToxicityColor(materialData.sustainability_metrics.toxicity_level)}>
                {materialData.sustainability_metrics.toxicity_level}
              </Badge>
            </div>
          </div>
        </div>

        {/* Environmental Impact */}
        {materialData.environmental_impact && (
          <div>
            <h3 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-blue-600" />
              Environmental Impact
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-blue-50 rounded-lg">
                <div className="text-lg font-semibold text-blue-600">
                  {materialData.environmental_impact.carbon_savings} kg
                </div>
                <div className="text-sm text-gray-600">CO2 Savings</div>
              </div>
              <div className="text-center p-3 bg-green-50 rounded-lg">
                <div className="text-lg font-semibold text-green-600">
                  {materialData.environmental_impact.energy_savings} kWh
                </div>
                <div className="text-sm text-gray-600">Energy Savings</div>
              </div>
            </div>
          </div>
        )}

        {/* Circular Economy Opportunities */}
        {materialData.circular_opportunities && materialData.circular_opportunities.length > 0 && (
          <div>
            <h3 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
              <Recycle className="h-4 w-4 text-purple-600" />
              Circular Economy Opportunities
            </h3>
            <div className="space-y-3">
              {materialData.circular_opportunities.map((opportunity, index) => (
                <div key={index} className="p-3 border border-purple-200 rounded-lg bg-purple-50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-purple-900">{opportunity.type}</span>
                    <Badge className="bg-purple-100 text-purple-700">
                      {opportunity.potential_impact}% impact
                    </Badge>
                  </div>
                  <p className="text-sm text-purple-800 mb-2">{opportunity.description}</p>
                  {opportunity.implementation_steps && (
                    <div className="text-xs text-purple-700">
                      <strong>Steps:</strong> {opportunity.implementation_steps.join(', ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Sustainable Alternatives */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-gray-900 flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-600" />
              Sustainable Alternatives
            </h3>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (!materialData.alternatives) {
                  loadAlternatives();
                }
                setShowAlternatives(!showAlternatives);
              }}
            >
              {showAlternatives ? 'Hide' : 'Show'} Alternatives
            </Button>
          </div>
          
          {showAlternatives && materialData.alternatives && (
            <div className="space-y-3">
              {materialData.alternatives.map((alternative) => (
                <div key={alternative.id} className="p-3 border border-green-200 rounded-lg bg-green-50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-green-900">{alternative.name}</span>
                    <Badge className="bg-green-100 text-green-700">
                      +{alternative.sustainability_improvement}% improvement
                    </Badge>
                  </div>
                  <p className="text-sm text-green-800 mb-2">
                    Cost impact: {alternative.cost_impact}
                  </p>
                  <Button
                    size="sm"
                    onClick={() => onAlternativeSelect?.(alternative)}
                    className="w-full"
                  >
                    Select Alternative
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default ScientificMaterialCard; 