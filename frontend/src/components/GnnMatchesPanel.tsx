import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  Users, 
  DollarSign, 
  Leaf, 
  Target, 
  ArrowRight,
  CheckCircle,
  XCircle,
  Clock,
  Star,
  TrendingUp,
  TrendingDown,
  MapPin,
  Building2,
  Factory,
  Truck,
  Recycle,
  Sparkles,
  Brain,
  Zap,
  Activity,
  BarChart3,
  Eye,
  MessageSquare,
  Phone,
  Mail,
  ExternalLink,
  Filter,
  Search,
  SortAsc,
  SortDesc,
  RefreshCw,
  Settings,
  MoreHorizontal,
  Calendar,
  Clock as ClockIcon,
  Award,
  Target as TargetIcon,
  Lightbulb,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  Percent,
  Thermometer,
  Droplets,
  Gauge,
  Scale,
  Calculator,
  FileText,
  PieChart,
  LineChart,
  AreaChart,
  ScatterChart,
  BarChart,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Calendar as CalendarIcon,
  MapPin as MapPinIcon,
  Building2 as Building2Icon,
  Factory as FactoryIcon,
  Truck as TruckIcon,
  Recycle as RecycleIcon,
  Leaf as LeafIcon2,
  DollarSign as DollarSignIcon,
  Percent as PercentIcon,
  Thermometer as ThermometerIcon,
  Droplets as DropletsIcon,
  Gauge as GaugeIcon,
  Scale as ScaleIcon,
  Calculator as CalculatorIcon,
  FileText as FileTextIcon,
  PieChart as PieChartIcon,
  LineChart as LineChartIcon,
  AreaChart as AreaChartIcon,
  ScatterChart as ScatterChartIcon,
  BarChart as BarChartIcon
} from 'lucide-react';
import { supabase } from '../lib/supabase';

interface Match {
  id: string;
  company_id: string;
  partner_company_id: string;
  match_score: number;
  potential_savings: number;
  carbon_reduction: number;
  materials_involved: string[];
  status: 'pending' | 'accepted' | 'rejected' | 'in_progress';
  created_at: string;
  partner_company?: {
    name: string;
    industry: string;
    location: string;
  };
}

interface GnnMatchesPanelProps {
  companyId?: string;
}

export const GnnMatchesPanel: React.FC<GnnMatchesPanelProps> = ({ companyId }) => {
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'pending' | 'accepted' | 'rejected'>('all');
  const [sortBy, setSortBy] = useState<'score' | 'savings' | 'date'>('score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [selectedMatch, setSelectedMatch] = useState<Match | null>(null);

  useEffect(() => {
    if (companyId) {
      loadMatches();
    }
  }, [companyId, filter, sortBy, sortOrder]);

  const loadMatches = async () => {
    try {
      setLoading(true);
      
      let query = supabase
        .from('matches')
        .select(`
          *,
          partner_company:companies!matches_partner_company_id_fkey(
            name,
            industry,
            location
          )
        `)
        .eq('company_id', companyId);

      if (filter !== 'all') {
        query = query.eq('status', filter);
      }

      const { data, error } = await query;

      if (error) {
        console.error('Error loading matches:', error);
        return;
      }

      // Sort matches
      const sortedMatches = data?.sort((a, b) => {
        let aValue: number;
        let bValue: number;

        switch (sortBy) {
          case 'score':
            aValue = a.match_score || 0;
            bValue = b.match_score || 0;
            break;
          case 'savings':
            aValue = a.potential_savings || 0;
            bValue = b.potential_savings || 0;
            break;
          case 'date':
            aValue = new Date(a.created_at).getTime();
            bValue = new Date(b.created_at).getTime();
            break;
          default:
            aValue = a.match_score || 0;
            bValue = b.match_score || 0;
        }

        return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
      }) || [];

      setMatches(sortedMatches);
    } catch (error) {
      console.error('Error loading matches:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateMatchStatus = async (matchId: string, status: 'accepted' | 'rejected') => {
    try {
      const { error } = await supabase
        .from('matches')
        .update({ 
          status,
          [status === 'accepted' ? 'accepted_at' : 'rejected_at']: new Date().toISOString()
        })
        .eq('id', matchId);

      if (error) {
        console.error('Error updating match status:', error);
        return;
      }

      // Update local state
      setMatches(prev => prev.map(match => 
        match.id === matchId 
          ? { ...match, status, [status === 'accepted' ? 'accepted_at' : 'rejected_at']: new Date().toISOString() }
          : match
      ));

      // Create notification
      await supabase
        .from('notifications')
        .insert({
          company_id: companyId,
          type: 'match_update',
          title: `Match ${status}`,
          message: `Your match with ${selectedMatch?.partner_company?.name || 'partner'} has been ${status}`,
          data: { match_id: matchId, status }
        });

    } catch (error) {
      console.error('Error updating match status:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'accepted': return 'bg-green-500';
      case 'rejected': return 'bg-red-500';
      case 'in_progress': return 'bg-blue-500';
      default: return 'bg-yellow-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'accepted': return 'Accepted';
      case 'rejected': return 'Rejected';
      case 'in_progress': return 'In Progress';
      default: return 'Pending';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="w-8 h-8 border-4 border-emerald-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-gray-400">Loading matches...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Brain className="w-6 h-6 text-emerald-400" />
            AI-Generated Matches
          </h2>
          <p className="text-gray-400 mt-1">
            Discovered through our advanced Graph Neural Network
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={loadMatches}
            className="border-slate-600 text-gray-300 hover:bg-slate-700"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Filters and Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Total Matches</p>
                <p className="text-2xl font-bold text-white">{matches.length}</p>
              </div>
              <Users className="w-8 h-8 text-emerald-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Pending</p>
                <p className="text-2xl font-bold text-yellow-400">
                  {matches.filter(m => m.status === 'pending').length}
                </p>
              </div>
              <Clock className="w-8 h-8 text-yellow-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Accepted</p>
                <p className="text-2xl font-bold text-green-400">
                  {matches.filter(m => m.status === 'accepted').length}
                </p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Potential Savings</p>
                <p className="text-2xl font-bold text-emerald-400">
                  ${matches.reduce((sum, m) => sum + (m.potential_savings || 0), 0).toLocaleString()}
                </p>
              </div>
              <DollarSign className="w-8 h-8 text-emerald-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-400" />
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="bg-slate-800 border border-slate-600 text-white rounded-md px-3 py-1 text-sm"
          >
            <option value="all">All Matches</option>
            <option value="pending">Pending</option>
            <option value="accepted">Accepted</option>
            <option value="rejected">Rejected</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <SortAsc className="w-4 h-4 text-gray-400" />
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-slate-800 border border-slate-600 text-white rounded-md px-3 py-1 text-sm"
          >
            <option value="score">Match Score</option>
            <option value="savings">Potential Savings</option>
            <option value="date">Date</option>
          </select>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}
            className="text-gray-400 hover:text-white"
          >
            {sortOrder === 'asc' ? <SortAsc className="w-4 h-4" /> : <SortDesc className="w-4 h-4" />}
          </Button>
        </div>
      </div>

      {/* Matches List */}
      <div className="space-y-4">
        {matches.length === 0 ? (
          <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-700">
            <CardContent className="p-8 text-center">
              <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-white mb-2">No Matches Found</h3>
              <p className="text-gray-400">
                Our AI is analyzing your profile and will generate matches soon.
              </p>
            </CardContent>
          </Card>
        ) : (
          matches.map((match) => (
            <Card 
              key={match.id} 
              className="bg-slate-800/50 backdrop-blur-sm border-slate-700 hover:border-slate-600 transition-colors cursor-pointer"
              onClick={() => setSelectedMatch(match)}
            >
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-3">
                      <div className="flex items-center gap-2">
                        <Building2 className="w-5 h-5 text-emerald-400" />
                        <h3 className="font-semibold text-white">
                          {match.partner_company?.name || 'Unknown Company'}
                        </h3>
                      </div>
                      
                      <Badge 
                        className={`${getStatusColor(match.status)} text-white`}
                      >
                        {getStatusText(match.status)}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      <div className="flex items-center gap-2">
                        <Target className="w-4 h-4 text-emerald-400" />
                        <span className="text-sm text-gray-400">Match Score:</span>
                        <span className={`font-semibold ${getScoreColor(match.match_score || 0)}`}>
                          {match.match_score || 0}%
                        </span>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <DollarSign className="w-4 h-4 text-green-400" />
                        <span className="text-sm text-gray-400">Potential Savings:</span>
                        <span className="font-semibold text-green-400">
                          ${(match.potential_savings || 0).toLocaleString()}
                        </span>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        <Leaf className="w-4 h-4 text-emerald-400" />
                        <span className="text-sm text-gray-400">Carbon Reduction:</span>
                        <span className="font-semibold text-emerald-400">
                          {(match.carbon_reduction || 0).toFixed(1)} tons CO₂
                        </span>
                      </div>
                    </div>

                    {match.materials_involved && match.materials_involved.length > 0 && (
                      <div className="mb-4">
                        <p className="text-sm text-gray-400 mb-2">Materials Involved:</p>
                        <div className="flex flex-wrap gap-2">
                          {match.materials_involved.map((material, index) => (
                            <Badge key={index} variant="secondary" className="bg-slate-700 text-gray-300">
                              {material}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex items-center gap-2 text-sm text-gray-400">
                      <MapPin className="w-4 h-4" />
                      <span>{match.partner_company?.location || 'Unknown location'}</span>
                      <span>•</span>
                      <span>{match.partner_company?.industry || 'Unknown industry'}</span>
                      <span>•</span>
                      <Clock className="w-4 h-4" />
                      <span>{new Date(match.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>

                  {match.status === 'pending' && (
                    <div className="flex items-center gap-2 ml-4">
                      <Button
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          updateMatchStatus(match.id, 'accepted');
                        }}
                        className="bg-green-600 hover:bg-green-700"
                      >
                        <CheckCircle className="w-4 h-4 mr-1" />
                        Accept
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e) => {
                          e.stopPropagation();
                          updateMatchStatus(match.id, 'rejected');
                        }}
                        className="border-red-600 text-red-400 hover:bg-red-600 hover:text-white"
                      >
                        <XCircle className="w-4 h-4 mr-1" />
                        Reject
                      </Button>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {/* Match Details Modal */}
      {selectedMatch && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold text-white">Match Details</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedMatch(null)}
                className="text-gray-400 hover:text-white"
              >
                <XCircle className="w-5 h-5" />
              </Button>
            </div>

            <div className="space-y-6">
              <div>
                <h4 className="font-semibold text-white mb-2">Partner Company</h4>
                <div className="bg-slate-700 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <Building2 className="w-6 h-6 text-emerald-400" />
                    <div>
                      <p className="font-semibold text-white">
                        {selectedMatch.partner_company?.name || 'Unknown Company'}
                      </p>
                      <p className="text-sm text-gray-400">
                        {selectedMatch.partner_company?.industry || 'Unknown industry'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <MapPin className="w-4 h-4" />
                    <span>{selectedMatch.partner_company?.location || 'Unknown location'}</span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-slate-700 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Target className="w-5 h-5 text-emerald-400" />
                    <span className="font-semibold text-white">Match Score</span>
                  </div>
                  <p className={`text-2xl font-bold ${getScoreColor(selectedMatch.match_score || 0)}`}>
                    {selectedMatch.match_score || 0}%
                  </p>
                </div>

                <div className="bg-slate-700 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <DollarSign className="w-5 h-5 text-green-400" />
                    <span className="font-semibold text-white">Potential Savings</span>
                  </div>
                  <p className="text-2xl font-bold text-green-400">
                    ${(selectedMatch.potential_savings || 0).toLocaleString()}
                  </p>
                </div>

                <div className="bg-slate-700 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Leaf className="w-5 h-5 text-emerald-400" />
                    <span className="font-semibold text-white">Carbon Reduction</span>
                  </div>
                  <p className="text-2xl font-bold text-emerald-400">
                    {(selectedMatch.carbon_reduction || 0).toFixed(1)} tons CO₂
                  </p>
                </div>
              </div>

              {selectedMatch.materials_involved && selectedMatch.materials_involved.length > 0 && (
                <div>
                  <h4 className="font-semibold text-white mb-2">Materials Involved</h4>
                  <div className="bg-slate-700 rounded-lg p-4">
                    <div className="flex flex-wrap gap-2">
                      {selectedMatch.materials_involved.map((material, index) => (
                        <Badge key={index} variant="secondary" className="bg-slate-600 text-gray-300">
                          {material}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              <div className="flex items-center gap-2">
                <Button
                  onClick={() => {
                    updateMatchStatus(selectedMatch.id, 'accepted');
                    setSelectedMatch(null);
                  }}
                  className="bg-green-600 hover:bg-green-700"
                >
                  <CheckCircle className="w-4 h-4 mr-2" />
                  Accept Match
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    updateMatchStatus(selectedMatch.id, 'rejected');
                    setSelectedMatch(null);
                  }}
                  className="border-red-600 text-red-400 hover:bg-red-600 hover:text-white"
                >
                  <XCircle className="w-4 h-4 mr-2" />
                  Reject Match
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setSelectedMatch(null)}
                  className="border-slate-600 text-gray-400 hover:bg-slate-700"
                >
                  Close
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 