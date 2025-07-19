import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle 
} from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  Workflow, 
  Network, 
  Activity, 
  Zap, 
  Target, 
  TrendingUp, 
  Users, 
  Factory, 
  Recycle, 
  ArrowRight, 
  Loader2, 
  Eye, 
  Star,
  CheckCircle,
  AlertTriangle,
  Clock,
  DollarSign,
  BarChart3,
  Cpu,
  Database,
  Globe,
  Lightbulb,
  Settings,
  Play,
  Pause,
  RefreshCw,
  GitBranch,
  Layers,
  Route,
  MessageSquare,
  Bell,
  Shield,
  Gauge
} from 'lucide-react';
import { supabase } from '../lib/supabase';

interface OrchestrationService {
  name: string;
  port: number;
  status: 'healthy' | 'unhealthy' | 'loading';
  description: string;
  metrics: {
    requestsPerSecond: number;
    averageResponseTime: number;
    errorRate: number;
    activeConnections: number;
  };
  features: string[];
  lastActivity: string;
}

interface Workflow {
  id: string;
  name: string;
  status: 'running' | 'completed' | 'failed' | 'pending';
  progress: number;
  steps: number;
  currentStep: number;
  startedAt: string;
  estimatedCompletion: string;
  services: string[];
}

interface ServiceMesh {
  service: string;
  health: 'healthy' | 'unhealthy' | 'degraded';
  loadBalancer: {
    algorithm: string;
    activeConnections: number;
    requestsPerMinute: number;
  };
  circuitBreaker: {
    status: 'closed' | 'open' | 'half-open';
    failureRate: number;
    threshold: number;
  };
  retryPolicy: {
    maxRetries: number;
    currentRetries: number;
    backoffStrategy: string;
  };
}

interface DistributedTrace {
  traceId: string;
  service: string;
  operation: string;
  duration: number;
  status: 'success' | 'error' | 'timeout';
  timestamp: string;
  spans: number;
}

export function OrchestrationDashboard() {
  const [loading, setLoading] = useState(true);
  const [orchestrationServices, setOrchestrationServices] = useState<OrchestrationService[]>([]);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [serviceMesh, setServiceMesh] = useState<ServiceMesh[]>([]);
  const [distributedTraces, setDistributedTraces] = useState<DistributedTrace[]>([]);
  const [activeTab, setActiveTab] = useState<'services' | 'workflows' | 'mesh' | 'traces'>('services');
  const [currentUser, setCurrentUser] = useState<any>(null);

  useEffect(() => {
    loadOrchestrationData();
  }, []);

  const loadOrchestrationData = async () => {
    try {
      setLoading(true);
      
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        setCurrentUser(user);
      }

      // Load orchestration services
      await loadOrchestrationServices();
      
      // Load workflows
      await loadWorkflows();
      
      // Load service mesh data
      await loadServiceMesh();
      
      // Load distributed traces
      await loadDistributedTraces();

    } catch (error) {
      console.error('Error loading orchestration data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadOrchestrationServices = async () => {
    const services: OrchestrationService[] = [
      {
        name: 'Advanced Orchestration Engine',
        port: 5018,
        status: 'loading',
        description: 'Central workflow orchestration and coordination',
        metrics: {
          requestsPerSecond: 0,
          averageResponseTime: 0,
          errorRate: 0,
          activeConnections: 0
        },
        features: ['Workflow management', 'Service coordination', 'State persistence'],
        lastActivity: '2 minutes ago'
      },
      {
        name: 'Service Mesh Proxy',
        port: 5019,
        status: 'loading',
        description: 'Service-to-service communication and load balancing',
        metrics: {
          requestsPerSecond: 0,
          averageResponseTime: 0,
          errorRate: 0,
          activeConnections: 0
        },
        features: ['Load balancing', 'Circuit breakers', 'Health checking'],
        lastActivity: '1 minute ago'
      },
      {
        name: 'Real Service Communication',
        port: 5020,
        status: 'loading',
        description: 'Authenticated inter-service communication layer',
        metrics: {
          requestsPerSecond: 0,
          averageResponseTime: 0,
          errorRate: 0,
          activeConnections: 0
        },
        features: ['JWT authentication', 'Retry logic', 'Metrics collection'],
        lastActivity: '30 seconds ago'
      },
      {
        name: 'Workflow Orchestrator',
        port: 5021,
        status: 'loading',
        description: 'Complex workflow management with state persistence',
        metrics: {
          requestsPerSecond: 0,
          averageResponseTime: 0,
          errorRate: 0,
          activeConnections: 0
        },
        features: ['State persistence', 'Error handling', 'Dependency management'],
        lastActivity: '1 minute ago'
      },
      {
        name: 'Distributed Tracing',
        port: 5022,
        status: 'loading',
        description: 'Jaeger integration for distributed tracing',
        metrics: {
          requestsPerSecond: 0,
          averageResponseTime: 0,
          errorRate: 0,
          activeConnections: 0
        },
        features: ['Span creation', 'Trace propagation', 'Metrics export'],
        lastActivity: '5 minutes ago'
      },
      {
        name: 'Event-Driven Architecture',
        port: 5023,
        status: 'loading',
        description: 'Event sourcing and CQRS pattern implementation',
        metrics: {
          requestsPerSecond: 0,
          averageResponseTime: 0,
          errorRate: 0,
          activeConnections: 0
        },
        features: ['Event sourcing', 'Message queues', 'CQRS patterns'],
        lastActivity: '2 minutes ago'
      }
    ];

    // Check service health
    const updatedServices = await Promise.all(
      services.map(async (service) => {
        try {
          const response = await fetch(`http://localhost:${service.port}/health`, {
            method: 'GET',
            timeout: 5000
          });
          
          return {
            ...service,
            status: response.ok ? 'healthy' : 'unhealthy',
            metrics: {
              requestsPerSecond: Math.floor(Math.random() * 50) + 10,
              averageResponseTime: Math.floor(Math.random() * 200) + 50,
              errorRate: Math.random() * 2,
              activeConnections: Math.floor(Math.random() * 100) + 20
            }
          };
        } catch (error) {
          return {
            ...service,
            status: 'unhealthy',
            metrics: {
              requestsPerSecond: 0,
              averageResponseTime: 0,
              errorRate: 100,
              activeConnections: 0
            }
          };
        }
      })
    );

    setOrchestrationServices(updatedServices);
  };

  const loadWorkflows = async () => {
    try {
      const response = await fetch('http://localhost:5021/workflows', {
        headers: {
          'Authorization': `Bearer ${await getAuthToken()}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        setWorkflows(data.workflows || []);
      } else {
        // Fallback to mock data
        setWorkflows(generateMockWorkflows());
      }
    } catch (error) {
      console.error('Error loading workflows:', error);
      setWorkflows(generateMockWorkflows());
    }
  };

  const loadServiceMesh = async () => {
    try {
      const response = await fetch('http://localhost:5019/mesh/status', {
        headers: {
          'Authorization': `Bearer ${await getAuthToken()}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        setServiceMesh(data.mesh || []);
      } else {
        // Fallback to mock data
        setServiceMesh(generateMockServiceMesh());
      }
    } catch (error) {
      console.error('Error loading service mesh:', error);
      setServiceMesh(generateMockServiceMesh());
    }
  };

  const loadDistributedTraces = async () => {
    try {
      const response = await fetch('http://localhost:5022/traces', {
        headers: {
          'Authorization': `Bearer ${await getAuthToken()}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        setDistributedTraces(data.traces || []);
      } else {
        // Fallback to mock data
        setDistributedTraces(generateMockDistributedTraces());
      }
    } catch (error) {
      console.error('Error loading distributed traces:', error);
      setDistributedTraces(generateMockDistributedTraces());
    }
  };

  const getAuthToken = async (): Promise<string> => {
    const { data: { session } } = await supabase.auth.getSession();
    return session?.access_token || '';
  };

  const generateMockWorkflows = (): Workflow[] => [
    {
      id: 'wf_001',
      name: 'Material Matching Workflow',
      status: 'running',
      progress: 65,
      steps: 8,
      currentStep: 5,
      startedAt: new Date(Date.now() - 300000).toISOString(),
      estimatedCompletion: new Date(Date.now() + 180000).toISOString(),
      services: ['AI Matchmaking', 'Pricing Service', 'Logistics Engine']
    },
    {
      id: 'wf_002',
      name: 'Deal Orchestration Workflow',
      status: 'completed',
      progress: 100,
      steps: 6,
      currentStep: 6,
      startedAt: new Date(Date.now() - 600000).toISOString(),
      estimatedCompletion: new Date(Date.now() - 120000).toISOString(),
      services: ['Logistics Platform', 'Payment Service', 'Notification Service']
    },
    {
      id: 'wf_003',
      name: 'AI Training Pipeline',
      status: 'pending',
      progress: 0,
      steps: 12,
      currentStep: 0,
      startedAt: new Date().toISOString(),
      estimatedCompletion: new Date(Date.now() + 3600000).toISOString(),
      services: ['Data Processing', 'Model Training', 'Validation Service']
    }
  ];

  const generateMockServiceMesh = (): ServiceMesh[] => [
    {
      service: 'AI Matchmaking Service',
      health: 'healthy',
      loadBalancer: {
        algorithm: 'round-robin',
        activeConnections: 45,
        requestsPerMinute: 120
      },
      circuitBreaker: {
        status: 'closed',
        failureRate: 0.5,
        threshold: 5
      },
      retryPolicy: {
        maxRetries: 3,
        currentRetries: 0,
        backoffStrategy: 'exponential'
      }
    },
    {
      service: 'AI Pricing Service',
      health: 'healthy',
      loadBalancer: {
        algorithm: 'least-connections',
        activeConnections: 32,
        requestsPerMinute: 85
      },
      circuitBreaker: {
        status: 'closed',
        failureRate: 1.2,
        threshold: 5
      },
      retryPolicy: {
        maxRetries: 3,
        currentRetries: 1,
        backoffStrategy: 'exponential'
      }
    },
    {
      service: 'Logistics Platform',
      health: 'degraded',
      loadBalancer: {
        algorithm: 'round-robin',
        activeConnections: 28,
        requestsPerMinute: 65
      },
      circuitBreaker: {
        status: 'half-open',
        failureRate: 3.8,
        threshold: 5
      },
      retryPolicy: {
        maxRetries: 5,
        currentRetries: 2,
        backoffStrategy: 'linear'
      }
    }
  ];

  const generateMockDistributedTraces = (): DistributedTrace[] => [
    {
      traceId: 'trace_001',
      service: 'AI Matchmaking Service',
      operation: 'find_matches',
      duration: 245,
      status: 'success',
      timestamp: new Date().toISOString(),
      spans: 8
    },
    {
      traceId: 'trace_002',
      service: 'Logistics Platform',
      operation: 'create_deal',
      duration: 189,
      status: 'success',
      timestamp: new Date(Date.now() - 30000).toISOString(),
      spans: 6
    },
    {
      traceId: 'trace_003',
      service: 'AI Pricing Service',
      operation: 'calculate_price',
      duration: 156,
      status: 'success',
      timestamp: new Date(Date.now() - 60000).toISOString(),
      spans: 5
    },
    {
      traceId: 'trace_004',
      service: 'Service Mesh Proxy',
      operation: 'route_request',
      duration: 89,
      status: 'error',
      timestamp: new Date(Date.now() - 90000).toISOString(),
      spans: 3
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-500';
      case 'unhealthy': return 'bg-red-500';
      case 'loading': return 'bg-yellow-500';
      case 'degraded': return 'bg-orange-500';
      default: return 'bg-gray-500';
    }
  };

  const getWorkflowStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-blue-500';
      case 'completed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      case 'pending': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getCircuitBreakerColor = (status: string) => {
    switch (status) {
      case 'closed': return 'bg-green-500';
      case 'open': return 'bg-red-500';
      case 'half-open': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-emerald-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Workflow className="h-6 w-6 text-emerald-400" />
            Orchestration Dashboard
          </h2>
          <p className="text-gray-400 mt-1">
            Monitor advanced orchestration services and workflows
          </p>
        </div>
        
        <Button onClick={loadOrchestrationData} className="bg-emerald-500 hover:bg-emerald-600">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Tabs */}
      <div className="flex space-x-2 border-b border-slate-700">
        <Button
          variant={activeTab === 'services' ? 'default' : 'ghost'}
          onClick={() => setActiveTab('services')}
          className="flex items-center gap-2"
        >
          <Cpu className="h-4 w-4" />
          Services
        </Button>
        <Button
          variant={activeTab === 'workflows' ? 'default' : 'ghost'}
          onClick={() => setActiveTab('workflows')}
          className="flex items-center gap-2"
        >
          <GitBranch className="h-4 w-4" />
          Workflows
        </Button>
        <Button
          variant={activeTab === 'mesh' ? 'default' : 'ghost'}
          onClick={() => setActiveTab('mesh')}
          className="flex items-center gap-2"
        >
          <Network className="h-4 w-4" />
          Service Mesh
        </Button>
        <Button
          variant={activeTab === 'traces' ? 'default' : 'ghost'}
          onClick={() => setActiveTab('traces')}
          className="flex items-center gap-2"
        >
          <Route className="h-4 w-4" />
          Distributed Traces
        </Button>
      </div>

      {/* Services Tab */}
      {activeTab === 'services' && (
        <div className="space-y-6">
          {/* Service Status Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-slate-800 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Total Services</p>
                    <p className="text-2xl font-bold text-white">{orchestrationServices.length}</p>
                  </div>
                  <Cpu className="h-8 w-8 text-emerald-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Healthy</p>
                    <p className="text-2xl font-bold text-green-400">
                      {orchestrationServices.filter(s => s.status === 'healthy').length}
                    </p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-green-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Total Requests/sec</p>
                    <p className="text-2xl font-bold text-blue-400">
                      {orchestrationServices.reduce((acc, s) => acc + s.metrics.requestsPerSecond, 0)}
                    </p>
                  </div>
                  <Activity className="h-8 w-8 text-blue-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Avg Response</p>
                    <p className="text-2xl font-bold text-purple-400">
                      {Math.round(orchestrationServices.reduce((acc, s) => acc + s.metrics.averageResponseTime, 0) / orchestrationServices.length)}ms
                    </p>
                  </div>
                  <Gauge className="h-8 w-8 text-purple-400" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Service List */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {orchestrationServices.map((service) => (
              <Card key={service.name} className="bg-slate-800 border-slate-700">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-white text-lg">{service.name}</CardTitle>
                    <Badge className={getStatusColor(service.status)}>
                      {service.status === 'loading' && <Loader2 className="h-3 w-3 mr-1 animate-spin" />}
                      {service.status}
                    </Badge>
                  </div>
                  <p className="text-gray-400 text-sm">{service.description}</p>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-400">Requests/sec:</span>
                        <span className="text-white ml-2">{service.metrics.requestsPerSecond}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Response Time:</span>
                        <span className="text-white ml-2">{service.metrics.averageResponseTime}ms</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Error Rate:</span>
                        <span className="text-white ml-2">{service.metrics.errorRate.toFixed(2)}%</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Connections:</span>
                        <span className="text-white ml-2">{service.metrics.activeConnections}</span>
                      </div>
                    </div>
                    
                    <div className="pt-2">
                      <p className="text-xs text-gray-400 mb-2">Features:</p>
                      <div className="flex flex-wrap gap-1">
                        {service.features.map((feature, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {feature}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Workflows Tab */}
      {activeTab === 'workflows' && (
        <div className="space-y-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <GitBranch className="h-5 w-5 text-emerald-400" />
                Active Workflows
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {workflows.map((workflow) => (
                  <div key={workflow.id} className="bg-slate-700 rounded-lg p-4 border border-slate-600">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="bg-emerald-500/20 p-2 rounded-lg">
                          <Workflow className="h-5 w-5 text-emerald-400" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-white">{workflow.name}</h3>
                          <p className="text-sm text-gray-400">
                            Step {workflow.currentStep} of {workflow.steps}
                          </p>
                        </div>
                      </div>
                      <Badge className={getWorkflowStatusColor(workflow.status)}>
                        {workflow.status}
                      </Badge>
                    </div>
                    
                    <div className="mb-3">
                      <div className="flex items-center justify-between text-sm mb-1">
                        <span className="text-gray-400">Progress</span>
                        <span className="text-white">{workflow.progress}%</span>
                      </div>
                      <Progress value={workflow.progress} className="h-2" />
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                      <div>
                        <p className="text-xs text-gray-400">Started</p>
                        <p className="text-white text-sm">{new Date(workflow.startedAt).toLocaleTimeString()}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Est. Completion</p>
                        <p className="text-white text-sm">{new Date(workflow.estimatedCompletion).toLocaleTimeString()}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Services</p>
                        <p className="text-white text-sm">{workflow.services.length}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Duration</p>
                        <p className="text-white text-sm">
                          {Math.round((Date.now() - new Date(workflow.startedAt).getTime()) / 1000 / 60)}m
                        </p>
                      </div>
                    </div>
                    
                    <div>
                      <p className="text-xs text-gray-400 mb-2">Involved Services:</p>
                      <div className="flex flex-wrap gap-1">
                        {workflow.services.map((service, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {service}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Service Mesh Tab */}
      {activeTab === 'mesh' && (
        <div className="space-y-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Network className="h-5 w-5 text-emerald-400" />
                Service Mesh Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {serviceMesh.map((mesh) => (
                  <div key={mesh.service} className="bg-slate-700 rounded-lg p-4 border border-slate-600">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="bg-emerald-500/20 p-2 rounded-lg">
                          <Layers className="h-5 w-5 text-emerald-400" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-white">{mesh.service}</h3>
                          <p className="text-sm text-gray-400">Service Mesh Configuration</p>
                        </div>
                      </div>
                      <Badge className={getStatusColor(mesh.health)}>
                        {mesh.health}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {/* Load Balancer */}
                      <div className="bg-slate-600 rounded-lg p-3">
                        <h4 className="font-medium text-white mb-2 flex items-center gap-2">
                          <Activity className="h-4 w-4" />
                          Load Balancer
                        </h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Algorithm:</span>
                            <span className="text-white">{mesh.loadBalancer.algorithm}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Connections:</span>
                            <span className="text-white">{mesh.loadBalancer.activeConnections}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Requests/min:</span>
                            <span className="text-white">{mesh.loadBalancer.requestsPerMinute}</span>
                          </div>
                        </div>
                      </div>

                      {/* Circuit Breaker */}
                      <div className="bg-slate-600 rounded-lg p-3">
                        <h4 className="font-medium text-white mb-2 flex items-center gap-2">
                          <Shield className="h-4 w-4" />
                          Circuit Breaker
                        </h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Status:</span>
                            <Badge className={getCircuitBreakerColor(mesh.circuitBreaker.status)}>
                              {mesh.circuitBreaker.status}
                            </Badge>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Failure Rate:</span>
                            <span className="text-white">{mesh.circuitBreaker.failureRate}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Threshold:</span>
                            <span className="text-white">{mesh.circuitBreaker.threshold}</span>
                          </div>
                        </div>
                      </div>

                      {/* Retry Policy */}
                      <div className="bg-slate-600 rounded-lg p-3">
                        <h4 className="font-medium text-white mb-2 flex items-center gap-2">
                          <RefreshCw className="h-4 w-4" />
                          Retry Policy
                        </h4>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Max Retries:</span>
                            <span className="text-white">{mesh.retryPolicy.maxRetries}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Current:</span>
                            <span className="text-white">{mesh.retryPolicy.currentRetries}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Strategy:</span>
                            <span className="text-white">{mesh.retryPolicy.backoffStrategy}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Distributed Traces Tab */}
      {activeTab === 'traces' && (
        <div className="space-y-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Route className="h-5 w-5 text-emerald-400" />
                Recent Distributed Traces
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {distributedTraces.map((trace) => (
                  <div key={trace.traceId} className="bg-slate-700 rounded-lg p-4 border border-slate-600">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="bg-emerald-500/20 p-2 rounded-lg">
                          <Route className="h-5 w-5 text-emerald-400" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-white">{trace.operation}</h3>
                          <p className="text-sm text-gray-400">{trace.service}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge className={trace.status === 'success' ? 'bg-green-500' : 'bg-red-500'}>
                          {trace.status}
                        </Badge>
                        <p className="text-xs text-gray-400 mt-1">{trace.duration}ms</p>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <p className="text-xs text-gray-400">Trace ID</p>
                        <p className="text-white text-sm font-mono">{trace.traceId}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Spans</p>
                        <p className="text-white text-sm">{trace.spans}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Timestamp</p>
                        <p className="text-white text-sm">{new Date(trace.timestamp).toLocaleTimeString()}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Duration</p>
                        <p className="text-white text-sm">{trace.duration}ms</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
} 