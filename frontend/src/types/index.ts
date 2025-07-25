// Core Types
export interface User {
  id: string;
  email: string;
  role: 'user' | 'admin' | 'moderator';
  created_at: string;
  updated_at: string;
  user_metadata?: Record<string, unknown>;
}

export interface Company {
  id: string;
  name: string;
  email: string;
  role: string;
  created_at: string;
  level: number;
  xp: number;
  industry?: string;
  location?: string;
  employee_count?: number;
  sustainability_score?: number;
  carbon_footprint?: number;
  water_usage?: number;
  subscription?: {
    tier: string;
    status?: string;
  };
  data_quality_score?: number;
  processing_status?: string;
  enriched_data?: Record<string, unknown>;
  business_metrics?: BusinessMetrics;
  ai_insights?: AIInsights;
  recommendations?: Recommendation[];
}

export interface BusinessMetrics {
  annual_revenue?: number;
  employee_count?: number;
  market_cap?: number;
  growth_rate?: number;
  profitability?: number;
  sustainability_score?: number;
  carbon_footprint?: number;
  water_usage?: number;
  waste_generation?: number;
  energy_consumption?: number;
}

export interface AIInsights {
  market_trends?: MarketTrend[];
  opportunities?: Opportunity[];
  risks?: Risk[];
  recommendations?: Recommendation[];
  symbiosis_potential?: number;
  sustainability_improvements?: number;
  logistics_optimizations?: number;
}

export interface MarketTrend {
  category: string;
  trend: 'increasing' | 'decreasing' | 'stable';
  confidence: number;
  description: string;
  impact_score: number;
}

export interface Opportunity {
  id: string;
  title: string;
  description: string;
  category: string;
  potential_value: number;
  confidence: number;
  timeframe: string;
  requirements: string[];
}

export interface Risk {
  id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  probability: number;
  mitigation_strategies: string[];
}

export interface Recommendation {
  id: string;
  title: string;
  description: string;
  category: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimated_impact: number;
  implementation_cost: number;
  timeframe: string;
  ai_reasoning: string;
}

export interface Material {
  id: string;
  company_id: string;
  material_name: string;
  quantity: number;
  unit: string;
  type: 'waste' | 'requirement';
  created_at: string;
  price_per_unit?: number;
  quality_grade?: string;
  ai_generated?: boolean;
  confidence_score?: number;
  logistics_cost?: number;
  carbon_footprint?: number;
  sustainability_score?: number;
  potential_value?: number;
  company?: {
    name: string;
    industry?: string;
    location?: string;
  };
  properties?: MaterialProperties;
  analysis?: MaterialAnalysis;
}

export interface MaterialProperties {
  chemical_composition?: string;
  physical_properties?: Record<string, unknown>;
  environmental_impact?: EnvironmentalImpact;
  safety_data?: SafetyData;
  regulatory_status?: RegulatoryStatus;
}

export interface EnvironmentalImpact {
  carbon_footprint: number;
  water_usage: number;
  energy_consumption: number;
  waste_generation: number;
  recyclability: number;
  biodegradability: number;
}

export interface SafetyData {
  toxicity_level: string;
  flammability: string;
  reactivity: string;
  health_hazards: string[];
  safety_precautions: string[];
}

export interface RegulatoryStatus {
  compliance_status: 'compliant' | 'non_compliant' | 'pending';
  regulations: string[];
  certifications: string[];
  restrictions: string[];
}

export interface MaterialAnalysis {
  ai_confidence: number;
  market_demand: number;
  price_trend: 'increasing' | 'decreasing' | 'stable';
  availability: 'abundant' | 'moderate' | 'scarce';
  substitution_options: string[];
  processing_requirements: string[];
}

export interface Match {
  id: string;
  waste_company_id: string;
  requirement_company_id: string;
  waste_material_id: string;
  requirement_material_id: string;
  match_score: number;
  symbiosis_potential: number;
  logistics_score: number;
  sustainability_impact: number;
  economic_value: number;
  created_at: string;
  status: 'pending' | 'active' | 'completed' | 'rejected';
  waste_company?: {
    name: string;
    industry?: string;
    location?: string;
  };
  requirement_company?: {
    name: string;
    industry?: string;
    location?: string;
  };
  waste_material?: {
    material_name: string;
    quantity: number;
    unit: string;
  };
  requirement_material?: {
    material_name: string;
    quantity: number;
    unit: string;
  };
  ai_reasoning?: AIRasoning;
  comprehensive_analysis?: ComprehensiveAnalysis;
}

export interface AIRasoning {
  semantic_similarity: number;
  trust_score: number;
  confidence: number;
  explanation: string;
  factors: string[];
  recommendations: string[];
}

export interface ComprehensiveAnalysis {
  readiness_assessment: ReadinessAssessment;
  financial_analysis: FinancialAnalysis;
  logistics_analysis: LogisticsAnalysis;
  carbon_analysis: CarbonAnalysis;
  waste_analysis: WasteAnalysis;
  risk_assessment: RiskAssessment;
  explanations: Record<string, string>;
  economic_summary: EconomicSummary;
  environmental_summary: EnvironmentalSummary;
  risk_summary: RiskSummary;
}

export interface ReadinessAssessment {
  overall_score: number;
  technical_readiness: number;
  operational_readiness: number;
  financial_readiness: number;
  regulatory_readiness: number;
  equipment_recommendations: EquipmentRecommendation[];
  training_requirements: string[];
  timeline_estimate: string;
}

export interface EquipmentRecommendation {
  equipment_name: string;
  purpose: string;
  estimated_cost: number;
  priority: 'low' | 'medium' | 'high';
  supplier_recommendations: string[];
}

export interface FinancialAnalysis {
  total_investment: number;
  operational_costs: number;
  revenue_potential: number;
  payback_period: number;
  roi: number;
  risk_factors: string[];
  funding_options: string[];
  cost_breakdown: CostBreakdown;
}

export interface CostBreakdown {
  equipment: number;
  installation: number;
  training: number;
  permits: number;
  operational: number;
  maintenance: number;
}

export interface LogisticsAnalysis {
  distance: number;
  transportation_cost: number;
  route_options: RouteOption[];
  storage_requirements: string[];
  handling_requirements: string[];
  delivery_schedule: string;
}

export interface RouteOption {
  route_id: string;
  distance: number;
  cost: number;
  duration: string;
  mode: 'road' | 'rail' | 'sea' | 'air';
  reliability: number;
  environmental_impact: number;
}

export interface CarbonAnalysis {
  current_emissions: number;
  projected_reduction: number;
  carbon_savings: number;
  carbon_price: number;
  carbon_credits: number;
  methodology: string;
  verification_requirements: string[];
}

export interface WasteAnalysis {
  current_waste: number;
  waste_reduction: number;
  waste_processing: string[];
  disposal_cost_savings: number;
  recycling_potential: number;
  circular_economy_impact: number;
}

export interface RiskAssessment {
  technical_risks: Risk[];
  financial_risks: Risk[];
  operational_risks: Risk[];
  regulatory_risks: Risk[];
  environmental_risks: Risk[];
  mitigation_strategies: string[];
  contingency_plans: string[];
}

export interface EconomicSummary {
  total_value: number;
  cost_savings: number;
  revenue_generation: number;
  investment_required: number;
  payback_period: number;
  roi: number;
  key_benefits: string[];
}

export interface EnvironmentalSummary {
  carbon_reduction: number;
  waste_reduction: number;
  energy_savings: number;
  water_savings: number;
  sustainability_score: number;
  environmental_benefits: string[];
}

export interface RiskSummary {
  overall_risk_level: 'low' | 'medium' | 'high';
  key_risks: string[];
  risk_mitigation: string[];
  success_probability: number;
}

export interface CompanyApplication {
  id: string;
  company_name: string;
  contact_email: string;
  contact_name: string;
  application_answers: Record<string, unknown>;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  reviewed_by?: string;
  reviewed_at?: string;
}

export interface AdminStats {
  total_companies: number;
  total_materials: number;
  total_connections: number;
  active_subscriptions: number;
  revenue_monthly: number;
  pending_applications: number;
  total_matches: number;
  active_matches: number;
  total_ai_listings: number;
  total_potential_value: number;
  total_carbon_reduction: number;
  average_sustainability_score: number;
  system_health_score: number;
}

export interface NetworkData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  clusters: string[][];
  metrics: {
    total_nodes: number;
    total_edges: number;
    density: number;
    average_clustering: number;
    symbiosis_potential: number;
  };
}

export interface NetworkNode {
  id: string;
  name?: string;
  industry?: string;
  location?: string;
  annual_waste?: number;
  carbon_footprint?: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
  explanation: Record<string, unknown>;
}

export interface AIMaterial {
  id: string;
  material_name: string;
  quantity: number;
  unit: string;
  type: 'waste' | 'requirement';
  confidence_score: number;
  ai_reasoning: string;
  potential_value: number;
  sustainability_score: number;
  logistics_cost: number;
  carbon_footprint: number;
  properties: MaterialProperties;
  analysis: MaterialAnalysis;
}

export interface AIMatch {
  id: string;
  waste_company: {
    id: string;
    name: string;
    industry: string;
    location: string;
  };
  requirement_company: {
    id: string;
    name: string;
    industry: string;
    location: string;
  };
  waste_material: AIMaterial;
  requirement_material: AIMaterial;
  match_score: number;
  symbiosis_potential: number;
  economic_value: number;
  sustainability_impact: number;
  ai_reasoning: AIRasoning;
  comprehensive_analysis: ComprehensiveAnalysis;
}

export interface PersonalizedRecommendation {
  id: string;
  title: string;
  description: string;
  category: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimated_impact: number;
  implementation_cost: number;
  timeframe: string;
  ai_reasoning: string;
  company_specific: boolean;
  market_alignment: number;
  sustainability_impact: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  user_id: string;
  title: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
  status: 'active' | 'archived';
}

export interface Notification {
  id: string;
  user_id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  read: boolean;
  created_at: string;
  action_url?: string;
  metadata?: Record<string, unknown>;
}

export interface Subscription {
  id: string;
  user_id: string;
  tier: 'free' | 'basic' | 'premium' | 'enterprise';
  status: 'active' | 'cancelled' | 'expired' | 'pending';
  created_at: string;
  expires_at?: string;
  features: string[];
  limits: Record<string, number>;
  pricing: {
    monthly: number;
    yearly: number;
  };
}

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  code?: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// Form Types
export interface LoginForm {
  email: string;
  password: string;
}

export interface RegisterForm {
  email: string;
  password: string;
  confirmPassword: string;
  companyName: string;
  industry: string;
  location: string;
  employeeCount: string;
}

export interface MaterialForm {
  material_name: string;
  quantity: number;
  unit: string;
  type: 'waste' | 'requirement';
  price_per_unit?: number;
  quality_grade?: string;
  description?: string;
}

export interface CompanyProfileForm {
  name: string;
  industry: string;
  location: string;
  employee_count: number;
  description?: string;
  website?: string;
  contact_email?: string;
  sustainability_goals?: string[];
}

// Event Types
export interface FormEvent<T = Element> {
  target: EventTarget & T;
}

export interface ChangeEvent<T = Element> {
  target: EventTarget & T;
}

// Component Props Types
export interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
}

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

export interface TableProps<T> {
  data: T[];
  columns: TableColumn<T>[];
  onRowClick?: (item: T) => void;
  loading?: boolean;
  pagination?: PaginationProps;
}

export interface TableColumn<T> {
  key: keyof T | string;
  header: string;
  render?: (value: unknown, item: T) => React.ReactNode;
  sortable?: boolean;
  width?: string;
}

export interface PaginationProps {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  itemsPerPage: number;
  onPageChange: (page: number) => void;
}

// Utility Types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type Required<T, K extends keyof T> = T & Required<Pick<T, K>>;

export type NonNullable<T> = T extends null | undefined ? never : T;

export type AsyncReturnType<T extends (...args: unknown[]) => Promise<unknown>> =
  T extends (...args: unknown[]) => Promise<infer R> ? R : unknown; 