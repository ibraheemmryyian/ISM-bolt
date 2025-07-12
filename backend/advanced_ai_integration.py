#!/usr/bin/env python3
"""
Advanced AI Integration Script
Demonstrates how to use the new AdvancedAIPromptsService with the four strategic prompts
for enhanced industrial symbiosis analysis.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import aiohttp
import redis
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# Import our advanced AI modules
from ai_service import AdvancedAIService, SymbiosisMatch, CompanyProfile, MaterialProfile
from advanced_ml_models import (
    SymbiosisPredictor, ModelManager, AdvancedClusteringModel, 
    AnomalyDetectionModel, FeatureEngineeringPipeline, EnsembleModel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIAnalysisResult:
    """Comprehensive AI analysis result"""
    analysis_id: str
    timestamp: datetime
    company_profiles: List[CompanyProfile]
    symbiosis_matches: List[SymbiosisMatch]
    network_analysis: Dict[str, Any]
    optimization_results: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    implementation_roadmap: Dict[str, Any]
    performance_metrics: Dict[str, float]
    ai_model_versions: Dict[str, str]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]

class AdvancedAIIntegration:
    """Advanced AI integration orchestrating all ML models and AI services"""
    
    def __init__(self):
        # Initialize AI services
        self.ai_service = AdvancedAIService()
        self.symbiosis_predictor = SymbiosisPredictor()
        self.model_manager = ModelManager()
        
        # Initialize specialized models
        self.clustering_model = AdvancedClusteringModel(method='dbscan', eps=0.3, min_samples=2)
        self.anomaly_detector = AnomalyDetectionModel(method='isolation_forest', contamination=0.1)
        self.feature_pipeline = FeatureEngineeringPipeline()
        
        # Initialize Redis for caching
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = 3600  # 1 hour
        
        # Analysis history
        self.analysis_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_matches': 0,
            'average_match_score': 0.0,
            'total_savings_predicted': 0.0,
            'total_carbon_reduction': 0.0
        }
    
    async def perform_comprehensive_analysis(self, companies_data: List[Dict]) -> AIAnalysisResult:
        """Perform comprehensive AI analysis on company data"""
        analysis_id = self._generate_analysis_id(companies_data)
        
        # Check cache
        cached_result = self._get_cached_analysis(analysis_id)
        if cached_result:
            logger.info(f"Returning cached analysis: {analysis_id}")
            return cached_result
        
        logger.info(f"Starting comprehensive analysis for {len(companies_data)} companies")
        start_time = datetime.now()
        
        try:
            # Step 1: Create company profiles
            company_profiles = await self._create_company_profiles(companies_data)
            
            # Step 2: Perform clustering analysis
            clustering_results = await self._perform_clustering_analysis(company_profiles)
            
            # Step 3: Detect anomalies
            anomaly_results = await self._detect_anomalies(company_profiles)
            
            # Step 4: Find symbiosis matches
            symbiosis_matches = await self._find_symbiosis_matches(company_profiles)
            
            # Step 5: Analyze network structure
            network_analysis = await self._analyze_network_structure(company_profiles, symbiosis_matches)
            
            # Step 6: Optimize network
            optimization_results = await self._optimize_network(company_profiles, symbiosis_matches)
            
            # Step 7: Generate recommendations
            recommendations = await self._generate_recommendations(company_profiles, symbiosis_matches)
            
            # Step 8: Assess risks
            risk_assessment = await self._assess_risks(company_profiles, symbiosis_matches)
            
            # Step 9: Create implementation roadmap
            implementation_roadmap = await self._create_implementation_roadmap(symbiosis_matches)
            
            # Step 10: Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(symbiosis_matches)
            
            # Create comprehensive result
            result = AIAnalysisResult(
                analysis_id=analysis_id,
                timestamp=datetime.now(),
                company_profiles=company_profiles,
                symbiosis_matches=symbiosis_matches,
                network_analysis=network_analysis,
                optimization_results=optimization_results,
                recommendations=recommendations,
                risk_assessment=risk_assessment,
                implementation_roadmap=implementation_roadmap,
                performance_metrics=performance_metrics,
                ai_model_versions={
                    'symbiosis_predictor': 'v2.0',
                    'clustering_model': 'v1.5',
                    'anomaly_detector': 'v1.2',
                    'network_analyzer': 'v2.1'
                },
                confidence_scores={
                    'matching_confidence': np.mean([m.confidence for m in symbiosis_matches]) if symbiosis_matches else 0.0,
                    'clustering_confidence': clustering_results.get('silhouette_score', 0.0),
                    'anomaly_detection_confidence': anomaly_results.get('detection_confidence', 0.0)
                },
                metadata={
                    'analysis_duration': (datetime.now() - start_time).total_seconds(),
                    'companies_analyzed': len(companies_data),
                    'matches_found': len(symbiosis_matches),
                    'cache_hit': False
                }
            )
            
            # Cache result
            self._cache_analysis_result(analysis_id, result)
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Add to history
            self.analysis_history.append(result)
            
            logger.info(f"Analysis completed successfully in {result.metadata['analysis_duration']:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise Exception(f"Comprehensive analysis failed: {str(e)}")
    
    async def _create_company_profiles(self, companies_data: List[Dict]) -> List[CompanyProfile]:
        """Create comprehensive company profiles"""
        profiles = []
        
        for company_data in companies_data:
            try:
                # Use AI service to create profile
                profile = self.ai_service._create_company_profile(company_data)
                profiles.append(profile)
            except Exception as e:
                logger.warning(f"Failed to create profile for company {company_data.get('name', 'Unknown')}: {str(e)}")
                continue
        
        return profiles
    
    async def _perform_clustering_analysis(self, company_profiles: List[CompanyProfile]) -> Dict[str, Any]:
        """Perform clustering analysis on companies"""
        if len(company_profiles) < 2:
            return {'clusters': [], 'silhouette_score': 0.0}
        
        # Extract features for clustering
        features = []
        for profile in company_profiles:
            company_features = [
                profile.sustainability_score,
                profile.carbon_footprint,
                profile.annual_revenue,
                profile.employee_count,
                profile.symbiosis_potential
            ]
            features.append(company_features)
        
        features = np.array(features)
        
        # Perform clustering
        self.clustering_model.fit(features)
        cluster_labels = self.clustering_model.predict(features)
        
        # Calculate clustering quality metrics
        if len(np.unique(cluster_labels)) > 1:
            silhouette_score = self._calculate_silhouette_score(features, cluster_labels)
        else:
            silhouette_score = 0.0
        
        # Group companies by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(company_profiles[i].company_id)
        
        return {
            'clusters': clusters,
            'cluster_labels': cluster_labels.tolist(),
            'silhouette_score': silhouette_score,
            'num_clusters': len(np.unique(cluster_labels)),
            'cluster_centers': self.clustering_model.get_cluster_centers().tolist() if self.clustering_model.get_cluster_centers() is not None else []
        }
    
    async def _detect_anomalies(self, company_profiles: List[CompanyProfile]) -> Dict[str, Any]:
        """Detect anomalies in company profiles"""
        if len(company_profiles) < 3:
            return {'anomalies': [], 'detection_confidence': 0.0}
        
        # Extract features for anomaly detection
        features = []
        for profile in company_profiles:
            company_features = [
                profile.sustainability_score,
                profile.carbon_footprint,
                profile.annual_revenue,
                profile.employee_count,
                profile.symbiosis_potential
            ]
            features.append(company_features)
        
        features = np.array(features)
        
        # Perform anomaly detection
        self.anomaly_detector.fit(features)
        anomaly_labels = self.anomaly_detector.predict(features)
        anomaly_scores = self.anomaly_detector.score_samples(features)
        
        # Identify anomalous companies
        anomalies = []
        for i, (label, score) in enumerate(zip(anomaly_labels, anomaly_scores)):
            if label == -1:  # Anomaly detected
                anomalies.append({
                    'company_id': company_profiles[i].company_id,
                    'company_name': company_profiles[i].name,
                    'anomaly_score': float(score),
                    'anomaly_type': self._classify_anomaly_type(company_profiles[i], score)
                })
        
        return {
            'anomalies': anomalies,
            'anomaly_labels': anomaly_labels.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'detection_confidence': 1.0 - np.mean(anomaly_scores) if len(anomaly_scores) > 0 else 0.0,
            'anomaly_threshold': np.percentile(anomaly_scores, 10) if len(anomaly_scores) > 0 else 0.0
        }
    
    async def _find_symbiosis_matches(self, company_profiles: List[CompanyProfile]) -> List[SymbiosisMatch]:
        """Find optimal symbiosis matches using advanced algorithms"""
        if len(company_profiles) < 2:
            return []
        
        # Use the advanced symbiosis analyzer
        analysis_result = self.ai_service.symbiosis_analyzer.analyze_symbiosis_network(company_profiles)
        
        # Convert to SymbiosisMatch objects
        matches = []
        for match_data in analysis_result['matches']:
            match = SymbiosisMatch(**match_data)
            matches.append(match)
        
        return matches
    
    async def _analyze_network_structure(self, company_profiles: List[CompanyProfile], 
                                       symbiosis_matches: List[SymbiosisMatch]) -> Dict[str, Any]:
        """Analyze the structure of the symbiosis network"""
        if len(symbiosis_matches) == 0:
            return {'network_density': 0.0, 'connected_components': 0, 'centrality_measures': {}}
        
        # Create network graph
        G = self._create_network_graph(company_profiles, symbiosis_matches)
        
        # Calculate network metrics
        metrics = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'network_density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G),
            'largest_component_size': len(max(nx.connected_components(G), key=len)) if nx.number_connected_components(G) > 0 else 0,
            'network_diameter': nx.diameter(G) if nx.is_connected(G) else 0,
            'average_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
            'centrality_measures': self._calculate_centrality_measures(G),
            'community_structure': self._detect_communities(G),
            'network_efficiency': self._calculate_network_efficiency(symbiosis_matches),
            'resilience_metrics': self._calculate_resilience_metrics(G)
        }
        
        return metrics
    
    async def _optimize_network(self, company_profiles: List[CompanyProfile], 
                              symbiosis_matches: List[SymbiosisMatch]) -> Dict[str, Any]:
        """Optimize the symbiosis network"""
        if len(symbiosis_matches) == 0:
            return {'optimization_score': 0.0, 'recommendations': []}
        
        # Calculate optimization objectives
        total_savings = sum(match.expected_savings for match in symbiosis_matches)
        total_carbon_reduction = sum(match.carbon_reduction for match in symbiosis_matches)
        total_waste_reduction = sum(match.waste_reduction for match in symbiosis_matches)
        
        # Calculate network efficiency
        network_efficiency = self._calculate_network_efficiency(symbiosis_matches)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(company_profiles, symbiosis_matches)
        
        return {
            'optimization_score': network_efficiency,
            'total_savings': total_savings,
            'total_carbon_reduction': total_carbon_reduction,
            'total_waste_reduction': total_waste_reduction,
            'network_efficiency': network_efficiency,
            'recommendations': recommendations,
            'pareto_frontier': self._find_pareto_frontier(symbiosis_matches),
            'implementation_priority': self._prioritize_implementation(symbiosis_matches)
        }
    
    async def _generate_recommendations(self, company_profiles: List[CompanyProfile], 
                                      symbiosis_matches: List[SymbiosisMatch]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # High-priority matches
        high_priority_matches = [m for m in symbiosis_matches if m.priority == 'high']
        if high_priority_matches:
            recommendations.append({
                'type': 'high_priority_matches',
                'title': 'High-Priority Symbiosis Opportunities',
                'description': f'Found {len(high_priority_matches)} high-priority symbiosis opportunities',
                'matches': [asdict(m) for m in high_priority_matches[:5]],
                'priority': 'high',
                'expected_impact': sum(m.expected_savings for m in high_priority_matches)
            })
        
        # Material optimization opportunities
        material_opportunities = self._identify_material_opportunities(company_profiles)
        if material_opportunities:
            recommendations.append({
                'type': 'material_optimization',
                'title': 'Material Optimization Opportunities',
                'description': 'Opportunities to optimize material usage and reduce waste',
                'opportunities': material_opportunities,
                'priority': 'medium',
                'expected_impact': sum(opp['potential_savings'] for opp in material_opportunities)
            })
        
        # Technology upgrade recommendations
        tech_recommendations = self._generate_tech_recommendations(company_profiles)
        if tech_recommendations:
            recommendations.append({
                'type': 'technology_upgrades',
                'title': 'Technology Upgrade Recommendations',
                'description': 'Recommended technology upgrades for improved symbiosis',
                'recommendations': tech_recommendations,
                'priority': 'medium',
                'expected_impact': sum(rec['expected_benefit'] for rec in tech_recommendations)
            })
        
        return recommendations
    
    async def _assess_risks(self, company_profiles: List[CompanyProfile], 
                          symbiosis_matches: List[SymbiosisMatch]) -> Dict[str, Any]:
        """Assess risks in the symbiosis network"""
        risks = {
            'high_risk_matches': [],
            'medium_risk_matches': [],
            'low_risk_matches': [],
            'network_risks': [],
            'implementation_risks': [],
            'regulatory_risks': [],
            'financial_risks': [],
            'overall_risk_score': 0.0
        }
        
        # Assess individual match risks
        for match in symbiosis_matches:
            risk_score = self._calculate_match_risk(match)
            match_risk = {
                'match_id': f"{match.company_id}_{match.partner_id}",
                'risk_score': risk_score,
                'risk_factors': self._identify_risk_factors(match),
                'mitigation_strategies': self._generate_mitigation_strategies(match)
            }
            
            if risk_score > 0.7:
                risks['high_risk_matches'].append(match_risk)
            elif risk_score > 0.4:
                risks['medium_risk_matches'].append(match_risk)
            else:
                risks['low_risk_matches'].append(match_risk)
        
        # Calculate overall risk score
        if symbiosis_matches:
            risks['overall_risk_score'] = np.mean([m.risk_assessment for m in symbiosis_matches])
        
        return risks
    
    async def _create_implementation_roadmap(self, symbiosis_matches: List[SymbiosisMatch]) -> Dict[str, Any]:
        """Create implementation roadmap for symbiosis matches"""
        if not symbiosis_matches:
            return {'phases': [], 'total_duration': 0, 'total_cost': 0}
        
        # Group matches by complexity and priority
        simple_matches = [m for m in symbiosis_matches if m.complexity_level == 'simple']
        moderate_matches = [m for m in symbiosis_matches if m.complexity_level == 'moderate']
        complex_matches = [m for m in symbiosis_matches if m.complexity_level == 'complex']
        
        # Create implementation phases
        phases = []
        
        # Phase 1: Simple matches (0-6 months)
        if simple_matches:
            phases.append({
                'phase': 1,
                'name': 'Quick Wins',
                'duration': '0-6 months',
                'matches': [asdict(m) for m in simple_matches],
                'expected_savings': sum(m.expected_savings for m in simple_matches),
                'implementation_cost': sum(m.expected_savings * 0.1 for m in simple_matches),
                'success_probability': 0.9
            })
        
        # Phase 2: Moderate matches (6-18 months)
        if moderate_matches:
            phases.append({
                'phase': 2,
                'name': 'Medium-Term Projects',
                'duration': '6-18 months',
                'matches': [asdict(m) for m in moderate_matches],
                'expected_savings': sum(m.expected_savings for m in moderate_matches),
                'implementation_cost': sum(m.expected_savings * 0.2 for m in moderate_matches),
                'success_probability': 0.7
            })
        
        # Phase 3: Complex matches (18+ months)
        if complex_matches:
            phases.append({
                'phase': 3,
                'name': 'Long-Term Strategic Projects',
                'duration': '18+ months',
                'matches': [asdict(m) for m in complex_matches],
                'expected_savings': sum(m.expected_savings for m in complex_matches),
                'implementation_cost': sum(m.expected_savings * 0.3 for m in complex_matches),
                'success_probability': 0.5
            })
        
        total_duration = len(phases) * 6  # months
        total_cost = sum(phase['implementation_cost'] for phase in phases)
        
        return {
            'phases': phases,
            'total_duration': total_duration,
            'total_cost': total_cost,
            'total_expected_savings': sum(m.expected_savings for m in symbiosis_matches),
            'roi': (sum(m.expected_savings for m in symbiosis_matches) - total_cost) / total_cost if total_cost > 0 else 0
        }
    
    def _calculate_performance_metrics(self, symbiosis_matches: List[SymbiosisMatch]) -> Dict[str, float]:
        """Calculate performance metrics for the analysis"""
        if not symbiosis_matches:
            return {
                'total_matches': 0,
                'average_match_score': 0.0,
                'total_savings': 0.0,
                'total_carbon_reduction': 0.0,
                'network_efficiency': 0.0
            }
        
        return {
            'total_matches': len(symbiosis_matches),
            'average_match_score': np.mean([m.match_score for m in symbiosis_matches]),
            'total_savings': sum(m.expected_savings for m in symbiosis_matches),
            'total_carbon_reduction': sum(m.carbon_reduction for m in symbiosis_matches),
            'total_waste_reduction': sum(m.waste_reduction for m in symbiosis_matches),
            'network_efficiency': self._calculate_network_efficiency(symbiosis_matches),
            'high_priority_matches': len([m for m in symbiosis_matches if m.priority == 'high']),
            'medium_priority_matches': len([m for m in symbiosis_matches if m.priority == 'medium']),
            'low_priority_matches': len([m for m in symbiosis_matches if m.priority == 'low'])
        }
    
    def _generate_analysis_id(self, companies_data: List[Dict]) -> str:
        """Generate unique analysis ID"""
        data_hash = hashlib.md5(json.dumps(companies_data, sort_keys=True).encode()).hexdigest()
        return f"analysis_{data_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _get_cached_analysis(self, analysis_id: str) -> Optional[AIAnalysisResult]:
        """Get cached analysis result"""
        try:
            cached_data = self.redis_client.get(f"analysis_{analysis_id}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Failed to get cached analysis: {str(e)}")
        return None
    
    def _cache_analysis_result(self, analysis_id: str, result: AIAnalysisResult):
        """Cache analysis result"""
        try:
            result.metadata['cache_hit'] = True
            self.redis_client.setex(
                f"analysis_{analysis_id}", 
                self.cache_ttl, 
                json.dumps(asdict(result), default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to cache analysis result: {str(e)}")
    
    def _update_performance_metrics(self, result: AIAnalysisResult):
        """Update global performance metrics"""
        self.performance_metrics['total_analyses'] += 1
        self.performance_metrics['successful_matches'] += len(result.symbiosis_matches)
        self.performance_metrics['total_savings_predicted'] += result.performance_metrics['total_savings']
        self.performance_metrics['total_carbon_reduction'] += result.performance_metrics['total_carbon_reduction']
        
        if result.symbiosis_matches:
            self.performance_metrics['average_match_score'] = np.mean([
                self.performance_metrics['average_match_score'],
                result.performance_metrics['average_match_score']
            ])
    
    # Helper methods (implementations would be added here)
    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(features, labels)
        except:
            return 0.0
    
    def _classify_anomaly_type(self, profile: CompanyProfile, score: float) -> str:
        """Classify the type of anomaly"""
        if score < -0.5:
            return 'severe_anomaly'
        elif score < -0.2:
            return 'moderate_anomaly'
        else:
            return 'minor_anomaly'
    
    def _create_network_graph(self, company_profiles: List[CompanyProfile], 
                            symbiosis_matches: List[SymbiosisMatch]):
        """Create network graph from matches"""
        import networkx as nx
        G = nx.Graph()
        
        # Add nodes
        for profile in company_profiles:
            G.add_node(profile.company_id, **asdict(profile))
        
        # Add edges
        for match in symbiosis_matches:
            G.add_edge(match.company_id, match.partner_id, 
                      weight=match.match_score,
                      **asdict(match))
        
        return G
    
    def _calculate_centrality_measures(self, G):
        """Calculate centrality measures for network"""
        return {
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'closeness_centrality': nx.closeness_centrality(G)
        }
    
    def _detect_communities(self, G):
        """Detect communities in network"""
        try:
            communities = nx.community.louvain_communities(G)
            return {
                'num_communities': len(communities),
                'community_sizes': [len(comm) for comm in communities],
                'modularity': nx.community.modularity(G, communities)
            }
        except:
            return {'num_communities': 0, 'community_sizes': [], 'modularity': 0.0}
    
    def _calculate_network_efficiency(self, matches: List[SymbiosisMatch]) -> float:
        """Calculate network efficiency"""
        if not matches:
            return 0.0
        return np.mean([m.match_score for m in matches])
    
    def _calculate_resilience_metrics(self, G):
        """Calculate network resilience metrics"""
        return {
            'connectivity': nx.node_connectivity(G) if G.number_of_nodes() > 1 else 0,
            'edge_connectivity': nx.edge_connectivity(G) if G.number_of_edges() > 0 else 0
        }
    
    def _generate_optimization_recommendations(self, company_profiles: List[CompanyProfile], 
                                            symbiosis_matches: List[SymbiosisMatch]) -> List[Dict]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Add optimization logic here
        recommendations.append({
            'type': 'network_optimization',
            'title': 'Network Optimization',
            'description': 'Optimize network structure for better efficiency',
            'priority': 'high'
        })
        
        return recommendations
    
    def _find_pareto_frontier(self, matches: List[SymbiosisMatch]) -> List[Dict]:
        """Find Pareto frontier for multi-objective optimization"""
        # Implementation would go here
        return []
    
    def _prioritize_implementation(self, matches: List[SymbiosisMatch]) -> List[Dict]:
        """Prioritize matches for implementation"""
        return [asdict(match) for match in sorted(matches, key=lambda x: x.priority, reverse=True)]
    
    def _identify_material_opportunities(self, company_profiles: List[CompanyProfile]) -> List[Dict]:
        """Identify material optimization opportunities"""
        opportunities = []
        
        # Add material optimization logic here
        for profile in company_profiles:
            if profile.materials_inventory:
                opportunities.append({
                    'company_id': profile.company_id,
                    'company_name': profile.name,
                    'opportunity_type': 'material_optimization',
                    'potential_savings': 10000.0,
                    'description': 'Optimize material usage and reduce waste'
                })
        
        return opportunities
    
    def _generate_tech_recommendations(self, company_profiles: List[CompanyProfile]) -> List[Dict]:
        """Generate technology upgrade recommendations"""
        recommendations = []
        
        # Add technology recommendation logic here
        for profile in company_profiles:
            recommendations.append({
                'company_id': profile.company_id,
                'company_name': profile.name,
                'technology': 'IoT Sensors',
                'expected_benefit': 5000.0,
                'description': 'Implement IoT sensors for better resource tracking'
            })
        
        return recommendations
    
    def _calculate_match_risk(self, match: SymbiosisMatch) -> float:
        """Calculate risk score for a match"""
        return match.risk_assessment
    
    def _identify_risk_factors(self, match: SymbiosisMatch) -> List[str]:
        """Identify risk factors for a match"""
        risk_factors = []
        
        if match.geographic_proximity < 0.5:
            risk_factors.append('geographic_distance')
        
        if match.regulatory_compliance < 0.7:
            risk_factors.append('regulatory_compliance')
        
        if match.technology_compatibility < 0.6:
            risk_factors.append('technology_incompatibility')
        
        return risk_factors
    
    def _generate_mitigation_strategies(self, match: SymbiosisMatch) -> List[str]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        if match.geographic_proximity < 0.5:
            strategies.append('Implement efficient logistics solutions')
        
        if match.regulatory_compliance < 0.7:
            strategies.append('Ensure regulatory compliance before implementation')
        
        if match.technology_compatibility < 0.6:
            strategies.append('Invest in technology integration')
        
        return strategies

# Global instance
advanced_ai_integration = AdvancedAIIntegration() 