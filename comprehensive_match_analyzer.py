import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime
import logging

# Import all the analysis engines
from refinement_analysis_engine import refinement_engine
from financial_analysis_engine import FinancialAnalysisEngine
from logistics_cost_engine import LogisticsCostEngine
from carbon_calculation_engine import carbon_engine
from waste_tracking_engine import waste_engine

logger = logging.getLogger(__name__)

@dataclass
class Location:
    """Location data structure"""
    name: str
    latitude: float
    longitude: float
    country: str
    city: str

@dataclass
class ComprehensiveMatchAnalysis:
    """Complete analysis of a potential match"""
    match_id: str
    buyer_id: str
    seller_id: str
    material_type: str
    quantity_ton: float
    
    # Readiness and refinement analysis
    readiness_assessment: Dict
    
    # Financial analysis
    financial_analysis: Dict
    
    # Logistics analysis
    logistics_analysis: Dict
    
    # Carbon analysis
    carbon_analysis: Dict
    
    # Waste analysis
    waste_analysis: Dict
    
    # Overall match assessment
    overall_score: float
    match_quality: str
    confidence_level: float
    risk_level: str
    
    # Detailed explanations
    explanations: Dict
    
    # Recommendations
    recommendations: List[str]
    
    # Economic summary
    economic_summary: Dict
    
    # Environmental summary
    environmental_summary: Dict
    
    # Risk summary
    risk_summary: Dict
    
    analysis_date: str
    methodology: str

class ComprehensiveMatchAnalyzer:
    def __init__(self):
        self.logistics_engine = LogisticsCostEngine()
        
        # Weight factors for different analysis components
        self.analysis_weights = {
            'financial': 0.35,
            'readiness': 0.25,
            'logistics': 0.20,
            'carbon': 0.15,
            'waste': 0.05
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.85,
            'good': 0.70,
            'moderate': 0.50,
            'poor': 0.30
        }

    def analyze_match_comprehensive(self, buyer_data: Dict, seller_data: Dict, 
                                  match_data: Dict) -> ComprehensiveMatchAnalysis:
        """Perform comprehensive analysis of a potential match"""
        try:
            match_id = match_data.get('id', str(uuid.uuid4()))
            
            # 1. Readiness and Refinement Analysis
            readiness_assessment = self._analyze_readiness(buyer_data, seller_data, match_data)
            
            # 2. Logistics Analysis
            logistics_analysis = self._analyze_logistics(buyer_data, seller_data, match_data)
            
            # 3. Financial Analysis
            financial_analysis = self._analyze_financials(buyer_data, seller_data, match_data, logistics_analysis)
            
            # 4. Carbon Analysis
            carbon_analysis = self._analyze_carbon(buyer_data, seller_data, match_data, logistics_analysis)
            
            # 5. Waste Analysis
            waste_analysis = self._analyze_waste(buyer_data, seller_data, match_data)
            
            # 6. Overall Assessment
            overall_score, match_quality, confidence_level, risk_level = self._calculate_overall_assessment(
                readiness_assessment, financial_analysis, logistics_analysis, carbon_analysis, waste_analysis
            )
            
            # 7. Generate Explanations
            explanations = self._generate_explanations(
                readiness_assessment, financial_analysis, logistics_analysis, carbon_analysis, waste_analysis
            )
            
            # 8. Generate Recommendations
            recommendations = self._generate_recommendations(
                readiness_assessment, financial_analysis, logistics_analysis, carbon_analysis, waste_analysis
            )
            
            # 9. Create Summaries
            economic_summary = self._create_economic_summary(financial_analysis)
            environmental_summary = self._create_environmental_summary(carbon_analysis, waste_analysis)
            risk_summary = self._create_risk_summary(readiness_assessment, financial_analysis, logistics_analysis)
            
            return ComprehensiveMatchAnalysis(
                match_id=match_id,
                buyer_id=buyer_data.get('id', ''),
                seller_id=seller_data.get('id', ''),
                material_type=match_data.get('material_type', ''),
                quantity_ton=match_data.get('quantity', 0),
                readiness_assessment=readiness_assessment,
                financial_analysis=financial_analysis,
                logistics_analysis=logistics_analysis,
                carbon_analysis=carbon_analysis,
                waste_analysis=waste_analysis,
                overall_score=overall_score,
                match_quality=match_quality,
                confidence_level=confidence_level,
                risk_level=risk_level,
                explanations=explanations,
                recommendations=recommendations,
                economic_summary=economic_summary,
                environmental_summary=environmental_summary,
                risk_summary=risk_summary,
                analysis_date=datetime.now().isoformat(),
                methodology='Comprehensive multi-engine analysis with risk assessment'
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive match analysis: {e}")
            return self._create_error_analysis(buyer_data, seller_data, match_data)

    def _analyze_readiness(self, buyer_data: Dict, seller_data: Dict, match_data: Dict) -> Dict:
        """Analyze material readiness and refinement requirements"""
        try:
            # Prepare material data for refinement analysis
            material_data = {
                'id': match_data.get('id'),
                'material_type': match_data.get('material_type'),
                'quantity': match_data.get('quantity', 1.0),
                'impurity_level': match_data.get('impurity_level', 0.2),
                'quality_factor': match_data.get('quality_factor', 0.8),
                'market_price_per_ton': match_data.get('market_price_per_ton', 500),
                'waste_price_per_ton': match_data.get('waste_price_per_ton', 100)
            }
            
            # Perform readiness assessment
            readiness_assessment = refinement_engine.assess_material_readiness(material_data, buyer_data)
            
            return {
                'is_ready_for_use': readiness_assessment.is_ready_for_use,
                'readiness_score': readiness_assessment.readiness_score,
                'refinement_required': readiness_assessment.refinement_required,
                'refinement_requirements': readiness_assessment.refinement_requirements.__dict__ if readiness_assessment.refinement_requirements else None,
                'buyer_equipment_available': readiness_assessment.buyer_equipment_available,
                'missing_equipment': readiness_assessment.missing_equipment,
                'equipment_recommendations': [eq.__dict__ for eq in readiness_assessment.equipment_recommendations],
                'total_refinement_cost': readiness_assessment.total_refinement_cost,
                'total_equipment_cost': readiness_assessment.total_equipment_cost,
                'payback_period_months': readiness_assessment.payback_period_months,
                'risk_assessment': readiness_assessment.risk_assessment,
                'recommendations': readiness_assessment.recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in readiness analysis: {e}")
            return {'error': str(e)}

    def _analyze_logistics(self, buyer_data: Dict, seller_data: Dict, match_data: Dict) -> Dict:
        """Analyze logistics and transportation"""
        try:
            # Create location objects
            buyer_location = Location(
                name=buyer_data.get('location', 'Unknown'),
                latitude=buyer_data.get('latitude', 0),
                longitude=buyer_data.get('longitude', 0),
                country=buyer_data.get('country', 'Unknown'),
                city=buyer_data.get('city', 'Unknown')
            )
            
            seller_location = Location(
                name=seller_data.get('location', 'Unknown'),
                latitude=seller_data.get('latitude', 0),
                longitude=seller_data.get('longitude', 0),
                country=seller_data.get('country', 'Unknown'),
                city=seller_data.get('city', 'Unknown')
            )
            
            # Calculate distance
            distance = self.logistics_engine.calculate_distance(buyer_location, seller_location)
            
            # Get route planning
            cargo_weight = match_data.get('quantity', 1.0)
            cargo_value = cargo_weight * match_data.get('market_price_per_ton', 500)
            
            routes = self.logistics_engine.get_route_planning(
                seller_location, buyer_location, cargo_weight, cargo_value
            )
            
            # Select best route
            best_route = routes[0] if routes else None
            
            # Calculate carbon impact
            carbon_impact = None
            if best_route:
                carbon_impact = self.logistics_engine.calculate_carbon_impact(best_route, cargo_weight)
            
            return {
                'distance_km': distance,
                'transit_days': best_route.total_duration / 24 if best_route else 3,
                'transport_cost': best_route.total_cost if best_route else 0,
                'carbon_kg': best_route.total_carbon if best_route else 0,
                'carbon_impact': carbon_impact,
                'route_options': [route.__dict__ for route in routes[:3]],  # Top 3 routes
                'best_route': best_route.__dict__ if best_route else None,
                'international': buyer_location.country != seller_location.country
            }
            
        except Exception as e:
            logger.error(f"Error in logistics analysis: {e}")
            return {'error': str(e)}

    def _analyze_financials(self, buyer_data: Dict, seller_data: Dict, match_data: Dict, 
                           logistics_analysis: Dict) -> Dict:
        """Analyze financial aspects of the match"""
        try:
            # Initialize financial analysis engine
            financial_engine = FinancialAnalysisEngine()
            
            # Prepare material data for analysis
            material_data = {
                'material_name': match_data.get('material_type', ''),
                'quantity': match_data.get('quantity', 1.0),
                'unit': 'ton',
                'type': 'waste' if match_data.get('is_waste', True) else 'requirement'
            }
            
            # Calculate distance from logistics analysis
            distance_km = logistics_analysis.get('distance_km', 50)
            
            # Perform financial analysis
            financial_analysis = financial_engine.calculate_material_financials(material_data, distance_km)
            
            # Add additional financial metrics
            financial_analysis.update({
                'buyer_savings': financial_analysis.get('net_savings', 0),
                'seller_revenue': financial_analysis.get('market_value', 0),
                'total_transport_cost': financial_analysis.get('transport_cost', 0),
                'total_processing_cost': financial_analysis.get('processing_cost', 0),
                'roi_percentage': financial_analysis.get('roi_percentage', 0)
            })
            
            return financial_analysis
            
        except Exception as e:
            logger.error(f"Error in financial analysis: {e}")
            return {'error': str(e)}

    def _analyze_carbon(self, buyer_data: Dict, seller_data: Dict, match_data: Dict, 
                       logistics_analysis: Dict) -> Dict:
        """Analyze carbon impact and savings"""
        try:
            # Calculate carbon footprint for both companies
            buyer_carbon = carbon_engine.calculate_company_carbon_footprint(buyer_data)
            seller_carbon = carbon_engine.calculate_company_carbon_footprint(seller_data)
            
            # Calculate transport carbon
            transport_carbon = logistics_analysis.get('carbon_kg', 0)
            
            # Calculate carbon savings from using waste vs fresh material
            material_type = match_data.get('material_type', '').lower()
            quantity = match_data.get('quantity', 1.0)
            
            # Estimate carbon savings (waste typically has lower embodied carbon)
            carbon_savings_per_ton = {
                'steel': 1800,  # kg CO2 saved per ton
                'aluminum': 17000,
                'plastic': 2000,
                'paper': 800,
                'textiles': 1500
            }
            
            carbon_savings = carbon_savings_per_ton.get(material_type, 1000) * quantity
            
            # Net carbon impact (savings minus transport)
            net_carbon_savings = carbon_savings - transport_carbon
            
            return {
                'buyer_carbon_footprint': buyer_carbon,
                'seller_carbon_footprint': seller_carbon,
                'transport_carbon_kg': transport_carbon,
                'material_carbon_savings_kg': carbon_savings,
                'net_carbon_savings_kg': net_carbon_savings,
                'carbon_savings_percentage': (carbon_savings / (carbon_savings + transport_carbon)) * 100 if (carbon_savings + transport_carbon) > 0 else 0,
                'carbon_equivalent': {
                    'trees_needed': net_carbon_savings / 22,
                    'car_km_equivalent': net_carbon_savings / 0.2,
                    'flight_km_equivalent': net_carbon_savings / 0.25
                }
            }
            
        except Exception as e:
            logger.error(f"Error in carbon analysis: {e}")
            return {'error': str(e)}

    def _analyze_waste(self, buyer_data: Dict, seller_data: Dict, match_data: Dict) -> Dict:
        """Analyze waste management aspects"""
        try:
            # Calculate waste profile for seller
            seller_waste_profile = waste_engine.calculate_company_waste_profile(seller_data)
            
            # Calculate waste costs
            waste_costs = waste_engine._calculate_waste_costs(
                seller_waste_profile.get('waste_by_type', {}),
                seller_waste_profile.get('material_waste', {})
            )
            
            return {
                'seller_waste_profile': seller_waste_profile,
                'waste_costs': waste_costs,
                'waste_management_efficiency': seller_waste_profile.get('efficiency_score', 0),
                'recycling_potential': seller_waste_profile.get('recycling_potential', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in waste analysis: {e}")
            return {'error': str(e)}

    def _calculate_overall_assessment(self, readiness_assessment: Dict, financial_analysis: Dict,
                                    logistics_analysis: Dict, carbon_analysis: Dict, 
                                    waste_analysis: Dict) -> Tuple[float, str, float, str]:
        """Calculate overall match assessment"""
        try:
            # Calculate component scores
            readiness_score = readiness_assessment.get('readiness_score', 0)
            financial_score = min(1.0, financial_analysis.get('scenario_comparison', {}).get('savings_percentage', 0) / 100)
            logistics_score = 1 - (logistics_analysis.get('transport_cost', 0) / 1000)  # Normalize transport cost
            carbon_score = min(1.0, carbon_analysis.get('net_carbon_savings_kg', 0) / 1000)  # Normalize carbon savings
            waste_score = waste_analysis.get('waste_management_efficiency', 0)
            
            # Calculate weighted overall score
            overall_score = (
                readiness_score * self.analysis_weights['readiness'] +
                financial_score * self.analysis_weights['financial'] +
                logistics_score * self.analysis_weights['logistics'] +
                carbon_score * self.analysis_weights['carbon'] +
                waste_score * self.analysis_weights['waste']
            )
            
            # Determine match quality
            if overall_score >= self.quality_thresholds['excellent']:
                match_quality = 'Excellent'
            elif overall_score >= self.quality_thresholds['good']:
                match_quality = 'Good'
            elif overall_score >= self.quality_thresholds['moderate']:
                match_quality = 'Moderate'
            elif overall_score >= self.quality_thresholds['poor']:
                match_quality = 'Poor'
            else:
                match_quality = 'Very Poor'
            
            # Calculate confidence level
            confidence_factors = []
            if 'error' not in readiness_assessment:
                confidence_factors.append(0.9)
            if 'error' not in financial_analysis:
                confidence_factors.append(0.8)
            if 'error' not in logistics_analysis:
                confidence_factors.append(0.7)
            if 'error' not in carbon_analysis:
                confidence_factors.append(0.6)
            if 'error' not in waste_analysis:
                confidence_factors.append(0.5)
            
            confidence_level = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.3
            
            # Determine risk level
            risk_factors = []
            if readiness_assessment.get('risk_assessment', {}).get('risk_level') == 'high':
                risk_factors.append(1)
            if financial_analysis.get('scenario_comparison', {}).get('risk_assessment', {}).get('overall_risk_level') == 'high':
                risk_factors.append(1)
            
            risk_level = 'high' if len(risk_factors) >= 2 else 'medium' if len(risk_factors) >= 1 else 'low'
            
            return overall_score, match_quality, confidence_level, risk_level
            
        except Exception as e:
            logger.error(f"Error in overall assessment: {e}")
            return 0.0, 'Error', 0.0, 'high'

    def _generate_explanations(self, readiness_assessment: Dict, financial_analysis: Dict,
                             logistics_analysis: Dict, carbon_analysis: Dict, 
                             waste_analysis: Dict) -> Dict:
        """Generate detailed explanations for all aspects"""
        explanations = {}
        
        # Readiness explanation
        if 'error' not in readiness_assessment:
            if readiness_assessment.get('is_ready_for_use'):
                explanations['readiness'] = "Material is ready for immediate use without processing"
            else:
                explanations['readiness'] = f"Material requires {len(readiness_assessment.get('refinement_requirements', {}).get('required_processes', []))} processing steps"
        
        # Financial explanation
        if 'error' not in financial_analysis:
            scenario_comparison = financial_analysis.get('scenario_comparison', {})
            net_savings = scenario_comparison.get('net_savings', 0)
            if net_savings > 0:
                explanations['financial'] = f"Waste scenario saves €{net_savings:.2f} compared to fresh material"
            else:
                explanations['financial'] = "Fresh material is currently more cost-effective"
        
        # Logistics explanation
        if 'error' not in logistics_analysis:
            distance = logistics_analysis.get('distance_km', 0)
            transport_cost = logistics_analysis.get('transport_cost', 0)
            explanations['logistics'] = f"Transport distance: {distance:.1f}km, cost: €{transport_cost:.2f}"
        
        # Carbon explanation
        if 'error' not in carbon_analysis:
            net_savings = carbon_analysis.get('net_carbon_savings_kg', 0)
            if net_savings > 0:
                explanations['carbon'] = f"Net carbon savings: {net_savings:.1f} kg CO2"
            else:
                explanations['carbon'] = "Transport emissions exceed material savings"
        
        # Waste explanation
        if 'error' not in waste_analysis:
            efficiency = waste_analysis.get('waste_management_efficiency', 0)
            explanations['waste'] = f"Waste management efficiency: {efficiency:.1%}"
        
        return explanations

    def _generate_recommendations(self, readiness_assessment: Dict, financial_analysis: Dict,
                                logistics_analysis: Dict, carbon_analysis: Dict, 
                                waste_analysis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Readiness recommendations
        if 'error' not in readiness_assessment:
            recommendations.extend(readiness_assessment.get('recommendations', []))
        
        # Financial recommendations
        if 'error' not in financial_analysis:
            scenario_comparison = financial_analysis.get('scenario_comparison', {})
            recommendations.extend(scenario_comparison.get('recommendations', []))
        
        # Logistics recommendations
        if 'error' not in logistics_analysis:
            distance = logistics_analysis.get('distance_km', 0)
            if distance > 500:
                recommendations.append("Consider intermediate storage or consolidation")
        
        # Carbon recommendations
        if 'error' not in carbon_analysis:
            net_savings = carbon_analysis.get('net_carbon_savings_kg', 0)
            if net_savings < 0:
                recommendations.append("Consider alternative transport modes to reduce carbon impact")
        
        return recommendations

    def _create_economic_summary(self, financial_analysis: Dict) -> Dict:
        """Create economic summary"""
        if 'error' in financial_analysis:
            return {'error': 'Financial analysis failed'}
        
        scenario_comparison = financial_analysis.get('scenario_comparison', {})
        
        return {
            'net_savings': scenario_comparison.get('net_savings', 0),
            'savings_percentage': scenario_comparison.get('savings_percentage', 0),
            'payback_period_months': scenario_comparison.get('payback_period_months', 0),
            'roi_percentage': scenario_comparison.get('roi_percentage', 0),
            'buyer_savings': financial_analysis.get('buyer_savings', 0),
            'seller_profit': financial_analysis.get('seller_profit', 0),
            'total_economic_value': financial_analysis.get('total_economic_value', 0)
        }

    def _create_environmental_summary(self, carbon_analysis: Dict, waste_analysis: Dict) -> Dict:
        """Create environmental summary"""
        summary = {}
        
        if 'error' not in carbon_analysis:
            summary['net_carbon_savings_kg'] = carbon_analysis.get('net_carbon_savings_kg', 0)
            summary['carbon_savings_percentage'] = carbon_analysis.get('carbon_savings_percentage', 0)
            summary['carbon_equivalent'] = carbon_analysis.get('carbon_equivalent', {})
        
        if 'error' not in waste_analysis:
            summary['waste_management_efficiency'] = waste_analysis.get('waste_management_efficiency', 0)
            summary['recycling_potential'] = waste_analysis.get('recycling_potential', 0)
        
        return summary

    def _create_risk_summary(self, readiness_assessment: Dict, financial_analysis: Dict, 
                           logistics_analysis: Dict) -> Dict:
        """Create risk summary"""
        risks = []
        
        if 'error' not in readiness_assessment:
            risk_level = readiness_assessment.get('risk_assessment', {}).get('risk_level', 'low')
            if risk_level == 'high':
                risks.append('Material processing risk')
        
        if 'error' not in financial_analysis:
            risk_level = financial_analysis.get('scenario_comparison', {}).get('risk_assessment', {}).get('overall_risk_level', 'low')
            if risk_level == 'high':
                risks.append('Financial risk')
        
        return {
            'risk_factors': risks,
            'overall_risk_level': 'high' if len(risks) >= 2 else 'medium' if len(risks) >= 1 else 'low'
        }

    def _create_error_analysis(self, buyer_data: Dict, seller_data: Dict, match_data: Dict) -> ComprehensiveMatchAnalysis:
        """Create error analysis when comprehensive analysis fails"""
        return ComprehensiveMatchAnalysis(
            match_id=match_data.get('id', str(uuid.uuid4())),
            buyer_id=buyer_data.get('id', ''),
            seller_id=seller_data.get('id', ''),
            material_type=match_data.get('material_type', 'unknown'),
            quantity_ton=match_data.get('quantity', 0),
            readiness_assessment={'error': 'Analysis failed'},
            financial_analysis={'error': 'Analysis failed'},
            logistics_analysis={'error': 'Analysis failed'},
            carbon_analysis={'error': 'Analysis failed'},
            waste_analysis={'error': 'Analysis failed'},
            overall_score=0.0,
            match_quality='Error',
            confidence_level=0.0,
            risk_level='high',
            explanations={'error': 'Comprehensive analysis failed'},
            recommendations=['Manual analysis required'],
            economic_summary={'error': 'Analysis failed'},
            environmental_summary={'error': 'Analysis failed'},
            risk_summary={'error': 'Analysis failed'},
            analysis_date=datetime.now().isoformat(),
            methodology='Error in comprehensive analysis'
        )

# Initialize the analyzer
comprehensive_analyzer = ComprehensiveMatchAnalyzer() 