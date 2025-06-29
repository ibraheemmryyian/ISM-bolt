import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a scenario"""
    transport_cost: float
    handling_cost: float
    customs_cost: float
    insurance_cost: float
    storage_cost: float
    processing_cost: float
    equipment_cost: float
    carbon_tax: float
    total_cost: float
    cost_per_ton: float

@dataclass
class ScenarioComparison:
    """Comparison between waste and fresh material scenarios"""
    waste_scenario: Dict
    fresh_scenario: Dict
    net_savings: float
    savings_percentage: float
    payback_period_months: float
    roi_percentage: float
    break_even_quantity: float
    risk_assessment: Dict
    recommendations: List[str]

@dataclass
class FinancialAnalysis:
    """Comprehensive financial analysis for a match"""
    match_id: str
    buyer_id: str
    seller_id: str
    material_type: str
    quantity_ton: float
    scenario_comparison: ScenarioComparison
    buyer_savings: float
    seller_profit: float
    total_economic_value: float
    carbon_savings_value: float
    risk_adjusted_roi: float
    confidence_level: float
    analysis_date: str
    methodology: str

class FinancialAnalysisEngine:
    def __init__(self):
        # Market price database (EUR per ton)
        self.market_prices = {
            'cotton': 2500,
            'polyester': 1800,
            'steel': 800,
            'aluminum': 2500,
            'plastic': 1200,
            'paper': 600,
            'glass': 400,
            'organic_waste': 50,
            'metal_scrap': 300,
            'plastic_waste': 200,
            'paper_waste': 150,
            'textile_waste': 100
        }
        
        # Cost factors
        self.cost_factors = {
            'transport_per_km_ton': 0.15,  # EUR per km per ton
            'handling_per_ton': 25,        # EUR per ton
            'customs_rate': 0.05,          # 5% of cargo value
            'insurance_rate': 0.02,        # 2% of cargo value
            'storage_per_day_ton': 2,      # EUR per day per ton
            'carbon_tax_per_ton_co2': 50,  # EUR per ton CO2
            'processing_markup': 0.15,     # 15% markup for processing
            'equipment_depreciation_years': 10,
            'maintenance_rate': 0.05,      # 5% of equipment cost per year
            'energy_cost_per_kwh': 0.15    # EUR per kWh
        }
        
        # Risk factors for different scenarios
        self.risk_factors = {
            'waste_quality_uncertainty': 0.1,
            'supply_reliability': 0.15,
            'processing_complexity': 0.2,
            'market_volatility': 0.1,
            'regulatory_risk': 0.05
        }

    def analyze_match_financials(self, match_data: Dict, buyer_data: Dict, 
                               seller_data: Dict, logistics_data: Dict,
                               refinement_data: Optional[Dict] = None) -> FinancialAnalysis:
        """Comprehensive financial analysis for a match"""
        try:
            material_type = match_data.get('material_type', '').lower()
            quantity = match_data.get('quantity', 1.0)
            
            # Calculate waste scenario (using seller's waste)
            waste_scenario = self._calculate_waste_scenario(
                match_data, buyer_data, seller_data, logistics_data, refinement_data
            )
            
            # Calculate fresh scenario (buying new material)
            fresh_scenario = self._calculate_fresh_scenario(
                match_data, buyer_data, logistics_data
            )
            
            # Compare scenarios
            scenario_comparison = self._compare_scenarios(waste_scenario, fresh_scenario, quantity)
            
            # Calculate buyer and seller benefits
            buyer_savings = scenario_comparison.net_savings
            seller_profit = self._calculate_seller_profit(seller_data, match_data, logistics_data)
            
            # Calculate total economic value
            total_economic_value = buyer_savings + seller_profit
            
            # Calculate carbon savings value
            carbon_savings_value = self._calculate_carbon_savings_value(match_data)
            
            # Calculate risk-adjusted ROI
            risk_adjusted_roi = self._calculate_risk_adjusted_roi(scenario_comparison, match_data)
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(match_data, logistics_data, refinement_data)
            
            return FinancialAnalysis(
                match_id=match_data.get('id', str(uuid.uuid4())),
                buyer_id=buyer_data.get('id', ''),
                seller_id=seller_data.get('id', ''),
                material_type=material_type,
                quantity_ton=quantity,
                scenario_comparison=scenario_comparison,
                buyer_savings=buyer_savings,
                seller_profit=seller_profit,
                total_economic_value=total_economic_value,
                carbon_savings_value=carbon_savings_value,
                risk_adjusted_roi=risk_adjusted_roi,
                confidence_level=confidence_level,
                analysis_date=datetime.now().isoformat(),
                methodology='Comprehensive cost-benefit analysis with risk adjustment'
            )
            
        except Exception as e:
            logger.error(f"Error in financial analysis: {e}")
            return self._create_error_analysis(match_data)

    def _calculate_waste_scenario(self, match_data: Dict, buyer_data: Dict, 
                                seller_data: Dict, logistics_data: Dict,
                                refinement_data: Optional[Dict]) -> Dict:
        """Calculate costs for using waste material scenario"""
        material_type = match_data.get('material_type', '').lower()
        quantity = match_data.get('quantity', 1.0)
        distance = logistics_data.get('distance_km', 100)
        
        # Get market price for comparison
        market_price = self.market_prices.get(material_type, 500)
        
        # Calculate transport cost
        transport_cost = distance * quantity * self.cost_factors['transport_per_km_ton']
        
        # Calculate handling cost
        handling_cost = quantity * self.cost_factors['handling_per_ton']
        
        # Calculate customs cost (if international)
        cargo_value = quantity * market_price * 0.7  # Waste typically 70% of market value
        customs_cost = cargo_value * self.cost_factors['customs_rate'] if logistics_data.get('international', False) else 0
        
        # Calculate insurance cost
        insurance_cost = cargo_value * self.cost_factors['insurance_rate']
        
        # Calculate storage cost
        transit_days = logistics_data.get('transit_days', 3)
        storage_cost = quantity * transit_days * self.cost_factors['storage_per_day_ton']
        
        # Calculate processing cost (if refinement needed)
        processing_cost = 0
        if refinement_data and refinement_data.get('refinement_required', False):
            processing_cost = quantity * refinement_data.get('cost_per_ton', 200)
            processing_cost *= (1 + self.cost_factors['processing_markup'])
        
        # Calculate equipment cost (amortized)
        equipment_cost = 0
        if refinement_data and refinement_data.get('equipment_cost', 0) > 0:
            annual_equipment_cost = refinement_data['equipment_cost'] / self.cost_factors['equipment_depreciation_years']
            annual_equipment_cost += refinement_data['equipment_cost'] * self.cost_factors['maintenance_rate']
            equipment_cost = annual_equipment_cost / 12  # Monthly cost
        
        # Calculate carbon tax
        carbon_emissions = logistics_data.get('carbon_kg', 0)
        carbon_tax = carbon_emissions * self.cost_factors['carbon_tax_per_ton_co2'] / 1000
        
        # Calculate total cost
        total_cost = (transport_cost + handling_cost + customs_cost + insurance_cost + 
                     storage_cost + processing_cost + equipment_cost + carbon_tax)
        
        # Waste material cost (what buyer pays to seller)
        waste_material_cost = quantity * match_data.get('waste_price_per_ton', 100)
        
        # Total cost including waste material
        total_cost_with_material = total_cost + waste_material_cost
        
        return {
            'scenario': 'waste',
            'material_cost': waste_material_cost,
            'transport_cost': transport_cost,
            'handling_cost': handling_cost,
            'customs_cost': customs_cost,
            'insurance_cost': insurance_cost,
            'storage_cost': storage_cost,
            'processing_cost': processing_cost,
            'equipment_cost': equipment_cost,
            'carbon_tax': carbon_tax,
            'total_cost': total_cost_with_material,
            'cost_per_ton': total_cost_with_material / quantity,
            'quality_factor': match_data.get('quality_factor', 0.8),
            'supply_reliability': seller_data.get('reliability_score', 0.7)
        }

    def _calculate_fresh_scenario(self, match_data: Dict, buyer_data: Dict, 
                                logistics_data: Dict) -> Dict:
        """Calculate costs for buying fresh material scenario"""
        material_type = match_data.get('material_type', '').lower()
        quantity = match_data.get('quantity', 1.0)
        distance = logistics_data.get('distance_km', 100)
        
        # Get market price
        market_price = self.market_prices.get(material_type, 500)
        
        # Calculate transport cost (typically higher for fresh materials)
        transport_cost = distance * quantity * self.cost_factors['transport_per_km_ton'] * 1.2
        
        # Calculate handling cost
        handling_cost = quantity * self.cost_factors['handling_per_ton']
        
        # Calculate customs cost (if international)
        cargo_value = quantity * market_price
        customs_cost = cargo_value * self.cost_factors['customs_rate'] if logistics_data.get('international', False) else 0
        
        # Calculate insurance cost
        insurance_cost = cargo_value * self.cost_factors['insurance_rate']
        
        # Calculate storage cost
        transit_days = logistics_data.get('transit_days', 3)
        storage_cost = quantity * transit_days * self.cost_factors['storage_per_day_ton']
        
        # No processing cost for fresh materials
        processing_cost = 0
        equipment_cost = 0
        
        # Calculate carbon tax (typically higher for fresh materials)
        carbon_emissions = logistics_data.get('carbon_kg', 0) * 1.5  # Higher emissions for fresh
        carbon_tax = carbon_emissions * self.cost_factors['carbon_tax_per_ton_co2'] / 1000
        
        # Calculate total cost
        total_cost = (transport_cost + handling_cost + customs_cost + insurance_cost + 
                     storage_cost + processing_cost + equipment_cost + carbon_tax)
        
        # Fresh material cost
        fresh_material_cost = quantity * market_price
        
        # Total cost including fresh material
        total_cost_with_material = total_cost + fresh_material_cost
        
        return {
            'scenario': 'fresh',
            'material_cost': fresh_material_cost,
            'transport_cost': transport_cost,
            'handling_cost': handling_cost,
            'customs_cost': customs_cost,
            'insurance_cost': insurance_cost,
            'storage_cost': storage_cost,
            'processing_cost': processing_cost,
            'equipment_cost': equipment_cost,
            'carbon_tax': carbon_tax,
            'total_cost': total_cost_with_material,
            'cost_per_ton': total_cost_with_material / quantity,
            'quality_factor': 1.0,  # Fresh materials have perfect quality
            'supply_reliability': 0.95  # High reliability for fresh materials
        }

    def _compare_scenarios(self, waste_scenario: Dict, fresh_scenario: Dict, 
                          quantity: float) -> ScenarioComparison:
        """Compare waste vs fresh scenarios"""
        waste_cost = waste_scenario['total_cost']
        fresh_cost = fresh_scenario['total_cost']
        
        net_savings = fresh_cost - waste_cost
        savings_percentage = (net_savings / fresh_cost) * 100 if fresh_cost > 0 else 0
        
        # Calculate payback period (if equipment investment needed)
        equipment_investment = waste_scenario.get('equipment_cost', 0) * 12  # Annual equipment cost
        if equipment_investment > 0 and net_savings > 0:
            payback_period = equipment_investment / net_savings
        else:
            payback_period = 0
        
        # Calculate ROI
        total_investment = waste_scenario.get('equipment_cost', 0) * 12
        roi_percentage = (net_savings / total_investment) * 100 if total_investment > 0 else float('inf')
        
        # Calculate break-even quantity
        cost_difference_per_ton = fresh_scenario['cost_per_ton'] - waste_scenario['cost_per_ton']
        if cost_difference_per_ton > 0:
            break_even_quantity = total_investment / cost_difference_per_ton
        else:
            break_even_quantity = float('inf')
        
        # Risk assessment
        risk_assessment = self._assess_scenario_risks(waste_scenario, fresh_scenario)
        
        # Generate recommendations
        recommendations = self._generate_financial_recommendations(
            net_savings, payback_period, roi_percentage, risk_assessment
        )
        
        return ScenarioComparison(
            waste_scenario=waste_scenario,
            fresh_scenario=fresh_scenario,
            net_savings=net_savings,
            savings_percentage=savings_percentage,
            payback_period_months=payback_period,
            roi_percentage=roi_percentage,
            break_even_quantity=break_even_quantity,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )

    def _calculate_seller_profit(self, seller_data: Dict, match_data: Dict, 
                               logistics_data: Dict) -> float:
        """Calculate seller's profit from the transaction"""
        quantity = match_data.get('quantity', 1.0)
        waste_price = match_data.get('waste_price_per_ton', 100)
        
        # Seller's revenue
        revenue = quantity * waste_price
        
        # Seller's costs (disposal cost savings + transport to buyer)
        disposal_cost_savings = quantity * seller_data.get('disposal_cost_per_ton', 50)
        transport_cost = logistics_data.get('distance_km', 100) * quantity * self.cost_factors['transport_per_km_ton'] * 0.5  # Seller pays half
        
        # Seller's profit
        profit = revenue + disposal_cost_savings - transport_cost
        
        return max(0, profit)  # Profit cannot be negative

    def _calculate_carbon_savings_value(self, match_data: Dict) -> float:
        """Calculate monetary value of carbon savings"""
        carbon_savings_kg = match_data.get('carbon_savings_kg', 0)
        carbon_price_per_ton = 50  # EUR per ton CO2
        
        carbon_savings_value = (carbon_savings_kg / 1000) * carbon_price_per_ton
        
        return carbon_savings_value

    def _calculate_risk_adjusted_roi(self, scenario_comparison: ScenarioComparison, 
                                   match_data: Dict) -> float:
        """Calculate risk-adjusted ROI"""
        base_roi = scenario_comparison.roi_percentage
        
        # Risk factors
        quality_risk = 1 - match_data.get('quality_factor', 0.8)
        supply_risk = 1 - match_data.get('supply_reliability', 0.7)
        market_risk = self.risk_factors['market_volatility']
        
        total_risk = quality_risk + supply_risk + market_risk
        
        # Risk adjustment
        risk_adjusted_roi = base_roi * (1 - total_risk)
        
        return max(0, risk_adjusted_roi)

    def _calculate_confidence_level(self, match_data: Dict, logistics_data: Dict, 
                                  refinement_data: Optional[Dict]) -> float:
        """Calculate confidence level in the financial analysis"""
        confidence_factors = []
        
        # Data quality factors
        if match_data.get('quantity') and match_data.get('material_type'):
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Logistics data quality
        if logistics_data.get('distance_km') and logistics_data.get('transit_days'):
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Refinement data quality
        if refinement_data and refinement_data.get('refinement_required'):
            if refinement_data.get('cost_per_ton'):
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.9)  # No refinement needed
        
        # Market price availability
        material_type = match_data.get('material_type', '').lower()
        if material_type in self.market_prices:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Calculate average confidence
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        return min(1.0, confidence)

    def _assess_scenario_risks(self, waste_scenario: Dict, fresh_scenario: Dict) -> Dict:
        """Assess risks for both scenarios"""
        waste_risks = []
        fresh_risks = []
        
        # Waste scenario risks
        if waste_scenario['quality_factor'] < 0.9:
            waste_risks.append('quality_uncertainty')
        
        if waste_scenario['supply_reliability'] < 0.8:
            waste_risks.append('supply_reliability')
        
        if waste_scenario['processing_cost'] > 0:
            waste_risks.append('processing_complexity')
        
        # Fresh scenario risks
        if fresh_scenario['carbon_tax'] > waste_scenario['carbon_tax'] * 1.5:
            fresh_risks.append('environmental_regulations')
        
        if fresh_scenario['material_cost'] > waste_scenario['material_cost'] * 3:
            fresh_risks.append('price_volatility')
        
        # Calculate risk scores
        waste_risk_score = len(waste_risks) * 0.2
        fresh_risk_score = len(fresh_risks) * 0.15
        
        return {
            'waste_scenario_risks': waste_risks,
            'fresh_scenario_risks': fresh_risks,
            'waste_risk_score': min(1.0, waste_risk_score),
            'fresh_risk_score': min(1.0, fresh_risk_score),
            'overall_risk_level': 'low' if max(waste_risk_score, fresh_risk_score) < 0.3 else 'medium' if max(waste_risk_score, fresh_risk_score) < 0.6 else 'high'
        }

    def _generate_financial_recommendations(self, net_savings: float, payback_period: float, 
                                          roi_percentage: float, risk_assessment: Dict) -> List[str]:
        """Generate financial recommendations"""
        recommendations = []
        
        if net_savings > 0:
            recommendations.append(f"Waste scenario saves €{net_savings:.2f} compared to fresh material")
            
            if payback_period < 12:
                recommendations.append(f"Equipment investment has fast payback: {payback_period:.1f} months")
            elif payback_period < 36:
                recommendations.append(f"Equipment investment has reasonable payback: {payback_period:.1f} months")
            else:
                recommendations.append("Consider equipment rental or alternative solutions")
            
            if roi_percentage > 20:
                recommendations.append(f"Excellent ROI: {roi_percentage:.1f}%")
            elif roi_percentage > 10:
                recommendations.append(f"Good ROI: {roi_percentage:.1f}%")
            else:
                recommendations.append(f"Moderate ROI: {roi_percentage:.1f}%")
        else:
            recommendations.append("Fresh material is currently more cost-effective")
        
        # Risk-based recommendations
        if risk_assessment['overall_risk_level'] == 'high':
            recommendations.append("High risk - consider risk mitigation strategies")
        elif risk_assessment['overall_risk_level'] == 'medium':
            recommendations.append("Moderate risk - monitor key risk factors")
        else:
            recommendations.append("Low risk - proceed with confidence")
        
        return recommendations

    def _create_error_analysis(self, match_data: Dict) -> FinancialAnalysis:
        """Create error analysis when financial analysis fails"""
        return FinancialAnalysis(
            match_id=match_data.get('id', str(uuid.uuid4())),
            buyer_id='',
            seller_id='',
            material_type=match_data.get('material_type', 'unknown'),
            quantity_ton=match_data.get('quantity', 0),
            scenario_comparison=ScenarioComparison(
                waste_scenario={},
                fresh_scenario={},
                net_savings=0,
                savings_percentage=0,
                payback_period_months=0,
                roi_percentage=0,
                break_even_quantity=0,
                risk_assessment={'overall_risk_level': 'high'},
                recommendations=['Error in financial analysis - manual review required']
            ),
            buyer_savings=0,
            seller_profit=0,
            total_economic_value=0,
            carbon_savings_value=0,
            risk_adjusted_roi=0,
            confidence_level=0,
            analysis_date=datetime.now().isoformat(),
            methodology='Error in analysis'
        )

# Initialize the engine
financial_engine = FinancialAnalysisEngine() 