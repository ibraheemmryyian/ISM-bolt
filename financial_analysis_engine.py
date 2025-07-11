import os
import sys
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from supabase import create_client, Client
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("FinancialAnalysisEngine")

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for financial analysis"""
    waste_disposal_cost: float = 0.0
    transportation_cost: float = 0.0
    processing_cost: float = 0.0
    storage_cost: float = 0.0
    regulatory_fees: float = 0.0
    insurance_cost: float = 0.0
    opportunity_cost: float = 0.0
    carbon_tax: float = 0.0
    total_cost: float = 0.0

@dataclass
class SymbiosisAnalysis:
    """Complete symbiosis financial analysis"""
    company_id: str
    partner_id: str
    material_id: str
    traditional_cost: CostBreakdown
    symbiosis_cost: CostBreakdown
    net_savings: float
    payback_period_months: float
    roi_percentage: float
    carbon_savings_kg: float
    carbon_savings_value: float
    risk_score: float
    confidence_level: float
    recommendations: List[str]

class FinancialAnalysisEngine:
    """
    Comprehensive Financial Analysis Engine for Industrial Symbiosis
    Calculates costs, savings, ROI, and financial impact of waste exchange partnerships
    """
    
    def __init__(self):
        # Market prices for common materials (USD per unit)
        self.material_prices = {
            # Waste materials - disposal costs and potential value
            'metal scrap': {'disposal_cost': 0.15, 'market_value': 0.25, 'unit': 'kg'},
            'plastic waste': {'disposal_cost': 0.10, 'market_value': 0.18, 'unit': 'kg'},
            'wood waste': {'disposal_cost': 0.08, 'market_value': 0.12, 'unit': 'kg'},
            'electronic waste': {'disposal_cost': 0.50, 'market_value': 0.80, 'unit': 'kg'},
            'chemical byproducts': {'disposal_cost': 0.30, 'market_value': 0.45, 'unit': 'kg'},
            'contaminated water': {'disposal_cost': 0.05, 'market_value': 0.08, 'unit': 'liters'},
            'filter media': {'disposal_cost': 0.20, 'market_value': 0.35, 'unit': 'kg'},
            'chemical sludge': {'disposal_cost': 0.25, 'market_value': 0.40, 'unit': 'kg'},
            'solvent waste': {'disposal_cost': 0.35, 'market_value': 0.55, 'unit': 'liters'},
            'organic waste': {'disposal_cost': 0.06, 'market_value': 0.15, 'unit': 'kg'},
            'cooking oil': {'disposal_cost': 0.12, 'market_value': 0.25, 'unit': 'liters'},
            'packaging waste': {'disposal_cost': 0.08, 'market_value': 0.15, 'unit': 'kg'},
            'fabric scraps': {'disposal_cost': 0.10, 'market_value': 0.20, 'unit': 'kg'},
            'dye waste': {'disposal_cost': 0.20, 'market_value': 0.30, 'unit': 'liters'},
            'fiber waste': {'disposal_cost': 0.09, 'market_value': 0.18, 'unit': 'kg'},
            'yarn waste': {'disposal_cost': 0.11, 'market_value': 0.22, 'unit': 'kg'},
            'textile waste': {'disposal_cost': 0.10, 'market_value': 0.20, 'unit': 'kg'},
            'medical waste': {'disposal_cost': 0.40, 'market_value': 0.60, 'unit': 'kg'},
            'contaminated linens': {'disposal_cost': 0.15, 'market_value': 0.25, 'unit': 'pieces'},
            'pharmaceutical waste': {'disposal_cost': 0.50, 'market_value': 0.75, 'unit': 'kg'},
            'medical sharps': {'disposal_cost': 0.30, 'market_value': 0.45, 'unit': 'units'},
            'sterilization byproducts': {'disposal_cost': 0.18, 'market_value': 0.28, 'unit': 'kg'},
            'food waste': {'disposal_cost': 0.07, 'market_value': 0.18, 'unit': 'kg'},
            'wastewater': {'disposal_cost': 0.03, 'market_value': 0.06, 'unit': 'liters'},
            'soiled linens': {'disposal_cost': 0.12, 'market_value': 0.20, 'unit': 'pieces'},
            'animal byproducts': {'disposal_cost': 0.15, 'market_value': 0.25, 'unit': 'kg'},
            
            # Requirements - procurement costs
            'raw materials': {'procurement_cost': 0.80, 'unit': 'kg'},
            'energy': {'procurement_cost': 0.12, 'unit': 'kwh'},
            'water': {'procurement_cost': 0.02, 'unit': 'liters'},
            'packaging materials': {'procurement_cost': 0.25, 'unit': 'kg'},
            'lubricants': {'procurement_cost': 0.45, 'unit': 'liters'},
            'chemical reagents': {'procurement_cost': 0.60, 'unit': 'kg'},
            'catalysts': {'procurement_cost': 0.90, 'unit': 'kg'},
            'purified water': {'procurement_cost': 0.05, 'unit': 'liters'},
            'raw chemicals': {'procurement_cost': 0.70, 'unit': 'kg'},
            'fresh ingredients': {'procurement_cost': 0.35, 'unit': 'kg'},
            'refrigeration': {'procurement_cost': 0.15, 'unit': 'kwh'},
            'cooking oil': {'procurement_cost': 0.40, 'unit': 'liters'},
            'spices and seasonings': {'procurement_cost': 0.55, 'unit': 'kg'},
            'raw fibers': {'procurement_cost': 0.50, 'unit': 'kg'},
            'dyes and chemicals': {'procurement_cost': 0.65, 'unit': 'kg'},
            'medical supplies': {'procurement_cost': 0.85, 'unit': 'units'},
            'pharmaceuticals': {'procurement_cost': 1.20, 'unit': 'units'},
            'medical equipment': {'procurement_cost': 2.50, 'unit': 'units'},
            'sterilization materials': {'procurement_cost': 0.75, 'unit': 'kg'},
            'cleaning supplies': {'procurement_cost': 0.30, 'unit': 'kg'},
            'linens and towels': {'procurement_cost': 0.45, 'unit': 'pieces'}
        }
        
        # Transportation costs per km per ton
        self.transport_costs = {
            'local': 0.15,      # Within 50km
            'regional': 0.12,   # 50-200km
            'national': 0.10,   # 200-1000km
            'international': 0.08  # >1000km
        }
        
        # Processing costs for different material types
        self.processing_costs = {
            'sorting': 0.05,    # per kg
            'cleaning': 0.08,   # per kg
            'crushing': 0.03,   # per kg
            'drying': 0.06,     # per kg
            'compacting': 0.04, # per kg
            'chemical_treatment': 0.15,  # per kg
            'sterilization': 0.20,       # per kg
            'refining': 0.25    # per kg
        }
        
        # Environmental impact factors (CO2 equivalent per kg)
        self.environmental_factors = {
            'landfill_emissions': 0.5,    # kg CO2e per kg waste
            'incineration_emissions': 0.8, # kg CO2e per kg waste
            'recycling_savings': -0.3,     # kg CO2e saved per kg recycled
            'transport_emissions': 0.1     # kg CO2e per km per ton
        }
    
    def calculate_material_financials(self, material: Dict[str, Any], distance_km: float = 50) -> Dict[str, Any]:
        """
        Calculate comprehensive financial analysis for a single material
        """
        material_name = material.get('material_name', material.get('name', '')).lower()
        quantity = material.get('quantity', 0)
        unit = material.get('unit', 'kg')
        
        # Get price data for this material
        price_data = self.material_prices.get(material_name, {})
        
        # Calculate costs and values
        disposal_cost = price_data.get('disposal_cost', 0.10) * quantity
        market_value = price_data.get('market_value', 0.15) * quantity
        procurement_cost = price_data.get('procurement_cost', 0.50) * quantity
        
        # Calculate transportation costs
        transport_cost = self._calculate_transport_cost(quantity, distance_km, unit)
        
        # Calculate processing costs
        processing_cost = self._calculate_processing_cost(material_name, quantity)
        
        # Calculate environmental impact
        environmental_impact = self._calculate_environmental_impact(material_name, quantity, distance_km)
        
        # Calculate net savings/value
        if material.get('type') == 'waste':
            # For waste materials: savings from avoiding disposal + potential revenue
            net_savings = disposal_cost + market_value - transport_cost - processing_cost
            roi_percentage = (net_savings / (transport_cost + processing_cost)) * 100 if (transport_cost + processing_cost) > 0 else 0
        else:
            # For requirements: savings from using waste instead of buying new
            net_savings = procurement_cost - market_value - transport_cost - processing_cost
            roi_percentage = (net_savings / (market_value + transport_cost + processing_cost)) * 100 if (market_value + transport_cost + processing_cost) > 0 else 0
        
        return {
            'material_name': material.get('material_name', material.get('name')),
            'quantity': quantity,
            'unit': unit,
            'type': material.get('type', 'unknown'),
            'disposal_cost': round(disposal_cost, 2),
            'market_value': round(market_value, 2),
            'procurement_cost': round(procurement_cost, 2),
            'transport_cost': round(transport_cost, 2),
            'processing_cost': round(processing_cost, 2),
            'net_savings': round(net_savings, 2),
            'roi_percentage': round(roi_percentage, 2),
            'environmental_impact': environmental_impact,
            'distance_km': distance_km
        }
    
    def calculate_partnership_financials(self, company_a: Dict[str, Any], company_b: Dict[str, Any], 
                                       materials_exchange: List[Dict[str, Any]], 
                                       distance_km: float = 50) -> Dict[str, Any]:
        """
        Calculate comprehensive financial analysis for a partnership between two companies
        """
        total_analysis = {
            'partnership_summary': {
                'company_a': company_a.get('name', 'Unknown'),
                'company_b': company_b.get('name', 'Unknown'),
                'distance_km': distance_km,
                'materials_count': len(materials_exchange)
            },
            'financial_summary': {
                'total_savings': 0,
                'total_investment': 0,
                'total_roi': 0,
                'annual_savings': 0,
                'payback_period_months': 0
            },
            'environmental_summary': {
                'co2_reduction_kg': 0,
                'landfill_diversion_kg': 0,
                'water_savings_liters': 0
            },
            'materials_analysis': []
        }
        
        # Analyze each material exchange
        for exchange in materials_exchange:
            material_analysis = self.calculate_material_financials(exchange, distance_km)
            total_analysis['materials_analysis'].append(material_analysis)
            
            # Accumulate totals
            total_analysis['financial_summary']['total_savings'] += material_analysis['net_savings']
            total_analysis['financial_summary']['total_investment'] += (
                material_analysis['transport_cost'] + material_analysis['processing_cost']
            )
            total_analysis['environmental_summary']['co2_reduction_kg'] += material_analysis['environmental_impact']['co2_reduction']
            total_analysis['environmental_summary']['landfill_diversion_kg'] += material_analysis['quantity']
        
        # Calculate overall ROI and payback period
        if total_analysis['financial_summary']['total_investment'] > 0:
            total_analysis['financial_summary']['total_roi'] = (
                total_analysis['financial_summary']['total_savings'] / 
                total_analysis['financial_summary']['total_investment']
            ) * 100
        
        # Calculate annual savings (assuming monthly exchange)
        total_analysis['financial_summary']['annual_savings'] = total_analysis['financial_summary']['total_savings'] * 12
        
        # Calculate payback period
        if total_analysis['financial_summary']['annual_savings'] > 0:
            total_analysis['financial_summary']['payback_period_months'] = (
                total_analysis['financial_summary']['total_investment'] / 
                total_analysis['financial_summary']['annual_savings']
            ) * 12
        
        # Round all financial values
        for key in ['total_savings', 'total_investment', 'total_roi', 'annual_savings', 'payback_period_months']:
            total_analysis['financial_summary'][key] = round(total_analysis['financial_summary'][key], 2)
        
        for key in ['co2_reduction_kg', 'landfill_diversion_kg', 'water_savings_liters']:
            total_analysis['environmental_summary'][key] = round(total_analysis['environmental_summary'][key], 2)
        
        return total_analysis
    
    def calculate_portfolio_financials(self, companies: List[Dict[str, Any]], 
                                     partnerships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive financial analysis for the entire portfolio
        """
        portfolio_analysis = {
            'portfolio_summary': {
                'total_companies': len(companies),
                'total_partnerships': len(partnerships),
                'total_materials': 0,
                'total_volume_kg': 0
            },
            'financial_summary': {
                'total_annual_savings': 0,
                'total_investment_required': 0,
                'average_roi': 0,
                'total_market_value': 0,
                'total_disposal_cost_savings': 0
            },
            'environmental_summary': {
                'total_co2_reduction_kg': 0,
                'total_landfill_diversion_kg': 0,
                'total_water_savings_liters': 0
            },
            'partnership_rankings': [],
            'risk_assessment': {
                'high_risk_partnerships': 0,
                'medium_risk_partnerships': 0,
                'low_risk_partnerships': 0
            }
        }
        
        # Analyze each partnership
        partnership_analyses = []
        for partnership in partnerships:
            analysis = self.calculate_partnership_financials(
                partnership.get('company_a', {}),
                partnership.get('company_b', {}),
                partnership.get('materials_exchange', []),
                partnership.get('distance_km', 50)
            )
            partnership_analyses.append(analysis)
            
            # Accumulate portfolio totals
            portfolio_analysis['financial_summary']['total_annual_savings'] += analysis['financial_summary']['annual_savings']
            portfolio_analysis['financial_summary']['total_investment_required'] += analysis['financial_summary']['total_investment']
            portfolio_analysis['environmental_summary']['total_co2_reduction_kg'] += analysis['environmental_summary']['co2_reduction_kg']
            portfolio_analysis['environmental_summary']['total_landfill_diversion_kg'] += analysis['environmental_summary']['landfill_diversion_kg']
            
            # Count materials
            portfolio_analysis['portfolio_summary']['total_materials'] += len(analysis['materials_analysis'])
            for material in analysis['materials_analysis']:
                if material['unit'] == 'kg':
                    portfolio_analysis['portfolio_summary']['total_volume_kg'] += material['quantity']
                elif material['unit'] == 'liters':
                    portfolio_analysis['portfolio_summary']['total_volume_kg'] += material['quantity'] * 0.001  # Approximate conversion
        
        # Calculate average ROI
        if len(partnership_analyses) > 0:
            total_roi = sum(analysis['financial_summary']['total_roi'] for analysis in partnership_analyses)
            portfolio_analysis['financial_summary']['average_roi'] = round(total_roi / len(partnership_analyses), 2)
        
        # Rank partnerships by ROI
        partnership_rankings = []
        for i, analysis in enumerate(partnership_analyses):
            partnership_rankings.append({
                'rank': i + 1,
                'partnership': f"{analysis['partnership_summary']['company_a']} ↔ {analysis['partnership_summary']['company_b']}",
                'annual_savings': analysis['financial_summary']['annual_savings'],
                'roi': analysis['financial_summary']['total_roi'],
                'payback_months': analysis['financial_summary']['payback_period_months']
            })
        
        # Sort by annual savings
        partnership_rankings.sort(key=lambda x: x['annual_savings'], reverse=True)
        portfolio_analysis['partnership_rankings'] = partnership_rankings
        
        # Risk assessment based on ROI and payback period
        for analysis in partnership_analyses:
            roi = analysis['financial_summary']['total_roi']
            payback = analysis['financial_summary']['payback_period_months']
            
            if roi < 50 or payback > 24:  # High risk
                portfolio_analysis['risk_assessment']['high_risk_partnerships'] += 1
            elif roi < 100 or payback > 12:  # Medium risk
                portfolio_analysis['risk_assessment']['medium_risk_partnerships'] += 1
            else:  # Low risk
                portfolio_analysis['risk_assessment']['low_risk_partnerships'] += 1
        
        # Round all values
        for key in ['total_annual_savings', 'total_investment_required', 'average_roi', 'total_market_value', 'total_disposal_cost_savings']:
            portfolio_analysis['financial_summary'][key] = round(portfolio_analysis['financial_summary'][key], 2)
        
        for key in ['total_co2_reduction_kg', 'total_landfill_diversion_kg', 'total_water_savings_liters']:
            portfolio_analysis['environmental_summary'][key] = round(portfolio_analysis['environmental_summary'][key], 2)
        
        return portfolio_analysis
    
    def _calculate_transport_cost(self, quantity: float, distance_km: float, unit: str) -> float:
        """Calculate transportation cost based on distance and quantity"""
        # Convert to tons for transport calculation
        if unit == 'kg':
            tons = quantity / 1000
        elif unit == 'liters':
            tons = quantity / 1000  # Approximate conversion
        elif unit == 'units':
            tons = quantity * 0.001  # Approximate conversion
        elif unit == 'pieces':
            tons = quantity * 0.002  # Approximate conversion
        else:
            tons = quantity / 1000  # Default conversion
        
        # Determine transport category
        if distance_km <= 50:
            rate = self.transport_costs['local']
        elif distance_km <= 200:
            rate = self.transport_costs['regional']
        elif distance_km <= 1000:
            rate = self.transport_costs['national']
        else:
            rate = self.transport_costs['international']
        
        return tons * distance_km * rate
    
    def _calculate_processing_cost(self, material_name: str, quantity: float) -> float:
        """Calculate processing cost based on material type"""
        processing_cost = 0
        
        # Determine required processing based on material type
        if 'waste' in material_name or 'scrap' in material_name:
            processing_cost += self.processing_costs['sorting'] * quantity
            processing_cost += self.processing_costs['cleaning'] * quantity
        
        if 'chemical' in material_name or 'sludge' in material_name:
            processing_cost += self.processing_costs['chemical_treatment'] * quantity
        
        if 'medical' in material_name or 'contaminated' in material_name:
            processing_cost += self.processing_costs['sterilization'] * quantity
        
        if 'fabric' in material_name or 'textile' in material_name:
            processing_cost += self.processing_costs['cleaning'] * quantity
        
        return processing_cost
    
    def _calculate_environmental_impact(self, material_name: str, quantity: float, distance_km: float) -> Dict[str, float]:
        """Calculate environmental impact of material exchange"""
        # CO2 reduction from avoiding landfill
        landfill_emissions = self.environmental_factors['landfill_emissions'] * quantity
        recycling_savings = self.environmental_factors['recycling_savings'] * quantity
        
        # Transport emissions
        transport_emissions = self.environmental_factors['transport_emissions'] * (quantity / 1000) * distance_km
        
        # Net CO2 reduction
        co2_reduction = landfill_emissions + recycling_savings - transport_emissions
        
        return {
            'co2_reduction': max(0, co2_reduction),
            'landfill_emissions_avoided': landfill_emissions,
            'transport_emissions': transport_emissions,
            'recycling_savings': recycling_savings
        }
    
    def generate_financial_report(self, analysis: Dict[str, Any], report_type: str = 'partnership') -> str:
        """
        Generate a formatted financial report
        """
        if report_type == 'partnership':
            return self._generate_partnership_report(analysis)
        elif report_type == 'portfolio':
            return self._generate_portfolio_report(analysis)
        else:
            return self._generate_material_report(analysis)
    
    def _generate_partnership_report(self, analysis: Dict[str, Any]) -> str:
        """Generate partnership financial report"""
        summary = analysis['partnership_summary']
        financial = analysis['financial_summary']
        environmental = analysis['environmental_summary']
        
        report = f"""
INDUSTRIAL SYMBIOSIS PARTNERSHIP FINANCIAL REPORT
=================================================

Partnership: {summary['company_a']} ↔ {summary['company_b']}
Distance: {summary['distance_km']} km
Materials Exchanged: {summary['materials_count']}

FINANCIAL SUMMARY
-----------------
Total Annual Savings: ${financial['annual_savings']:,.2f}
Total Investment Required: ${financial['total_investment']:,.2f}
ROI: {financial['total_roi']:.1f}%
Payback Period: {financial['payback_period_months']:.1f} months

ENVIRONMENTAL IMPACT
-------------------
CO2 Reduction: {environmental['co2_reduction_kg']:,.0f} kg CO2e/year
Landfill Diversion: {environmental['landfill_diversion_kg']:,.0f} kg/year

MATERIALS ANALYSIS
------------------
"""
        
        for material in analysis['materials_analysis']:
            report += f"""
• {material['material_name']} ({material['quantity']} {material['unit']})
  - Net Savings: ${material['net_savings']:,.2f}
  - ROI: {material['roi_percentage']:.1f}%
  - Transport Cost: ${material['transport_cost']:,.2f}
  - Processing Cost: ${material['processing_cost']:,.2f}
"""
        
        return report
    
    def _generate_portfolio_report(self, analysis: Dict[str, Any]) -> str:
        """Generate portfolio financial report"""
        portfolio = analysis['portfolio_summary']
        financial = analysis['financial_summary']
        environmental = analysis['environmental_summary']
        risk = analysis['risk_assessment']
        
        report = f"""
INDUSTRIAL SYMBIOSIS PORTFOLIO FINANCIAL REPORT
===============================================

PORTFOLIO OVERVIEW
------------------
Total Companies: {portfolio['total_companies']}
Total Partnerships: {portfolio['total_partnerships']}
Total Materials: {portfolio['total_materials']}
Total Volume: {portfolio['total_volume_kg']:,.0f} kg

FINANCIAL SUMMARY
-----------------
Total Annual Savings: ${financial['total_annual_savings']:,.2f}
Total Investment Required: ${financial['total_investment_required']:,.2f}
Average ROI: {financial['average_roi']:.1f}%

ENVIRONMENTAL IMPACT
-------------------
Total CO2 Reduction: {environmental['total_co2_reduction_kg']:,.0f} kg CO2e/year
Total Landfill Diversion: {environmental['total_landfill_diversion_kg']:,.0f} kg/year

RISK ASSESSMENT
---------------
Low Risk Partnerships: {risk['low_risk_partnerships']}
Medium Risk Partnerships: {risk['medium_risk_partnerships']}
High Risk Partnerships: {risk['high_risk_partnerships']}

TOP PARTNERSHIPS BY ANNUAL SAVINGS
----------------------------------
"""
        
        for i, partnership in enumerate(analysis['partnership_rankings'][:10]):
            report += f"""
{i+1}. {partnership['partnership']}
    Annual Savings: ${partnership['annual_savings']:,.2f}
    ROI: {partnership['roi']:.1f}%
    Payback: {partnership['payback_months']:.1f} months
"""
        
        return report

# Convenience functions
def analyze_material_financials(material: Dict[str, Any], distance_km: float = 50) -> Dict[str, Any]:
    """Convenience function for material financial analysis"""
    engine = FinancialAnalysisEngine()
    return engine.calculate_material_financials(material, distance_km)

def analyze_partnership_financials(company_a: Dict[str, Any], company_b: Dict[str, Any], 
                                 materials_exchange: List[Dict[str, Any]], 
                                 distance_km: float = 50) -> Dict[str, Any]:
    """Convenience function for partnership financial analysis"""
    engine = FinancialAnalysisEngine()
    return engine.calculate_partnership_financials(company_a, company_b, materials_exchange, distance_km)

def analyze_portfolio_financials(companies: List[Dict[str, Any]], 
                               partnerships: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convenience function for portfolio financial analysis"""
    engine = FinancialAnalysisEngine()
    return engine.calculate_portfolio_financials(companies, partnerships)

if __name__ == "__main__":
    # Example usage
    engine = FinancialAnalysisEngine()
    
    # Example material
    material = {
        'material_name': 'Metal Scrap',
        'quantity': 1000,
        'unit': 'kg',
        'type': 'waste'
    }
    
    analysis = engine.calculate_material_financials(material, 50)
    print(json.dumps(analysis, indent=2)) 