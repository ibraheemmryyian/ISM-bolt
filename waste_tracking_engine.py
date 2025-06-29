import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import uuid
from datetime import datetime, timedelta

@dataclass
class WasteStream:
    """Waste stream tracking"""
    material: str
    quantity: float
    unit: str
    waste_type: str  # solid, liquid, hazardous, recyclable
    disposal_method: str
    cost_per_ton: float
    environmental_impact: float

@dataclass
class RecyclingProcess:
    """Recycling process data"""
    material: str
    recycling_rate: float
    energy_savings: float
    cost_savings: float
    quality_grade: str

class WasteTrackingEngine:
    def __init__(self):
        # Industry-specific waste generation rates (kg per employee per month)
        self.waste_generation_rates = {
            'manufacturing': {
                'solid_waste': 150,
                'liquid_waste': 200,
                'hazardous_waste': 25,
                'recyclable_waste': 80
            },
            'textiles': {
                'solid_waste': 120,
                'liquid_waste': 300,
                'hazardous_waste': 15,
                'recyclable_waste': 60
            },
            'food_beverage': {
                'solid_waste': 200,
                'liquid_waste': 500,
                'hazardous_waste': 10,
                'recyclable_waste': 100
            },
            'chemicals': {
                'solid_waste': 100,
                'liquid_waste': 150,
                'hazardous_waste': 50,
                'recyclable_waste': 40
            }
        }
        
        # Material-specific waste characteristics
        self.material_waste_profiles = {
            'steel': {
                'scrap_rate': 0.08,  # 8% becomes scrap
                'recyclability': 0.95,  # 95% recyclable
                'hazardous_content': 0.02,  # 2% hazardous
                'disposal_cost': 150  # USD per ton
            },
            'aluminum': {
                'scrap_rate': 0.12,
                'recyclability': 0.98,
                'hazardous_content': 0.01,
                'disposal_cost': 200
            },
            'plastic': {
                'scrap_rate': 0.15,
                'recyclability': 0.70,
                'hazardous_content': 0.05,
                'disposal_cost': 100
            },
            'paper': {
                'scrap_rate': 0.20,
                'recyclability': 0.85,
                'hazardous_content': 0.01,
                'disposal_cost': 80
            },
            'textiles': {
                'scrap_rate': 0.25,
                'recyclability': 0.60,
                'hazardous_content': 0.03,
                'disposal_cost': 120
            },
            'chemicals': {
                'scrap_rate': 0.05,
                'recyclability': 0.40,
                'hazardous_content': 0.30,
                'disposal_cost': 500
            }
        }
        
        # Waste disposal costs and environmental impacts
        self.disposal_methods = {
            'landfill': {
                'cost_per_ton': 50,
                'environmental_impact': 1.0,  # baseline
                'carbon_emissions': 0.5  # tons CO2 per ton waste
            },
            'incineration': {
                'cost_per_ton': 80,
                'environmental_impact': 0.7,
                'carbon_emissions': 0.3
            },
            'recycling': {
                'cost_per_ton': -30,  # negative cost = savings
                'environmental_impact': 0.2,
                'carbon_emissions': 0.1
            },
            'composting': {
                'cost_per_ton': 20,
                'environmental_impact': 0.3,
                'carbon_emissions': 0.2
            }
        }

    def calculate_company_waste_profile(self, company_data: Dict) -> Dict:
        """Calculate comprehensive waste profile for a company"""
        try:
            industry = company_data.get('industry', '').lower()
            materials = company_data.get('materials', [])
            employee_count = company_data.get('employee_count', 0)
            processes = company_data.get('processes', '')
            
            # Calculate waste generation by type
            waste_by_type = self._calculate_waste_by_type(industry, employee_count)
            
            # Calculate material-specific waste
            material_waste = self._calculate_material_waste(materials, processes)
            
            # Calculate waste costs
            waste_costs = self._calculate_waste_costs(waste_by_type, material_waste)
            
            # Calculate recycling potential
            recycling_potential = self._calculate_recycling_potential(materials, waste_by_type)
            
            # Calculate environmental impact
            environmental_impact = self._calculate_environmental_impact(waste_by_type, material_waste)
            
            # Generate waste reduction recommendations
            recommendations = self._generate_waste_recommendations(company_data, waste_by_type, material_waste)
            
            return {
                'total_waste_generated': round(sum(waste_by_type.values()), 2),
                'waste_by_type': {k: round(v, 2) for k, v in waste_by_type.items()},
                'material_waste': material_waste,
                'waste_costs': waste_costs,
                'recycling_potential': recycling_potential,
                'environmental_impact': environmental_impact,
                'recommendations': recommendations,
                'calculation_date': datetime.now().isoformat(),
                'methodology': 'Industry Waste Generation Rates + Material Flow Analysis'
            }
            
        except Exception as e:
            return {'error': f'Waste calculation failed: {str(e)}'}

    def _calculate_waste_by_type(self, industry: str, employee_count: int) -> Dict[str, float]:
        """Calculate waste generation by type based on industry and employee count"""
        industry_rates = self.waste_generation_rates.get(industry, self.waste_generation_rates['manufacturing'])
        
        waste_by_type = {}
        for waste_type, rate_per_employee in industry_rates.items():
            total_waste = employee_count * rate_per_employee
            waste_by_type[waste_type] = total_waste
        
        return waste_by_type

    def _calculate_material_waste(self, materials: List[str], processes: str) -> Dict[str, Dict]:
        """Calculate waste generation from specific materials"""
        material_waste = {}
        
        for material in materials:
            material_lower = material.lower()
            profile = self.material_waste_profiles.get(material_lower, self.material_waste_profiles['plastic'])
            
            # Estimate material usage (kg per month)
            base_usage = 1000  # kg per month per material
            
            # Calculate waste quantities
            scrap_quantity = base_usage * profile['scrap_rate']
            recyclable_quantity = scrap_quantity * profile['recyclability']
            hazardous_quantity = scrap_quantity * profile['hazardous_content']
            
            material_waste[material] = {
                'total_waste': round(scrap_quantity, 2),
                'recyclable_waste': round(recyclable_quantity, 2),
                'hazardous_waste': round(hazardous_quantity, 2),
                'disposal_cost': round(scrap_quantity * profile['disposal_cost'] / 1000, 2),  # Convert to USD
                'recyclability_rate': profile['recyclability'],
                'scrap_rate': profile['scrap_rate']
            }
        
        return material_waste

    def _calculate_waste_costs(self, waste_by_type: Dict[str, float], material_waste: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate total waste management costs"""
        total_costs = {
            'disposal_costs': 0.0,
            'recycling_savings': 0.0,
            'hazardous_waste_costs': 0.0,
            'total_net_cost': 0.0
        }
        
        # Calculate disposal costs for general waste
        for waste_type, quantity in waste_by_type.items():
            if waste_type == 'recyclable_waste':
                # Recycling saves money
                savings = quantity * abs(self.disposal_methods['recycling']['cost_per_ton']) / 1000
                total_costs['recycling_savings'] += savings
            elif waste_type == 'hazardous_waste':
                # Hazardous waste costs more
                cost = quantity * self.disposal_methods['landfill']['cost_per_ton'] * 3 / 1000  # 3x normal cost
                total_costs['hazardous_waste_costs'] += cost
            else:
                # Regular disposal costs
                cost = quantity * self.disposal_methods['landfill']['cost_per_ton'] / 1000
                total_costs['disposal_costs'] += cost
        
        # Add material-specific disposal costs
        for material, waste_data in material_waste.items():
            total_costs['disposal_costs'] += waste_data['disposal_cost']
        
        # Calculate net cost
        total_costs['total_net_cost'] = (
            total_costs['disposal_costs'] + 
            total_costs['hazardous_waste_costs'] - 
            total_costs['recycling_savings']
        )
        
        return {k: round(v, 2) for k, v in total_costs.items()}

    def _calculate_recycling_potential(self, materials: List[str], waste_by_type: Dict[str, float]) -> Dict[str, float]:
        """Calculate recycling potential and savings"""
        total_recyclable = waste_by_type.get('recyclable_waste', 0)
        
        # Calculate material-specific recycling potential
        material_recycling = 0.0
        for material in materials:
            material_lower = material.lower()
            profile = self.material_waste_profiles.get(material_lower, self.material_waste_profiles['plastic'])
            material_recycling += 1000 * profile['scrap_rate'] * profile['recyclability']
        
        total_recycling_potential = total_recyclable + material_recycling
        
        # Calculate potential savings
        recycling_savings = total_recycling_potential * abs(self.disposal_methods['recycling']['cost_per_ton']) / 1000
        
        # Calculate energy savings from recycling
        energy_savings = total_recycling_potential * 0.5  # kWh per kg recycled
        
        return {
            'total_recyclable_waste': round(total_recycling_potential, 2),
            'current_recycling_rate': round((total_recyclable / max(sum(waste_by_type.values()), 1)) * 100, 1),
            'potential_recycling_rate': round((total_recycling_potential / max(sum(waste_by_type.values()), 1)) * 100, 1),
            'recycling_savings': round(recycling_savings, 2),
            'energy_savings': round(energy_savings, 2),
            'carbon_reduction': round(total_recycling_potential * 0.4, 2)  # kg CO2 saved
        }

    def _calculate_environmental_impact(self, waste_by_type: Dict[str, float], material_waste: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate environmental impact of waste"""
        total_waste = sum(waste_by_type.values())
        
        # Calculate carbon emissions from waste disposal
        landfill_emissions = total_waste * self.disposal_methods['landfill']['carbon_emissions'] / 1000
        
        # Calculate potential emissions reduction from recycling
        recyclable_waste = waste_by_type.get('recyclable_waste', 0)
        recycling_emissions = recyclable_waste * self.disposal_methods['recycling']['carbon_emissions'] / 1000
        
        net_emissions = landfill_emissions - recycling_emissions
        
        # Calculate environmental impact score (0-100, lower is better)
        impact_score = min(100, (net_emissions / max(total_waste, 1)) * 1000)
        
        return {
            'total_carbon_emissions': round(net_emissions, 2),
            'landfill_emissions': round(landfill_emissions, 2),
            'recycling_emissions': round(recycling_emissions, 2),
            'environmental_impact_score': round(impact_score, 1),
            'waste_to_landfill': round(total_waste - recyclable_waste, 2)
        }

    def _generate_waste_recommendations(self, company_data: Dict, waste_by_type: Dict[str, float], material_waste: Dict[str, Dict]) -> List[Dict]:
        """Generate specific waste reduction recommendations"""
        recommendations = []
        industry = company_data.get('industry', '').lower()
        
        # High waste generation recommendations
        if waste_by_type.get('solid_waste', 0) > 1000:
            recommendations.append({
                'category': 'Waste Reduction',
                'title': 'Implement Lean Manufacturing and Zero Waste Program',
                'potential_reduction': round(waste_by_type['solid_waste'] * 0.25, 2),
                'cost_savings': round(waste_by_type['solid_waste'] * 0.25 * 50 / 1000, 2),
                'implementation_cost': 75000,
                'payback_period': '2-3 years',
                'priority': 'high'
            })
        
        # Low recycling rate recommendations
        current_recycling = waste_by_type.get('recyclable_waste', 0)
        total_waste = sum(waste_by_type.values())
        recycling_rate = (current_recycling / max(total_waste, 1)) * 100
        
        if recycling_rate < 30:
            recommendations.append({
                'category': 'Recycling',
                'title': 'Implement Comprehensive Recycling Infrastructure',
                'potential_increase': round(total_waste * 0.3 - current_recycling, 2),
                'cost_savings': round((total_waste * 0.3 - current_recycling) * 30 / 1000, 2),
                'implementation_cost': 50000,
                'payback_period': '1-2 years',
                'priority': 'high'
            })
        
        # Hazardous waste recommendations
        if waste_by_type.get('hazardous_waste', 0) > 100:
            recommendations.append({
                'category': 'Hazardous Waste',
                'title': 'Implement Hazardous Waste Minimization Program',
                'potential_reduction': round(waste_by_type['hazardous_waste'] * 0.40, 2),
                'cost_savings': round(waste_by_type['hazardous_waste'] * 0.40 * 150 / 1000, 2),
                'implementation_cost': 100000,
                'payback_period': '3-4 years',
                'priority': 'medium'
            })
        
        # Industry-specific recommendations
        if 'food_beverage' in industry:
            recommendations.append({
                'category': 'Organic Waste',
                'title': 'Implement Food Waste Composting Program',
                'potential_reduction': round(waste_by_type.get('solid_waste', 0) * 0.20, 2),
                'cost_savings': round(waste_by_type.get('solid_waste', 0) * 0.20 * 30 / 1000, 2),
                'implementation_cost': 30000,
                'payback_period': '1 year',
                'priority': 'medium'
            })
        
        return recommendations

    def calculate_waste_reduction_potential(self, company_data: Dict, initiatives: List[Dict]) -> Dict:
        """Calculate waste reduction potential from specific initiatives"""
        try:
            current_profile = self.calculate_company_waste_profile(company_data)
            total_reduction = 0.0
            cost_savings = 0.0
            initiative_reductions = []
            
            for initiative in initiatives:
                reduction, savings = self._calculate_initiative_waste_reduction(initiative, company_data)
                total_reduction += reduction
                cost_savings += savings
                
                initiative_reductions.append({
                    'initiative_id': initiative.get('id'),
                    'title': initiative.get('question'),
                    'waste_reduction': round(reduction, 2),
                    'cost_savings': round(savings, 2),
                    'percentage_reduction': round((reduction / current_profile['total_waste_generated']) * 100, 1)
                })
            
            new_waste_total = current_profile['total_waste_generated'] - total_reduction
            
            return {
                'current_waste': current_profile['total_waste_generated'],
                'potential_reduction': round(total_reduction, 2),
                'new_waste_total': round(new_waste_total, 2),
                'cost_savings': round(cost_savings, 2),
                'reduction_percentage': round((total_reduction / current_profile['total_waste_generated']) * 100, 1),
                'initiative_breakdown': initiative_reductions
            }
            
        except Exception as e:
            return {'error': f'Waste reduction calculation failed: {str(e)}'}

    def _calculate_initiative_waste_reduction(self, initiative: Dict, company_data: Dict) -> Tuple[float, float]:
        """Calculate waste reduction and cost savings for a specific initiative"""
        initiative_type = initiative.get('category', '').lower()
        company_profile = self.calculate_company_waste_profile(company_data)
        
        reduction_factors = {
            'waste reduction': 0.25,  # 25% reduction in solid waste
            'recycling': 0.30,        # 30% increase in recycling
            'hazardous waste': 0.40,  # 40% reduction in hazardous waste
            'process optimization': 0.20, # 20% reduction in material waste
            'composting': 0.15,       # 15% reduction in organic waste
            'lean manufacturing': 0.30  # 30% reduction in overall waste
        }
        
        factor = reduction_factors.get(initiative_type, 0.20)
        
        if 'recycling' in initiative_type:
            # Increase recyclable waste
            current_recyclable = company_profile['waste_by_type'].get('recyclable_waste', 0)
            reduction = current_recyclable * factor
            savings = reduction * 30 / 1000  # $30 savings per ton recycled
        elif 'hazardous' in initiative_type:
            # Reduce hazardous waste
            current_hazardous = company_profile['waste_by_type'].get('hazardous_waste', 0)
            reduction = current_hazardous * factor
            savings = reduction * 150 / 1000  # $150 savings per ton hazardous waste
        else:
            # Reduce general waste
            current_solid = company_profile['waste_by_type'].get('solid_waste', 0)
            reduction = current_solid * factor
            savings = reduction * 50 / 1000  # $50 savings per ton solid waste
        
        return reduction, savings

    def generate_waste_report(self, company_data: Dict) -> Dict:
        """Generate comprehensive waste management report"""
        try:
            waste_profile = self.calculate_company_waste_profile(company_data)
            
            # Calculate efficiency metrics
            efficiency_metrics = self._calculate_waste_efficiency_metrics(company_data, waste_profile)
            
            # Generate action plan
            action_plan = self._generate_waste_action_plan(company_data, waste_profile)
            
            return {
                'company_info': {
                    'name': company_data.get('name'),
                    'industry': company_data.get('industry'),
                    'location': company_data.get('location'),
                    'employee_count': company_data.get('employee_count')
                },
                'waste_profile': waste_profile,
                'efficiency_metrics': efficiency_metrics,
                'action_plan': action_plan,
                'report_date': datetime.now().isoformat(),
                'report_id': str(uuid.uuid4())
            }
            
        except Exception as e:
            return {'error': f'Waste report generation failed: {str(e)}'}

    def _calculate_waste_efficiency_metrics(self, company_data: Dict, waste_profile: Dict) -> Dict:
        """Calculate waste efficiency metrics compared to industry benchmarks"""
        industry = company_data.get('industry', '').lower()
        employee_count = company_data.get('employee_count', 0)
        
        # Calculate per-employee waste generation
        total_waste = waste_profile['total_waste_generated']
        waste_per_employee = total_waste / max(employee_count, 1)
        
        # Industry benchmarks (kg per employee per month)
        industry_benchmarks = {
            'manufacturing': 455,  # Total waste per employee
            'textiles': 495,
            'food_beverage': 810,
            'chemicals': 290
        }
        
        benchmark = industry_benchmarks.get(industry, 455)
        efficiency_score = max(0, 100 - ((waste_per_employee - benchmark) / benchmark) * 100)
        
        return {
            'waste_per_employee': round(waste_per_employee, 2),
            'industry_benchmark': benchmark,
            'efficiency_score': round(efficiency_score, 1),
            'improvement_potential': round(100 - efficiency_score, 1),
            'recycling_rate': waste_profile['recycling_potential']['current_recycling_rate']
        }

    def _generate_waste_action_plan(self, company_data: Dict, waste_profile: Dict) -> Dict:
        """Generate actionable waste management plan"""
        recommendations = waste_profile['recommendations']
        
        # Prioritize recommendations
        high_priority = [r for r in recommendations if r['priority'] == 'high']
        medium_priority = [r for r in recommendations if r['priority'] == 'medium']
        
        # Calculate total investment and savings
        total_investment = sum(r['implementation_cost'] for r in recommendations)
        total_savings = sum(r['cost_savings'] for r in recommendations)
        
        # Calculate payback period
        payback_period = total_investment / max(total_savings, 1) * 12  # months
        
        return {
            'immediate_actions': high_priority[:3],  # Top 3 high priority actions
            'short_term_goals': medium_priority[:3],  # Top 3 medium priority actions
            'total_investment': round(total_investment, 2),
            'annual_savings': round(total_savings, 2),
            'payback_period_months': round(payback_period, 1),
            'roi_percentage': round((total_savings / max(total_investment, 1)) * 100, 1)
        }

# Initialize the engine
waste_engine = WasteTrackingEngine() 