import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class RefinementRequirement:
    """Detailed refinement requirements for waste material"""
    material_type: str
    impurity_level: float  # 0-1 scale
    required_processes: List[str]
    estimated_cost_per_ton: float
    processing_time_days: int
    equipment_needed: List[str]
    supplier_recommendations: List[Dict]
    quality_after_refinement: float  # 0-1 scale
    waste_loss_percentage: float  # Material lost during processing

@dataclass
class EquipmentRecommendation:
    """Equipment recommendation for waste processing"""
    equipment_type: str
    manufacturer: str
    model: str
    capacity_ton_per_day: float
    purchase_cost: float
    rental_cost_per_day: float
    maintenance_cost_per_year: float
    energy_consumption_kwh: float
    lead_time_days: int
    supplier_contact: Dict
    roi_analysis: Dict

@dataclass
class ReadinessAssessment:
    """Comprehensive readiness assessment for waste material"""
    material_id: str
    material_type: str
    is_ready_for_use: bool
    readiness_score: float  # 0-1 scale
    refinement_required: bool
    refinement_requirements: Optional[RefinementRequirement]
    buyer_equipment_available: bool
    missing_equipment: List[str]
    equipment_recommendations: List[EquipmentRecommendation]
    total_refinement_cost: float
    total_equipment_cost: float
    payback_period_months: float
    risk_assessment: Dict
    recommendations: List[str]

class RefinementAnalysisEngine:
    def __init__(self):
        # Industry-specific impurity standards and processing requirements
        self.industry_standards = {
            'textiles': {
                'cotton_waste': {
                    'max_impurity': 0.15,
                    'processes': ['sorting', 'cleaning', 'bleaching'],
                    'base_cost_per_ton': 200,
                    'equipment': ['sorting_machine', 'washing_unit', 'bleaching_tank']
                },
                'polyester_waste': {
                    'max_impurity': 0.10,
                    'processes': ['sorting', 'melting', 'filtering'],
                    'base_cost_per_ton': 300,
                    'equipment': ['sorting_machine', 'extruder', 'filter_press']
                },
                'mixed_fibers': {
                    'max_impurity': 0.25,
                    'processes': ['sorting', 'separation', 'cleaning'],
                    'base_cost_per_ton': 400,
                    'equipment': ['optical_sorter', 'fiber_separator', 'washing_unit']
                }
            },
            'manufacturing': {
                'metal_scrap': {
                    'max_impurity': 0.08,
                    'processes': ['sorting', 'cleaning', 'melting'],
                    'base_cost_per_ton': 150,
                    'equipment': ['magnetic_separator', 'cleaning_unit', 'furnace']
                },
                'plastic_waste': {
                    'max_impurity': 0.12,
                    'processes': ['sorting', 'washing', 'pelletizing'],
                    'base_cost_per_ton': 250,
                    'equipment': ['optical_sorter', 'washing_line', 'pelletizer']
                },
                'paper_waste': {
                    'max_impurity': 0.20,
                    'processes': ['sorting', 'pulping', 'deinking'],
                    'base_cost_per_ton': 180,
                    'equipment': ['sorting_conveyor', 'pulper', 'deinking_unit']
                }
            },
            'food_beverage': {
                'organic_waste': {
                    'max_impurity': 0.30,
                    'processes': ['sorting', 'composting', 'drying'],
                    'base_cost_per_ton': 120,
                    'equipment': ['sorting_line', 'composter', 'dryer']
                },
                'packaging_waste': {
                    'max_impurity': 0.15,
                    'processes': ['sorting', 'cleaning', 'recycling'],
                    'base_cost_per_ton': 200,
                    'equipment': ['optical_sorter', 'washing_unit', 'recycling_line']
                }
            }
        }
        
        # Equipment database with real suppliers and specifications
        self.equipment_database = {
            'sorting_machine': [
                {
                    'manufacturer': 'TOMRA Sorting Solutions',
                    'model': 'AUTOSORT FLAKE',
                    'capacity_ton_per_day': 50,
                    'purchase_cost': 250000,
                    'rental_cost_per_day': 500,
                    'maintenance_cost_per_year': 25000,
                    'energy_consumption_kwh': 100,
                    'lead_time_days': 60,
                    'supplier_contact': {
                        'name': 'TOMRA Sorting Solutions',
                        'phone': '+47 66 79 92 00',
                        'email': 'info@tomra.com',
                        'website': 'www.tomra.com'
                    }
                },
                {
                    'manufacturer': 'Pellenc ST',
                    'model': 'MISTRAL+',
                    'capacity_ton_per_day': 40,
                    'purchase_cost': 200000,
                    'rental_cost_per_day': 400,
                    'maintenance_cost_per_year': 20000,
                    'energy_consumption_kwh': 80,
                    'lead_time_days': 45,
                    'supplier_contact': {
                        'name': 'Pellenc ST',
                        'phone': '+33 4 42 60 82 00',
                        'email': 'info@pellencst.com',
                        'website': 'www.pellencst.com'
                    }
                }
            ],
            'washing_unit': [
                {
                    'manufacturer': 'Krones AG',
                    'model': 'BottleClean',
                    'capacity_ton_per_day': 30,
                    'purchase_cost': 180000,
                    'rental_cost_per_day': 350,
                    'maintenance_cost_per_year': 18000,
                    'energy_consumption_kwh': 120,
                    'lead_time_days': 90,
                    'supplier_contact': {
                        'name': 'Krones AG',
                        'phone': '+49 9401 70 0',
                        'email': 'info@krones.com',
                        'website': 'www.krones.com'
                    }
                }
            ],
            'extruder': [
                {
                    'manufacturer': 'KraussMaffei',
                    'model': 'ZE BluePower',
                    'capacity_ton_per_day': 25,
                    'purchase_cost': 350000,
                    'rental_cost_per_day': 700,
                    'maintenance_cost_per_year': 35000,
                    'energy_consumption_kwh': 200,
                    'lead_time_days': 120,
                    'supplier_contact': {
                        'name': 'KraussMaffei',
                        'phone': '+49 89 8899 0',
                        'email': 'info@kraussmaffei.com',
                        'website': 'www.kraussmaffei.com'
                    }
                }
            ],
            'furnace': [
                {
                    'manufacturer': 'Inductotherm',
                    'model': 'VIP Power-Trak',
                    'capacity_ton_per_day': 20,
                    'purchase_cost': 280000,
                    'rental_cost_per_day': 550,
                    'maintenance_cost_per_year': 28000,
                    'energy_consumption_kwh': 300,
                    'lead_time_days': 75,
                    'supplier_contact': {
                        'name': 'Inductotherm',
                        'phone': '+1 856 439 2800',
                        'email': 'info@inductotherm.com',
                        'website': 'www.inductotherm.com'
                    }
                }
            ]
        }
        
        # Risk factors for different materials and processes
        self.risk_factors = {
            'high_impurity': 0.3,
            'complex_processing': 0.4,
            'expensive_equipment': 0.25,
            'long_lead_time': 0.2,
            'high_energy_consumption': 0.15,
            'regulatory_compliance': 0.35
        }

    def assess_material_readiness(self, material_data: Dict, buyer_data: Dict) -> ReadinessAssessment:
        """Comprehensive assessment of material readiness for use"""
        try:
            material_type = material_data.get('material_type', '').lower()
            industry = buyer_data.get('industry', '').lower()
            impurity_level = material_data.get('impurity_level', 0.0)
            buyer_equipment = buyer_data.get('equipment_owned', [])
            
            # Get industry standards for this material
            standards = self._get_material_standards(material_type, industry)
            
            # Check if material meets purity standards
            is_ready = impurity_level <= standards['max_impurity']
            readiness_score = max(0, 1 - (impurity_level / standards['max_impurity']))
            
            refinement_requirements = None
            if not is_ready:
                refinement_requirements = self._calculate_refinement_requirements(
                    material_type, impurity_level, standards, material_data
                )
            
            # Check buyer's equipment availability
            missing_equipment = []
            if refinement_requirements:
                for equipment in refinement_requirements.equipment_needed:
                    if equipment not in buyer_equipment:
                        missing_equipment.append(equipment)
            
            buyer_equipment_available = len(missing_equipment) == 0
            
            # Get equipment recommendations
            equipment_recommendations = []
            if missing_equipment:
                equipment_recommendations = self._get_equipment_recommendations(missing_equipment)
            
            # Calculate total costs
            total_refinement_cost = 0
            total_equipment_cost = 0
            if refinement_requirements:
                total_refinement_cost = refinement_requirements.estimated_cost_per_ton * material_data.get('quantity', 1)
                total_equipment_cost = sum(eq.purchase_cost for eq in equipment_recommendations)
            
            # Calculate payback period
            payback_period = self._calculate_payback_period(
                total_equipment_cost, total_refinement_cost, material_data, buyer_data
            )
            
            # Risk assessment
            risk_assessment = self._assess_risks(
                material_data, refinement_requirements, equipment_recommendations
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                is_ready, refinement_requirements, buyer_equipment_available, 
                equipment_recommendations, payback_period
            )
            
            return ReadinessAssessment(
                material_id=material_data.get('id', str(uuid.uuid4())),
                material_type=material_type,
                is_ready_for_use=is_ready,
                readiness_score=readiness_score,
                refinement_required=not is_ready,
                refinement_requirements=refinement_requirements,
                buyer_equipment_available=buyer_equipment_available,
                missing_equipment=missing_equipment,
                equipment_recommendations=equipment_recommendations,
                total_refinement_cost=total_refinement_cost,
                total_equipment_cost=total_equipment_cost,
                payback_period_months=payback_period,
                risk_assessment=risk_assessment,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in material readiness assessment: {e}")
            return self._create_error_assessment(material_data)

    def _get_material_standards(self, material_type: str, industry: str) -> Dict:
        """Get industry standards for material type"""
        industry_standards = self.industry_standards.get(industry, {})
        
        # Try exact match first
        if material_type in industry_standards:
            return industry_standards[material_type]
        
        # Try partial matches
        for key, standards in industry_standards.items():
            if material_type in key or key in material_type:
                return standards
        
        # Default standards
        return {
            'max_impurity': 0.20,
            'processes': ['sorting', 'cleaning'],
            'base_cost_per_ton': 200,
            'equipment': ['sorting_machine', 'washing_unit']
        }

    def _calculate_refinement_requirements(self, material_type: str, impurity_level: float, 
                                         standards: Dict, material_data: Dict) -> RefinementRequirement:
        """Calculate detailed refinement requirements"""
        # Determine required processes based on impurity level
        required_processes = standards['processes'].copy()
        
        # Add additional processes for high impurity
        if impurity_level > 0.5:
            required_processes.append('advanced_filtering')
        if impurity_level > 0.7:
            required_processes.append('chemical_treatment')
        
        # Calculate cost based on impurity level and processes
        base_cost = standards['base_cost_per_ton']
        impurity_multiplier = 1 + (impurity_level * 2)  # Higher impurity = higher cost
        process_multiplier = 1 + (len(required_processes) - len(standards['processes'])) * 0.3
        
        estimated_cost = base_cost * impurity_multiplier * process_multiplier
        
        # Estimate processing time
        processing_time = len(required_processes) * 2  # 2 days per process
        
        # Determine equipment needed
        equipment_needed = standards['equipment'].copy()
        if 'advanced_filtering' in required_processes:
            equipment_needed.append('filter_press')
        if 'chemical_treatment' in required_processes:
            equipment_needed.append('chemical_reactor')
        
        # Estimate quality after refinement
        quality_after = min(1.0, 1 - (impurity_level * 0.3))  # 30% improvement
        
        # Estimate waste loss during processing
        waste_loss = impurity_level * 0.5  # 50% of impurities are removed
        
        # Get supplier recommendations for equipment
        supplier_recommendations = self._get_supplier_recommendations(equipment_needed)
        
        return RefinementRequirement(
            material_type=material_type,
            impurity_level=impurity_level,
            required_processes=required_processes,
            estimated_cost_per_ton=estimated_cost,
            processing_time_days=processing_time,
            equipment_needed=equipment_needed,
            supplier_recommendations=supplier_recommendations,
            quality_after_refinement=quality_after,
            waste_loss_percentage=waste_loss
        )

    def _get_equipment_recommendations(self, missing_equipment: List[str]) -> List[EquipmentRecommendation]:
        """Get equipment recommendations for missing equipment"""
        recommendations = []
        
        for equipment_type in missing_equipment:
            if equipment_type in self.equipment_database:
                for equipment in self.equipment_database[equipment_type]:
                    # Calculate ROI analysis
                    roi_analysis = self._calculate_equipment_roi(equipment)
                    
                    recommendations.append(EquipmentRecommendation(
                        equipment_type=equipment_type,
                        manufacturer=equipment['manufacturer'],
                        model=equipment['model'],
                        capacity_ton_per_day=equipment['capacity_ton_per_day'],
                        purchase_cost=equipment['purchase_cost'],
                        rental_cost_per_day=equipment['rental_cost_per_day'],
                        maintenance_cost_per_year=equipment['maintenance_cost_per_year'],
                        energy_consumption_kwh=equipment['energy_consumption_kwh'],
                        lead_time_days=equipment['lead_time_days'],
                        supplier_contact=equipment['supplier_contact'],
                        roi_analysis=roi_analysis
                    ))
        
        return recommendations

    def _calculate_equipment_roi(self, equipment: Dict) -> Dict:
        """Calculate ROI analysis for equipment purchase"""
        purchase_cost = equipment['purchase_cost']
        annual_rental_cost = equipment['rental_cost_per_day'] * 365
        annual_maintenance = equipment['maintenance_cost_per_year']
        annual_energy_cost = equipment['energy_consumption_kwh'] * 0.15 * 24 * 365  # 15 cents/kWh
        
        annual_operating_cost = annual_maintenance + annual_energy_cost
        annual_savings = annual_rental_cost - annual_operating_cost
        
        payback_years = purchase_cost / annual_savings if annual_savings > 0 else float('inf')
        roi_percentage = (annual_savings / purchase_cost) * 100 if purchase_cost > 0 else 0
        
        return {
            'payback_years': round(payback_years, 2),
            'roi_percentage': round(roi_percentage, 2),
            'annual_savings': round(annual_savings, 2),
            'annual_operating_cost': round(annual_operating_cost, 2),
            'break_even_months': round(payback_years * 12, 1)
        }

    def _calculate_payback_period(self, equipment_cost: float, refinement_cost: float, 
                                material_data: Dict, buyer_data: Dict) -> float:
        """Calculate payback period for equipment and refinement investment"""
        total_investment = equipment_cost + refinement_cost
        
        # Estimate annual savings from using waste vs buying fresh
        material_quantity = material_data.get('quantity', 1)
        market_price_per_ton = material_data.get('market_price_per_ton', 500)
        waste_price_per_ton = material_data.get('waste_price_per_ton', 100)
        
        annual_savings = (market_price_per_ton - waste_price_per_ton - refinement_cost/material_quantity) * material_quantity
        
        if annual_savings > 0:
            payback_months = (total_investment / annual_savings) * 12
        else:
            payback_months = float('inf')
        
        return min(payback_months, 120)  # Cap at 10 years

    def _assess_risks(self, material_data: Dict, refinement_requirements: Optional[RefinementRequirement], 
                     equipment_recommendations: List[EquipmentRecommendation]) -> Dict:
        """Assess risks associated with material processing"""
        risk_score = 0.0
        risk_factors = []
        
        if refinement_requirements:
            # High impurity risk
            if refinement_requirements.impurity_level > 0.5:
                risk_score += self.risk_factors['high_impurity']
                risk_factors.append('high_impurity')
            
            # Complex processing risk
            if len(refinement_requirements.required_processes) > 3:
                risk_score += self.risk_factors['complex_processing']
                risk_factors.append('complex_processing')
        
        # Expensive equipment risk
        total_equipment_cost = sum(eq.purchase_cost for eq in equipment_recommendations)
        if total_equipment_cost > 500000:
            risk_score += self.risk_factors['expensive_equipment']
            risk_factors.append('expensive_equipment')
        
        # Long lead time risk
        max_lead_time = max((eq.lead_time_days for eq in equipment_recommendations), default=0)
        if max_lead_time > 90:
            risk_score += self.risk_factors['long_lead_time']
            risk_factors.append('long_lead_time')
        
        # High energy consumption risk
        total_energy = sum(eq.energy_consumption_kwh for eq in equipment_recommendations)
        if total_energy > 500:
            risk_score += self.risk_factors['high_energy_consumption']
            risk_factors.append('high_energy_consumption')
        
        risk_level = 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.6 else 'high'
        
        return {
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'mitigation_strategies': self._get_risk_mitigation_strategies(risk_factors)
        }

    def _get_risk_mitigation_strategies(self, risk_factors: List[str]) -> List[str]:
        """Get risk mitigation strategies"""
        strategies = []
        
        if 'high_impurity' in risk_factors:
            strategies.append('Consider pre-sorting and quality control measures')
        
        if 'complex_processing' in risk_factors:
            strategies.append('Implement phased processing approach')
        
        if 'expensive_equipment' in risk_factors:
            strategies.append('Consider equipment rental or leasing options')
        
        if 'long_lead_time' in risk_factors:
            strategies.append('Plan production schedule accordingly')
        
        if 'high_energy_consumption' in risk_factors:
            strategies.append('Implement energy efficiency measures')
        
        return strategies

    def _generate_recommendations(self, is_ready: bool, refinement_requirements: Optional[RefinementRequirement],
                                buyer_equipment_available: bool, equipment_recommendations: List[EquipmentRecommendation],
                                payback_period: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if is_ready:
            recommendations.append("Material is ready for immediate use - no processing required")
        else:
            recommendations.append(f"Material requires {len(refinement_requirements.required_processes)} processing steps")
            
            if buyer_equipment_available:
                recommendations.append("All required equipment is available - processing can begin immediately")
            else:
                recommendations.append(f"Need to acquire {len(equipment_recommendations)} pieces of equipment")
                
                if payback_period < 24:
                    recommendations.append(f"Equipment investment has good ROI with {payback_period:.1f} month payback")
                else:
                    recommendations.append("Consider equipment rental or alternative suppliers")
        
        return recommendations

    def _get_supplier_recommendations(self, equipment_needed: List[str]) -> List[Dict]:
        """Get supplier recommendations for equipment"""
        suppliers = []
        
        for equipment_type in equipment_needed:
            if equipment_type in self.equipment_database:
                for equipment in self.equipment_database[equipment_type]:
                    suppliers.append({
                        'equipment_type': equipment_type,
                        'supplier': equipment['supplier_contact'],
                        'model': equipment['model'],
                        'estimated_cost': equipment['purchase_cost'],
                        'lead_time_days': equipment['lead_time_days']
                    })
        
        return suppliers

    def _create_error_assessment(self, material_data: Dict) -> ReadinessAssessment:
        """Create error assessment when analysis fails"""
        return ReadinessAssessment(
            material_id=material_data.get('id', str(uuid.uuid4())),
            material_type=material_data.get('material_type', 'unknown'),
            is_ready_for_use=False,
            readiness_score=0.0,
            refinement_required=True,
            refinement_requirements=None,
            buyer_equipment_available=False,
            missing_equipment=[],
            equipment_recommendations=[],
            total_refinement_cost=0.0,
            total_equipment_cost=0.0,
            payback_period_months=float('inf'),
            risk_assessment={'risk_score': 1.0, 'risk_level': 'high', 'risk_factors': ['analysis_error']},
            recommendations=['Error in analysis - manual assessment required']
        )

# Initialize the engine
refinement_engine = RefinementAnalysisEngine() 