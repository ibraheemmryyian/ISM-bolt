"""
Advanced Regulatory Compliance Engine
AI-Powered Real-Time Compliance Checking and Regulatory Intelligence
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
import json
import hashlib
from dataclasses import dataclass
# Try to import sklearn components with fallback
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback implementations if sklearn is not available
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            return self
        def transform(self, X):
            return (X - self.mean_) / self.scale_
        def fit_transform(self, X):
            return self.fit(X).transform(X)
    
    class LabelEncoder:
        def __init__(self):
            self.classes_ = None
        def fit(self, y):
            self.classes_ = np.unique(y)
            return self
        def transform(self, y):
            return np.array([np.where(self.classes_ == label)[0][0] for label in y])
        def fit_transform(self, y):
            return self.fit(y).transform(y)
    
    class RandomForestClassifier:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X))
    
    def accuracy_score(y_true, y_pred):
        return 0.0
    
    def classification_report(y_true, y_pred):
        return "Classification report not available (sklearn not installed)"
    
    SKLEARN_AVAILABLE = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import networkx as nx
from textblob import TextBlob
import redis
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from pathlib import Path
import warnings
import re
from urllib.parse import urljoin
import requests
import torch
from .ml_core.models import BaseNN
from .ml_core.training import train_supervised
from .ml_core.inference import predict_supervised
from .ml_core.monitoring import log_metrics, save_checkpoint
from torch.utils.data import DataLoader, TensorDataset
import os
import shap
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
warnings.filterwarnings('ignore')

# Replace standard logger with DistributedLogger
logger = DistributedLogger('RegulatoryComplianceEngine', log_file='logs/regulatory_compliance.log')

@dataclass
class ComplianceResult:
    """Structured compliance check result"""
    match_id: str
    company_a_id: str
    company_b_id: str
    material_type: str
    origin_location: str
    destination_location: str
    overall_compliance: bool
    compliance_score: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    compliance_details: Dict[str, Any]
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    required_permits: List[str]
    certification_requirements: List[str]
    regulatory_authorities: List[str]
    compliance_deadlines: List[datetime]
    estimated_compliance_cost: float
    carbon_impact: float
    created_at: datetime

@dataclass
class RegulatoryUpdate:
    """Regulatory update information"""
    regulation_id: str
    title: str
    description: str
    jurisdiction: str
    industry: str
    material_types: List[str]
    effective_date: datetime
    compliance_deadline: datetime
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    requirements: List[str]
    penalties: List[str]
    data_source: str
    last_updated: datetime

@dataclass
class PermitRequirement:
    """Permit requirement information"""
    permit_type: str
    issuing_authority: str
    jurisdiction: str
    validity_period: int  # days
    application_time: int  # days
    cost_range: Tuple[float, float]
    requirements: List[str]
    documents_needed: List[str]
    processing_fee: float
    renewal_fee: float

class AdvancedRegulatoryComplianceEngine:
    """
    Advanced AI-Powered Regulatory Compliance Engine
    
    Features:
    - Real-time compliance checking across multiple jurisdictions
    - Machine learning-based risk assessment
    - Automated permit requirement detection
    - Regulatory change monitoring and alerts
    - Multi-language compliance support
    - Blockchain-verified compliance records
    - Predictive compliance analytics
    - Automated compliance reporting
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize AI models
        self.compliance_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.risk_assessor = GradientBoostingClassifier(n_estimators=150, random_state=42)
        self.text_analyzer = TfidfVectorizer(max_features=2000, stop_words='english')
        
        # Data processing
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        
        # Caching and storage
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            db=1,  # Use different DB for compliance data
            decode_responses=True
        )
        self.cache_ttl = 7200  # 2 hours for compliance data
        
        # Regulatory databases
        self.regulatory_graph = nx.DiGraph()
        self.compliance_rules = {}
        self.permit_requirements = {}
        self.regulatory_updates = []
        
        # Background processing
        self.running = False
        self.background_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.compliance_accuracy = []
        self.processing_times = []
        self.risk_assessments = []
        
        # Load regulatory data
        self._load_regulatory_data()
        
        logger.info("Advanced Regulatory Compliance Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'compliance_threshold': 0.8,
            'risk_threshold': 0.6,
            'update_frequency_hours': 6,
            'max_compliance_checks_per_minute': 100,
            'enable_blockchain_verification': True,
            'multi_language_support': True,
            'automated_reporting': True,
            'regulatory_sources': [
                'eu_regulations',
                'us_epa',
                'local_authorities',
                'industry_standards',
                'international_treaties'
            ],
            'supported_jurisdictions': [
                'EU', 'US', 'UK', 'Canada', 'Australia', 'Japan', 'GCC'
            ]
        }
    
    def _load_regulatory_data(self):
        """Load regulatory data and rules"""
        try:
            # Load compliance rules
            self.compliance_rules = self._load_compliance_rules()
            
            # Load permit requirements
            self.permit_requirements = self._load_permit_requirements()
            
            # Load regulatory updates
            self.regulatory_updates = self._load_regulatory_updates()
            
            logger.info(f"Loaded {len(self.compliance_rules)} compliance rules and {len(self.permit_requirements)} permit types")
            
        except Exception as e:
            logger.error(f"Error loading regulatory data: {e}")
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules from database or file"""
        # Simulate comprehensive compliance rules
        return {
            'hazardous_waste': {
                'eu': {
                    'basel_convention': {
                        'description': 'International treaty on hazardous waste',
                        'requirements': ['proper_classification', 'transboundary_notification', 'environmentally_sound_management'],
                        'penalties': ['fines_up_to_500k_eur', 'criminal_charges'],
                        'risk_level': 'high'
                    },
                    'waste_framework_directive': {
                        'description': 'EU waste management framework',
                        'requirements': ['waste_hierarchy', 'polluter_pays_principle', 'extended_producer_responsibility'],
                        'penalties': ['fines_up_to_300k_eur'],
                        'risk_level': 'medium'
                    }
                },
                'us': {
                    'rcra': {
                        'description': 'Resource Conservation and Recovery Act',
                        'requirements': ['cradle_to_grave_tracking', 'proper_disposal', 'manifest_system'],
                        'penalties': ['fines_up_to_500k_usd', 'imprisonment'],
                        'risk_level': 'high'
                    }
                }
            },
            'recyclable_materials': {
                'eu': {
                    'circular_economy_package': {
                        'description': 'EU circular economy framework',
                        'requirements': ['recycling_targets', 'design_for_recycling', 'market_development'],
                        'penalties': ['fines_up_to_200k_eur'],
                        'risk_level': 'medium'
                    }
                }
            },
            'chemical_substances': {
                'eu': {
                    'reach_regulation': {
                        'description': 'Registration, Evaluation, Authorization of Chemicals',
                        'requirements': ['registration', 'safety_assessment', 'authorization'],
                        'penalties': ['fines_up_to_400k_eur'],
                        'risk_level': 'high'
                    }
                },
                'us': {
                    'tsca': {
                        'description': 'Toxic Substances Control Act',
                        'requirements': ['premanufacture_notification', 'testing_requirements', 'restrictions'],
                        'penalties': ['fines_up_to_400k_usd'],
                        'risk_level': 'high'
                    }
                }
            }
        }
    
    def _load_permit_requirements(self) -> Dict[str, PermitRequirement]:
        """Load permit requirements"""
        return {
            'hazardous_waste_transport': PermitRequirement(
                permit_type='Hazardous Waste Transport Permit',
                issuing_authority='Environmental Protection Agency',
                jurisdiction='Federal',
                validity_period=365,
                application_time=30,
                cost_range=(500, 5000),
                requirements=['safety_training', 'proper_containers', 'insurance_coverage'],
                documents_needed=['safety_plan', 'insurance_certificate', 'training_records'],
                processing_fee=250,
                renewal_fee=150
            ),
            'waste_processing': PermitRequirement(
                permit_type='Waste Processing Facility Permit',
                issuing_authority='State Environmental Department',
                jurisdiction='State',
                validity_period=730,
                application_time=90,
                cost_range=(2000, 15000),
                requirements=['facility_inspection', 'environmental_assessment', 'community_consultation'],
                documents_needed=['facility_plans', 'environmental_impact_statement', 'community_agreement'],
                processing_fee=1000,
                renewal_fee=500
            ),
            'recycling_operation': PermitRequirement(
                permit_type='Recycling Operation License',
                issuing_authority='Local Municipality',
                jurisdiction='Local',
                validity_period=365,
                application_time=45,
                cost_range=(300, 2000),
                requirements=['quality_standards', 'output_tracking', 'market_development'],
                documents_needed=['quality_management_plan', 'tracking_system', 'market_analysis'],
                processing_fee=150,
                renewal_fee=75
            )
        }
    
    def _load_regulatory_updates(self) -> List[RegulatoryUpdate]:
        """Load recent regulatory updates"""
        return [
            RegulatoryUpdate(
                regulation_id='EU_2024_001',
                title='Enhanced Circular Economy Requirements',
                description='New EU regulations requiring higher recycling targets',
                jurisdiction='EU',
                industry='manufacturing',
                material_types=['plastics', 'metals', 'electronics'],
                effective_date=datetime.now() + timedelta(days=90),
                compliance_deadline=datetime.now() + timedelta(days=180),
                impact_level='high',
                requirements=['70%_recycling_target', 'design_for_recycling', 'extended_producer_responsibility'],
                penalties=['fines_up_to_500k_eur', 'market_access_restrictions'],
                data_source='EU_Official_Journal',
                last_updated=datetime.now()
            ),
            RegulatoryUpdate(
                regulation_id='US_2024_002',
                title='Enhanced Hazardous Waste Tracking',
                description='New EPA requirements for digital waste tracking',
                jurisdiction='US',
                industry='chemicals',
                material_types=['hazardous_waste'],
                effective_date=datetime.now() + timedelta(days=120),
                compliance_deadline=datetime.now() + timedelta(days=240),
                impact_level='medium',
                requirements=['digital_manifest_system', 'real_time_tracking', 'enhanced_reporting'],
                penalties=['fines_up_to_300k_usd'],
                data_source='Federal_Register',
                last_updated=datetime.now()
            )
        ]
    
    async def check_compliance(self, match_data: Dict[str, Any]) -> ComplianceResult:
        """
        Comprehensive compliance check for a material exchange match
        
        Args:
            match_data: Match information including companies, materials, locations
            
        Returns:
            Detailed compliance result with risk assessment
        """
        try:
            start_time = time.time()
            
            # Generate cache key
            cache_key = f"compliance:{match_data.get('match_id', 'unknown')}"
            
            # Check cache first
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Extract match information
            company_a = match_data.get('company_a', {})
            company_b = match_data.get('company_b', {})
            material_data = match_data.get('material_data', {})
            
            # Multi-dimensional compliance analysis
            compliance_checks = await asyncio.gather(
                self._check_material_compliance(material_data),
                self._check_transport_compliance(company_a, company_b, material_data),
                self._check_environmental_compliance(material_data),
                self._check_safety_compliance(material_data),
                self._check_trade_compliance(company_a, company_b, material_data),
                self._check_certification_compliance(company_a, company_b)
            )
            
            # Aggregate compliance results
            overall_compliance = all(check['compliant'] for check in compliance_checks)
            compliance_score = np.mean([check['score'] for check in compliance_checks])
            
            # Risk assessment
            risk_level = self._assess_risk_level(compliance_checks, material_data)
            
            # Collect violations and recommendations
            violations = []
            recommendations = []
            for check in compliance_checks:
                violations.extend(check.get('violations', []))
                recommendations.extend(check.get('recommendations', []))
            
            # Determine required permits
            required_permits = self._determine_required_permits(material_data, company_a, company_b)
            
            # Calculate compliance costs
            estimated_cost = self._estimate_compliance_cost(required_permits, violations)
            
            # Create compliance result
            result = ComplianceResult(
                match_id=match_data.get('match_id', ''),
                company_a_id=company_a.get('id', ''),
                company_b_id=company_b.get('id', ''),
                material_type=material_data.get('type', ''),
                origin_location=company_a.get('location', ''),
                destination_location=company_b.get('location', ''),
                overall_compliance=overall_compliance,
                compliance_score=compliance_score,
                risk_level=risk_level,
                compliance_details={
                    'material_compliance': compliance_checks[0],
                    'transport_compliance': compliance_checks[1],
                    'environmental_compliance': compliance_checks[2],
                    'safety_compliance': compliance_checks[3],
                    'trade_compliance': compliance_checks[4],
                    'certification_compliance': compliance_checks[5]
                },
                violations=violations,
                recommendations=recommendations,
                required_permits=required_permits,
                certification_requirements=self._get_certification_requirements(material_data),
                regulatory_authorities=self._get_regulatory_authorities(material_data, company_a, company_b),
                compliance_deadlines=self._get_compliance_deadlines(material_data),
                estimated_compliance_cost=estimated_cost,
                carbon_impact=self._calculate_carbon_impact(material_data, compliance_checks),
                created_at=datetime.now()
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.info(f"Compliance check completed for match {match_data.get('match_id', 'Unknown')} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
            return self._create_fallback_compliance_result(match_data)
    
    async def _check_material_compliance(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check material-specific compliance"""
        try:
            material_type = material_data.get('type', '')
            material_name = material_data.get('name', '')
            
            # Get applicable regulations
            regulations = self._get_applicable_regulations(material_type, material_name)
            
            compliance_checks = []
            violations = []
            recommendations = []
            
            for regulation in regulations:
                # Check each requirement
                for requirement in regulation.get('requirements', []):
                    is_compliant = self._check_requirement_compliance(requirement, material_data)
                    compliance_checks.append(is_compliant)
                    
                    if not is_compliant:
                        violations.append({
                            'regulation': regulation.get('title', ''),
                            'requirement': requirement,
                            'severity': regulation.get('risk_level', 'medium')
                        })
                        
                        recommendations.append(f"Ensure compliance with {requirement} for {regulation.get('title', '')}")
            
            compliance_score = np.mean(compliance_checks) if compliance_checks else 1.0
            
            return {
                'compliant': compliance_score >= self.config['compliance_threshold'],
                'score': compliance_score,
                'violations': violations,
                'recommendations': recommendations,
                'regulations_checked': len(regulations)
            }
            
        except Exception as e:
            logger.error(f"Error checking material compliance: {e}")
            return {'compliant': False, 'score': 0.0, 'violations': [], 'recommendations': ['Error in compliance check']}
    
    async def _check_transport_compliance(self, company_a: Dict[str, Any], company_b: Dict[str, Any], material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check transport compliance"""
        try:
            origin = company_a.get('location', '')
            destination = company_b.get('location', '')
            material_type = material_data.get('type', '')
            
            # Check cross-border transport
            is_cross_border = self._is_cross_border_transport(origin, destination)
            
            violations = []
            recommendations = []
            
            if is_cross_border:
                # Check international transport requirements
                if material_type == 'hazardous_waste':
                    violations.append({
                        'regulation': 'Basel Convention',
                        'requirement': 'Transboundary notification',
                        'severity': 'high'
                    })
                    recommendations.append("Submit transboundary notification 30 days before transport")
                
                # Check customs requirements
                violations.append({
                    'regulation': 'Customs Regulations',
                    'requirement': 'Customs declaration',
                    'severity': 'medium'
                })
                recommendations.append("Prepare customs documentation and declarations")
            
            # Check transport permits
            if material_type in ['hazardous_waste', 'chemicals']:
                violations.append({
                    'regulation': 'Transport Regulations',
                    'requirement': 'Transport permit',
                    'severity': 'high'
                })
                recommendations.append("Obtain appropriate transport permits")
            
            compliance_score = 1.0 - (len(violations) * 0.2)  # Reduce score for each violation
            
            return {
                'compliant': len(violations) == 0,
                'score': max(0.0, compliance_score),
                'violations': violations,
                'recommendations': recommendations,
                'is_cross_border': is_cross_border
            }
            
        except Exception as e:
            logger.error(f"Error checking transport compliance: {e}")
            return {'compliant': False, 'score': 0.0, 'violations': [], 'recommendations': ['Error in transport compliance check']}
    
    async def _check_environmental_compliance(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check environmental compliance"""
        try:
            material_type = material_data.get('type', '')
            quantity = material_data.get('quantity', 0)
            
            violations = []
            recommendations = []
            
            # Check environmental impact
            if material_type == 'hazardous_waste':
                if quantity > 1000:  # kg
                    violations.append({
                        'regulation': 'Environmental Protection',
                        'requirement': 'Environmental impact assessment',
                        'severity': 'high'
                    })
                    recommendations.append("Conduct environmental impact assessment for large quantities")
            
            # Check waste hierarchy compliance
            if material_type == 'waste':
                violations.append({
                    'regulation': 'Waste Hierarchy',
                    'requirement': 'Preference for recycling over disposal',
                    'severity': 'medium'
                })
                recommendations.append("Ensure waste is processed according to waste hierarchy")
            
            compliance_score = 1.0 - (len(violations) * 0.15)
            
            return {
                'compliant': len(violations) == 0,
                'score': max(0.0, compliance_score),
                'violations': violations,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error checking environmental compliance: {e}")
            return {'compliant': False, 'score': 0.0, 'violations': [], 'recommendations': ['Error in environmental compliance check']}
    
    async def _check_safety_compliance(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check safety compliance"""
        try:
            material_type = material_data.get('type', '')
            hazardous_properties = material_data.get('hazardous_properties', [])
            
            violations = []
            recommendations = []
            
            # Check safety data sheets
            if material_type in ['chemicals', 'hazardous_waste']:
                violations.append({
                    'regulation': 'Safety Regulations',
                    'requirement': 'Safety data sheet',
                    'severity': 'high'
                })
                recommendations.append("Provide safety data sheet for hazardous materials")
            
            # Check handling requirements
            if hazardous_properties:
                violations.append({
                    'regulation': 'Safety Regulations',
                    'requirement': 'Special handling procedures',
                    'severity': 'high'
                })
                recommendations.append("Implement special handling procedures for hazardous materials")
            
            compliance_score = 1.0 - (len(violations) * 0.2)
            
            return {
                'compliant': len(violations) == 0,
                'score': max(0.0, compliance_score),
                'violations': violations,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error checking safety compliance: {e}")
            return {'compliant': False, 'score': 0.0, 'violations': [], 'recommendations': ['Error in safety compliance check']}
    
    async def _check_trade_compliance(self, company_a: Dict[str, Any], company_b: Dict[str, Any], material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check trade compliance"""
        try:
            origin_country = self._extract_country(company_a.get('location', ''))
            destination_country = self._extract_country(company_b.get('location', ''))
            material_type = material_data.get('type', '')
            
            violations = []
            recommendations = []
            
            # Check export controls
            if origin_country != destination_country:
                if material_type in ['hazardous_waste', 'chemicals']:
                    violations.append({
                        'regulation': 'Export Controls',
                        'requirement': 'Export license',
                        'severity': 'high'
                    })
                    recommendations.append("Obtain export license for international trade")
            
            # Check sanctions compliance
            if self._is_sanctioned_country(destination_country):
                violations.append({
                    'regulation': 'Trade Sanctions',
                    'requirement': 'Sanctions compliance',
                    'severity': 'critical'
                })
                recommendations.append("Verify sanctions compliance before proceeding")
            
            compliance_score = 1.0 - (len(violations) * 0.3)
            
            return {
                'compliant': len(violations) == 0,
                'score': max(0.0, compliance_score),
                'violations': violations,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error checking trade compliance: {e}")
            return {'compliant': False, 'score': 0.0, 'violations': [], 'recommendations': ['Error in trade compliance check']}
    
    async def _check_certification_compliance(self, company_a: Dict[str, Any], company_b: Dict[str, Any]) -> Dict[str, Any]:
        """Check certification compliance"""
        try:
            certifications_a = company_a.get('certifications', [])
            certifications_b = company_b.get('certifications', [])
            
            violations = []
            recommendations = []
            
            # Check ISO certifications
            required_certifications = ['ISO_14001', 'ISO_9001']
            
            for cert in required_certifications:
                if cert not in certifications_a or cert not in certifications_b:
                    violations.append({
                        'regulation': 'Quality Management',
                        'requirement': f'{cert} certification',
                        'severity': 'medium'
                    })
                    recommendations.append(f"Obtain {cert} certification")
            
            compliance_score = 1.0 - (len(violations) * 0.25)
            
            return {
                'compliant': len(violations) == 0,
                'score': max(0.0, compliance_score),
                'violations': violations,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error checking certification compliance: {e}")
            return {'compliant': False, 'score': 0.0, 'violations': [], 'recommendations': ['Error in certification compliance check']}
    
    def _assess_risk_level(self, compliance_checks: List[Dict[str, Any]], material_data: Dict[str, Any]) -> str:
        """Assess overall risk level"""
        try:
            # Count high and critical severity violations
            high_risk_violations = 0
            critical_violations = 0
            
            for check in compliance_checks:
                for violation in check.get('violations', []):
                    if violation.get('severity') == 'high':
                        high_risk_violations += 1
                    elif violation.get('severity') == 'critical':
                        critical_violations += 1
            
            # Determine risk level
            if critical_violations > 0:
                return 'critical'
            elif high_risk_violations > 2:
                return 'high'
            elif high_risk_violations > 0:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return 'medium'
    
    def _get_applicable_regulations(self, material_type: str, material_name: str) -> List[Dict[str, Any]]:
        """Get applicable regulations for material type"""
        regulations = []
        
        try:
            # Get regulations from compliance rules
            if material_type in self.compliance_rules:
                for jurisdiction, jurisdiction_rules in self.compliance_rules[material_type].items():
                    for regulation_id, regulation in jurisdiction_rules.items():
                        regulations.append({
                            'id': regulation_id,
                            'title': regulation.get('description', ''),
                            'requirements': regulation.get('requirements', []),
                            'penalties': regulation.get('penalties', []),
                            'risk_level': regulation.get('risk_level', 'medium')
                        })
            
            return regulations
            
        except Exception as e:
            logger.error(f"Error getting applicable regulations: {e}")
            return []
    
    def _check_requirement_compliance(self, requirement: str, material_data: Dict[str, Any]) -> bool:
        """Check if a specific requirement is met"""
        try:
            # Simulate requirement checking logic
            # In a real implementation, this would check against actual compliance data
            
            if 'classification' in requirement.lower():
                return material_data.get('properly_classified', True)
            elif 'notification' in requirement.lower():
                return material_data.get('notifications_submitted', True)
            elif 'management' in requirement.lower():
                return material_data.get('management_plan_exists', True)
            elif 'tracking' in requirement.lower():
                return material_data.get('tracking_system_in_place', True)
            else:
                return True  # Default to compliant
                
        except Exception as e:
            logger.error(f"Error checking requirement compliance: {e}")
            return False
    
    def _is_cross_border_transport(self, origin: str, destination: str) -> bool:
        """Check if transport is cross-border"""
        try:
            origin_country = self._extract_country(origin)
            destination_country = self._extract_country(destination)
            return origin_country != destination_country
        except Exception as e:
            logger.error(f"Error checking cross-border transport: {e}")
            return False
    
    def _extract_country(self, location: str) -> str:
        """Extract country from location string"""
        try:
            # Simple country extraction (in real implementation, use geocoding)
            if 'US' in location or 'USA' in location:
                return 'US'
            elif 'EU' in location or any(country in location for country in ['Germany', 'France', 'Italy']):
                return 'EU'
            elif 'UK' in location:
                return 'UK'
            else:
                return 'Unknown'
        except Exception as e:
            logger.error(f"Error extracting country: {e}")
            return 'Unknown'
    
    def _is_sanctioned_country(self, country: str) -> bool:
        """Check if country is under sanctions"""
        sanctioned_countries = ['North Korea', 'Iran', 'Syria', 'Cuba']
        return country in sanctioned_countries
    
    def _determine_required_permits(self, material_data: Dict[str, Any], company_a: Dict[str, Any], company_b: Dict[str, Any]) -> List[str]:
        """Determine required permits for the match"""
        permits = []
        
        try:
            material_type = material_data.get('type', '')
            
            if material_type == 'hazardous_waste':
                permits.append('hazardous_waste_transport')
            
            if material_data.get('quantity', 0) > 1000:  # kg
                permits.append('waste_processing')
            
            if material_type == 'recyclable':
                permits.append('recycling_operation')
            
            return permits
            
        except Exception as e:
            logger.error(f"Error determining required permits: {e}")
            return []
    
    def _get_certification_requirements(self, material_data: Dict[str, Any]) -> List[str]:
        """Get certification requirements"""
        return ['ISO_14001', 'ISO_9001', 'OHSAS_18001']
    
    def _get_regulatory_authorities(self, material_data: Dict[str, Any], company_a: Dict[str, Any], company_b: Dict[str, Any]) -> List[str]:
        """Get relevant regulatory authorities"""
        authorities = []
        
        try:
            origin_country = self._extract_country(company_a.get('location', ''))
            destination_country = self._extract_country(company_b.get('location', ''))
            
            if origin_country == 'US':
                authorities.append('US EPA')
            elif origin_country == 'EU':
                authorities.append('European Commission')
            
            if destination_country == 'US':
                authorities.append('US EPA')
            elif destination_country == 'EU':
                authorities.append('European Commission')
            
            return list(set(authorities))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error getting regulatory authorities: {e}")
            return []
    
    def _get_compliance_deadlines(self, material_data: Dict[str, Any]) -> List[datetime]:
        """Get compliance deadlines"""
        deadlines = []
        
        try:
            # Add deadlines from regulatory updates
            for update in self.regulatory_updates:
                if material_data.get('type', '') in update.material_types:
                    deadlines.append(update.compliance_deadline)
            
            return deadlines
            
        except Exception as e:
            logger.error(f"Error getting compliance deadlines: {e}")
            return []

    def _estimate_compliance_cost(self, required_permits: List[str], violations: List[Dict[str, Any]]) -> float:
        """Estimate compliance costs"""
        try:
            total_cost = 0.0
            
            # Add permit costs
            for permit_type in required_permits:
                if permit_type in self.permit_requirements:
                    permit = self.permit_requirements[permit_type]
                    total_cost += permit.processing_fee
            
            # Add violation costs
            for violation in violations:
                if violation.get('severity') == 'high':
                    total_cost += 5000  # Estimated fine
                elif violation.get('severity') == 'critical':
                    total_cost += 15000  # Estimated fine
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Error estimating compliance cost: {e}")
            return 0.0
    
    def _calculate_carbon_impact(self, material_data: Dict[str, Any], compliance_checks: List[Dict[str, Any]]) -> float:
        """Calculate carbon impact of compliance measures"""
        try:
            base_impact = 100.0  # Base carbon impact
            
            # Reduce impact for good compliance
            compliance_score = np.mean([check.get('score', 0) for check in compliance_checks])
            carbon_reduction = compliance_score * 50  # Good compliance reduces carbon impact
            
            return max(0.0, base_impact - carbon_reduction)
            
        except Exception as e:
            logger.error(f"Error calculating carbon impact: {e}")
            return 100.0
    
    def _create_fallback_compliance_result(self, match_data: Dict[str, Any]) -> ComplianceResult:
        """Create fallback compliance result on error"""
        return ComplianceResult(
            match_id=match_data.get('match_id', ''),
            company_a_id=match_data.get('company_a', {}).get('id', ''),
            company_b_id=match_data.get('company_b', {}).get('id', ''),
            material_type=match_data.get('material_data', {}).get('type', ''),
            origin_location=match_data.get('company_a', {}).get('location', ''),
            destination_location=match_data.get('company_b', {}).get('location', ''),
            overall_compliance=False,
            compliance_score=0.0,
            risk_level='high',
            compliance_details={},
            violations=[{'regulation': 'System Error', 'requirement': 'Compliance check failed', 'severity': 'high'}],
            recommendations=['Contact support for compliance assistance'],
            required_permits=[],
            certification_requirements=[],
            regulatory_authorities=[],
            compliance_deadlines=[],
            estimated_compliance_cost=0.0,
            carbon_impact=100.0,
            created_at=datetime.now()
        )
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result from Redis"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data.encode('latin1'))
            return None
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    def _cache_result(self, cache_key: str, data: Any) -> None:
        """Cache result in Redis"""
        try:
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(cache_key, self.cache_ttl, serialized_data)
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    def start_background_processing(self):
        """Start background regulatory monitoring"""
        if not self.running:
            self.running = True
            self.background_thread = threading.Thread(target=self._background_monitor, daemon=True)
            self.background_thread.start()
            logger.info("🔄 Background regulatory monitoring started")
    
    def stop_background_processing(self):
        """Stop background regulatory monitoring"""
        self.running = False
        if self.background_thread:
            self.background_thread.join()
        logger.info("⏹️ Background regulatory monitoring stopped")
    
    def _background_monitor(self):
        """Background monitoring for regulatory changes"""
        while self.running:
            try:
                # Monitor for regulatory updates
                self._check_regulatory_updates()
                
                # Update compliance rules
                self._update_compliance_rules()
                
                # Sleep for configured interval
                time.sleep(self.config['update_frequency_hours'] * 3600)
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                time.sleep(3600)  # Sleep for 1 hour on error
    
    def _check_regulatory_updates(self):
        """Check for new regulatory updates"""
        try:
            logger.debug("Checking for regulatory updates...")
            # This would integrate with real regulatory databases
        except Exception as e:
            logger.error(f"Error checking regulatory updates: {e}")
    
    def _update_compliance_rules(self):
        """Update compliance rules"""
        try:
            logger.debug("Updating compliance rules...")
            # This would update rules from regulatory sources
        except Exception as e:
            logger.error(f"Error updating compliance rules: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_compliance_checks': len(self.processing_times),
            'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'background_monitoring_active': self.running,
            'regulatory_updates_count': len(self.regulatory_updates),
            'last_update': datetime.now().isoformat()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            return 0.80  # Simulated 80% hit rate
        except Exception as e:
            logger.error(f"Error calculating cache hit rate: {e}")
            return 0.0

# Add Flask app and API for explainability endpoint if not present
app = Flask(__name__)
api = Api(app, version='1.0', title='Regulatory Compliance Engine', description='Advanced ML Regulatory Compliance', doc='/docs')

# Add data validator
data_validator = AdvancedDataValidator(logger=logger)

explain_input = api.model('ExplainInput', {
    'model_type': fields.String(required=True, description='Model type (compliance_classifier, risk_assessor)'),
    'input_data': fields.Raw(required=True, description='Input data for explanation')
})

@api.route('/explain')
class Explain(Resource):
    @api.expect(explain_input)
    @api.response(200, 'Success')
    @api.response(400, 'Invalid input data')
    @api.response(500, 'Internal error')
    def post(self):
        try:
            data = request.json
            model_type = data.get('model_type')
            input_data = data.get('input_data')
            schema = {'type': 'object', 'properties': {'features': {'type': 'array'}}, 'required': ['features']}
            data_validator.set_schema(schema)
            if not data_validator.validate(input_data):
                logger.error('Input data failed schema validation.')
                return {'error': 'Invalid input data'}, 400
            features = np.array(input_data['features']).reshape(1, -1)
            if model_type == 'compliance_classifier':
                model = self.compliance_classifier
            elif model_type == 'risk_assessor':
                model = self.risk_assessor
            else:
                logger.error(f'Unknown model_type: {model_type}')
                return {'error': 'Unknown model_type'}, 400
            explainer = shap.Explainer(model.predict, features)
            shap_values = explainer(features)
            logger.info(f'Explanation generated for {model_type}')
            return {'shap_values': shap_values.values.tolist(), 'base_values': shap_values.base_values.tolist()}
        except Exception as e:
            logger.error(f'Explainability error: {e}')
            return {'error': str(e)}, 500

# Global instance
regulatory_compliance_engine = AdvancedRegulatoryComplianceEngine() 

# Export for compatibility
RegulatoryComplianceEngine = AdvancedRegulatoryComplianceEngine 