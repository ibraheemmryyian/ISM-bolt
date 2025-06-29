#!/usr/bin/env python3
"""
Comprehensive AI Testing & Improvement Suite
Analyzes every aspect of the onboarding AI's responses and provides detailed improvement insights
"""

import json
import sys
import os
import time
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, Counter
import statistics

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_onboarding_ai import AdvancedOnboardingAI

@dataclass
class DetailedTestResult:
    """Comprehensive test result with detailed analysis"""
    company_id: str
    company_name: str
    industry: str
    location: str
    
    # Basic metrics
    success: bool
    execution_time: float
    listings_count: int
    input_listings: int
    waste_listings: int
    product_listings: int
    
    # Quality metrics
    coverage_score: float
    quality_score: float
    relevance_score: float
    specificity_score: float
    
    # Detailed analysis
    issues: List[str]
    warnings: List[str]
    strengths: List[str]
    
    # Content analysis
    empty_fields: List[str]
    generic_responses: List[str]
    hallucinated_content: List[str]
    industry_specific_content: List[str]
    
    # Material flow analysis
    material_coverage: Dict[str, float]
    process_coverage: Dict[str, float]
    waste_identification: Dict[str, float]
    
    # Raw data
    raw_output: List[Dict]
    input_data: Dict

class AIQualityAnalyzer:
    """Analyzes the quality and relevance of AI responses"""
    
    def __init__(self):
        self.generic_phrases = [
            "sustainable", "environmentally friendly", "eco-friendly", "green", 
            "recyclable", "renewable", "organic", "natural", "efficient",
            "high quality", "premium", "excellent", "superior", "advanced"
        ]
        
        self.industry_keywords = {
            "Furniture Production": ["wood", "plywood", "steel", "foam", "upholstery", "furniture", "chair", "table"],
            "Electronics Manufacturing": ["silicon", "battery", "copper", "pcb", "electronics", "smartphone", "tablet"],
            "Hospital": ["medical", "sterile", "patient", "diagnosis", "treatment", "hospital", "healthcare"],
            "Supermarket": ["produce", "packaging", "retail", "inventory", "checkout", "supermarket", "food"],
            "Plastic Recycling": ["plastic", "hdp", "ldp", "pellet", "recycling", "post-consumer"],
            "Water Treatment": ["water", "coagulant", "filtration", "disinfection", "sludge", "treatment"],
            "Metal Manufacturing": ["steel", "aluminum", "metal", "furnace", "casting", "rolling"],
            "Chemical Production": ["chemical", "solvent", "catalyst", "distillation", "reaction"],
            "Packaging Manufacturing": ["packaging", "pet", "injection", "molding", "container"],
            "Electronics Recycling": ["e-waste", "battery", "recycling", "metal recovery", "electronics"],
            "Food Production": ["food", "vegetable", "fruit", "canning", "processing", "organic"],
            "Textile Manufacturing": ["textile", "cotton", "fabric", "yarn", "dyeing", "weaving"],
            "Building Materials": ["concrete", "aggregate", "building", "construction", "material"],
            "Auto Parts Manufacturing": ["automotive", "vehicle", "transmission", "wheel", "parts"],
            "Transportation": ["transport", "logistics", "shipping", "cargo", "container"],
            "Renewable Energy": ["energy", "biogas", "digestion", "renewable", "electricity"]
        }
    
    def analyze_response_quality(self, company_data: Dict, listings: List) -> Dict:
        """Analyze the quality and relevance of AI responses"""
        analysis = {
            'empty_fields': [],
            'generic_responses': [],
            'hallucinated_content': [],
            'industry_specific_content': [],
            'material_coverage': {},
            'process_coverage': {},
            'waste_identification': {},
            'relevance_score': 0.0,
            'specificity_score': 0.0
        }
        
        # Analyze each listing
        for listing in listings:
            # Check for empty fields
            if not listing.description or listing.description.strip() == "":
                analysis['empty_fields'].append(f"Empty description for {listing.material_name}")
            
            if not listing.reasoning or listing.reasoning.strip() == "":
                analysis['empty_fields'].append(f"Empty reasoning for {listing.material_name}")
            
            # Check for generic responses
            if self._is_generic_response(listing.description):
                analysis['generic_responses'].append(f"Generic description: {listing.material_name}")
            
            if self._is_generic_response(listing.reasoning):
                analysis['generic_responses'].append(f"Generic reasoning: {listing.material_name}")
            
            # Check for industry-specific content
            if self._is_industry_specific(listing.description, company_data['industry']):
                analysis['industry_specific_content'].append(f"Industry-specific: {listing.material_name}")
            
            if self._is_industry_specific(listing.reasoning, company_data['industry']):
                analysis['industry_specific_content'].append(f"Industry-specific reasoning: {listing.material_name}")
        
        # Calculate coverage scores
        analysis['material_coverage'] = self._analyze_material_coverage(company_data, listings)
        analysis['process_coverage'] = self._analyze_process_coverage(company_data, listings)
        analysis['waste_identification'] = self._analyze_waste_identification(company_data, listings)
        
        # Calculate overall scores
        analysis['relevance_score'] = self._calculate_relevance_score(analysis, company_data)
        analysis['specificity_score'] = self._calculate_specificity_score(analysis, company_data)
        
        return analysis
    
    def _is_generic_response(self, text: str) -> bool:
        """Check if response is generic"""
        if not text:
            return False
        
        text_lower = text.lower()
        generic_count = sum(1 for phrase in self.generic_phrases if phrase in text_lower)
        return generic_count >= 2  # If 2+ generic phrases, consider it generic
    
    def _is_industry_specific(self, text: str, industry: str) -> bool:
        """Check if response is industry-specific"""
        if not text or industry not in self.industry_keywords:
            return False
        
        text_lower = text.lower()
        keywords = self.industry_keywords[industry]
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    def _analyze_material_coverage(self, company_data: Dict, listings: List) -> Dict[str, float]:
        """Analyze how well the AI covers the company's materials"""
        materials = company_data.get('materials', [])
        if isinstance(materials, str):
            materials = [m.strip() for m in materials.split(',')]
        
        coverage = {}
        for material in materials:
            material_lower = material.lower()
            covered = any(material_lower in listing.material_name.lower() or 
                         material_lower in listing.description.lower() 
                         for listing in listings)
            coverage[material] = 1.0 if covered else 0.0
        
        return coverage
    
    def _analyze_process_coverage(self, company_data: Dict, listings: List) -> Dict[str, float]:
        """Analyze how well the AI covers the company's processes"""
        processes = company_data.get('processes', '')
        process_steps = [p.strip() for p in processes.split('→')]
        
        coverage = {}
        for step in process_steps:
            step_lower = step.lower()
            covered = any(step_lower in listing.description.lower() or 
                         step_lower in listing.reasoning.lower() 
                         for listing in listings)
            coverage[step] = 1.0 if covered else 0.0
        
        return coverage
    
    def _analyze_waste_identification(self, company_data: Dict, listings: List) -> Dict[str, float]:
        """Analyze waste identification quality"""
        waste_listings = [l for l in listings if l.material_type == 'waste']
        
        analysis = {
            'waste_count': len(waste_listings),
            'expected_waste_types': 0,
            'waste_specificity': 0.0,
            'waste_quantification': 0.0
        }
        
        # Industry-specific waste expectations
        industry = company_data['industry']
        expected_wastes = self._get_expected_wastes(industry)
        analysis['expected_waste_types'] = len(expected_wastes)
        
        # Check waste specificity and quantification
        specific_wastes = 0
        quantified_wastes = 0
        
        for waste in waste_listings:
            if waste.description and len(waste.description) > 50:
                specific_wastes += 1
            if waste.quantity and waste.quantity > 0:
                quantified_wastes += 1
        
        if waste_listings:
            analysis['waste_specificity'] = specific_wastes / len(waste_listings)
            analysis['waste_quantification'] = quantified_wastes / len(waste_listings)
        
        return analysis
    
    def _get_expected_wastes(self, industry: str) -> List[str]:
        """Get expected waste types for each industry"""
        waste_map = {
            "Furniture Production": ["wood scraps", "sawdust", "metal shavings", "fabric scraps"],
            "Electronics Manufacturing": ["e-waste", "chemical waste", "metal scraps", "packaging waste"],
            "Hospital": ["medical waste", "biological waste", "chemical waste", "packaging waste"],
            "Supermarket": ["organic waste", "packaging waste", "expired food", "cardboard"],
            "Plastic Recycling": ["contaminated plastics", "processing waste", "water waste"],
            "Water Treatment": ["sludge", "chemical waste", "filter waste"],
            "Metal Manufacturing": ["metal scraps", "slag", "dust", "chemical waste"],
            "Chemical Production": ["chemical waste", "catalyst waste", "packaging waste"],
            "Food Production": ["organic waste", "packaging waste", "processing waste"],
            "Textile Manufacturing": ["fabric scraps", "dye waste", "water waste"]
        }
        return waste_map.get(industry, [])
    
    def _calculate_relevance_score(self, analysis: Dict, company_data: Dict) -> float:
        """Calculate overall relevance score"""
        scores = []
        
        # Material coverage
        material_coverage = analysis['material_coverage']
        if material_coverage:
            scores.append(sum(material_coverage.values()) / len(material_coverage))
        
        # Process coverage
        process_coverage = analysis['process_coverage']
        if process_coverage:
            scores.append(sum(process_coverage.values()) / len(process_coverage))
        
        # Industry-specific content
        industry_content = len(analysis['industry_specific_content'])
        total_listings = len(analysis['empty_fields']) + len(analysis['generic_responses']) + industry_content
        if total_listings > 0:
            scores.append(industry_content / total_listings)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _calculate_specificity_score(self, analysis: Dict, company_data: Dict) -> float:
        """Calculate specificity score"""
        scores = []
        
        # Generic responses penalty
        generic_penalty = len(analysis['generic_responses']) * 0.1
        scores.append(max(0, 1.0 - generic_penalty))
        
        # Empty fields penalty
        empty_penalty = len(analysis['empty_fields']) * 0.05
        scores.append(max(0, 1.0 - empty_penalty))
        
        # Industry-specific content bonus
        industry_bonus = len(analysis['industry_specific_content']) * 0.05
        scores.append(min(1.0, 0.5 + industry_bonus))
        
        return statistics.mean(scores) if scores else 0.0

class ComprehensiveAITestSuite:
    """Comprehensive test suite for onboarding AI with detailed analysis"""
    
    def __init__(self):
        self.ai = AdvancedOnboardingAI()
        self.analyzer = AIQualityAnalyzer()
        self.companies = self._load_companies()
        
    def _load_companies(self) -> List[Dict]:
        """Load companies from JSON file"""
        try:
            with open('companies.json', 'r') as f:
                data = json.load(f)
                return data.get('companies', [])
        except Exception as e:
            print(f"Error loading companies.json: {e}")
            return []
    
    def _convert_company_format(self, company: Dict) -> Dict:
        """Convert company data to the format expected by the AI"""
        return {
            "name": company['name'],
            "industry": company['industry'],
            "products": ", ".join(company['products']) if isinstance(company['products'], list) else company['products'],
            "location": company['location'],
            "productionVolume": company['volume'],
            "mainMaterials": ", ".join(company['materials']) if isinstance(company['materials'], list) else company['materials'],
            "processDescription": company['processes']
        }
    
    def test_company(self, company: Dict) -> DetailedTestResult:
        """Test a single company with comprehensive analysis"""
        start_time = time.time()
        issues = []
        warnings = []
        strengths = []
        
        try:
            # Convert company format
            company_data = self._convert_company_format(company)
            
            # Generate listings
            listings = self.ai.generate_advanced_listings(company_data)
            execution_time = time.time() - start_time
            
            # Analyze results
            input_listings = [l for l in listings if l.material_type == 'input']
            waste_listings = [l for l in listings if l.material_type == 'waste']
            product_listings = [l for l in listings if l.material_type == 'product']
            
            # Quality analysis
            quality_analysis = self.analyzer.analyze_response_quality(company_data, listings)
            
            # Check for issues
            if len(listings) == 0:
                issues.append("No listings generated")
            
            if len(input_listings) == 0:
                issues.append("No input/requirement listings")
            
            if len(waste_listings) == 0:
                issues.append("No waste listings")
            
            if len(product_listings) == 0:
                issues.append("No product listings")
            
            # Identify strengths
            if quality_analysis['relevance_score'] > 0.7:
                strengths.append("High relevance to industry")
            
            if quality_analysis['specificity_score'] > 0.7:
                strengths.append("High specificity in responses")
            
            if len(quality_analysis['industry_specific_content']) > 0:
                strengths.append("Industry-specific content identified")
            
            # Calculate scores
            coverage_score = min(len(listings) / 10, 1.0)
            quality_score = 1.0 - (len(issues) / max(len(listings), 1))
            
            # Convert to dict for JSON output
            raw_output = []
            for listing in listings:
                raw_output.append({
                    'name': listing.material_name,
                    'type': listing.material_type,
                    'quantity': listing.quantity,
                    'unit': listing.unit,
                    'description': listing.description,
                    'confidence_score': listing.confidence_score,
                    'reasoning': listing.reasoning,
                    'industry_relevance': listing.industry_relevance,
                    'sustainability_impact': listing.sustainability_impact,
                    'market_demand': listing.market_demand,
                    'regulatory_compliance': listing.regulatory_compliance,
                    'ai_generated': listing.ai_generated
                })
            
            return DetailedTestResult(
                company_id=company['id'],
                company_name=company['name'],
                industry=company['industry'],
                location=company['location'],
                success=len(issues) == 0,
                execution_time=execution_time,
                listings_count=len(listings),
                input_listings=len(input_listings),
                waste_listings=len(waste_listings),
                product_listings=len(product_listings),
                coverage_score=coverage_score,
                quality_score=quality_score,
                relevance_score=quality_analysis['relevance_score'],
                specificity_score=quality_analysis['specificity_score'],
                issues=issues,
                warnings=warnings,
                strengths=strengths,
                empty_fields=quality_analysis['empty_fields'],
                generic_responses=quality_analysis['generic_responses'],
                hallucinated_content=quality_analysis['hallucinated_content'],
                industry_specific_content=quality_analysis['industry_specific_content'],
                material_coverage=quality_analysis['material_coverage'],
                process_coverage=quality_analysis['process_coverage'],
                waste_identification=quality_analysis['waste_identification'],
                raw_output=raw_output,
                input_data=company_data
            )
            
        except Exception as e:
            return DetailedTestResult(
                company_id=company['id'],
                company_name=company['name'],
                industry=company['industry'],
                location=company['location'],
                success=False,
                execution_time=time.time() - start_time,
                listings_count=0,
                input_listings=0,
                waste_listings=0,
                product_listings=0,
                coverage_score=0.0,
                quality_score=0.0,
                relevance_score=0.0,
                specificity_score=0.0,
                issues=[f"Exception: {str(e)}"],
                warnings=[],
                strengths=[],
                empty_fields=[],
                generic_responses=[],
                hallucinated_content=[],
                industry_specific_content=[],
                material_coverage={},
                process_coverage={},
                waste_identification={},
                raw_output=[],
                input_data=self._convert_company_format(company)
            )
    
    def run_comprehensive_test_suite(self, max_companies: int = None) -> Dict:
        """Run comprehensive tests on all companies"""
        print("🧪 Running Comprehensive AI Testing & Improvement Suite")
        print("=" * 80)
        
        companies_to_test = self.companies[:max_companies] if max_companies else self.companies
        total_companies = len(companies_to_test)
        
        print(f"📊 Testing {total_companies} companies...")
        print(f"🏭 Industries covered: {len(set(c['industry'] for c in companies_to_test))}")
        print(f"🌍 Locations covered: {len(set(c['location'] for c in companies_to_test))}")
        print()
        
        results = []
        industry_analysis = defaultdict(lambda: {'count': 0, 'success': 0, 'scores': []})
        total_issues = []
        total_warnings = []
        total_strengths = []
        
        for i, company in enumerate(companies_to_test, 1):
            print(f"📋 Testing {i}/{total_companies}: {company['name']} ({company['industry']})")
            result = self.test_company(company)
            results.append(result)
            
            # Track industry performance
            industry = company['industry']
            industry_analysis[industry]['count'] += 1
            industry_analysis[industry]['success'] += 1 if result.success else 0
            industry_analysis[industry]['scores'].extend([
                result.coverage_score, 
                result.quality_score, 
                result.relevance_score, 
                result.specificity_score
            ])
            
            # Collect all issues and insights
            total_issues.extend(result.issues)
            total_warnings.extend(result.warnings)
            total_strengths.extend(result.strengths)
            
            # Print summary
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"   {status} - {result.listings_count} listings, {len(result.issues)} issues")
            print(f"   ⏱️  {result.execution_time:.2f}s | 📊 Coverage: {result.coverage_score:.2f} | Quality: {result.quality_score:.2f}")
            print(f"   🎯 Relevance: {result.relevance_score:.2f} | Specificity: {result.specificity_score:.2f}")
            
            if result.issues:
                print(f"   ⚠️  Issues: {', '.join(result.issues[:3])}")
            if result.strengths:
                print(f"   💪 Strengths: {', '.join(result.strengths[:2])}")
            print()
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(
            results, industry_analysis, total_issues, total_warnings, total_strengths
        )
        
        # Print summary
        self._print_comprehensive_summary(report)
        
        return report
    
    def _generate_comprehensive_report(self, results: List[DetailedTestResult], 
                                     industry_analysis: Dict, total_issues: List[str],
                                     total_warnings: List[str], total_strengths: List[str]) -> Dict:
        """Generate comprehensive analysis report"""
        
        # Calculate overall metrics
        total_companies = len(results)
        successful_companies = sum(1 for r in results if r.success)
        avg_scores = {
            'coverage': statistics.mean([r.coverage_score for r in results]),
            'quality': statistics.mean([r.quality_score for r in results]),
            'relevance': statistics.mean([r.relevance_score for r in results]),
            'specificity': statistics.mean([r.specificity_score for r in results]),
            'execution_time': statistics.mean([r.execution_time for r in results])
        }
        
        # Industry performance analysis
        industry_performance = {}
        for industry, data in industry_analysis.items():
            success_rate = data['success'] / data['count']
            avg_score = statistics.mean(data['scores']) if data['scores'] else 0
            industry_performance[industry] = {
                'count': data['count'],
                'success_rate': success_rate,
                'avg_score': avg_score,
                'performance_level': self._get_performance_level(avg_score)
            }
        
        # Issue analysis
        issue_analysis = self._analyze_issues(total_issues)
        
        # Improvement recommendations
        recommendations = self._generate_improvement_recommendations(
            results, industry_performance, issue_analysis
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_companies': total_companies,
                'successful_companies': successful_companies,
                'success_rate': successful_companies / total_companies,
                'avg_scores': avg_scores,
                'total_listings_generated': sum(r.listings_count for r in results),
                'avg_listings_per_company': sum(r.listings_count for r in results) / total_companies,
                'total_execution_time': sum(r.execution_time for r in results),
                'avg_execution_time': avg_scores['execution_time']
            },
            'industry_performance': industry_performance,
            'issue_analysis': issue_analysis,
            'strengths_analysis': self._analyze_strengths(total_strengths),
            'recommendations': recommendations,
            'detailed_results': [asdict(r) for r in results],
            'learning_insights': self._generate_learning_insights(results, industry_performance)
        }
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level based on score"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _analyze_issues(self, issues: List[str]) -> Dict:
        """Analyze common issues and patterns"""
        issue_counts = Counter(issues)
        
        # Categorize issues
        categories = {
            'coverage': [i for i in issues if 'no listings' in i.lower() or 'no ' in i.lower()],
            'quality': [i for i in issues if 'empty' in i.lower() or 'generic' in i.lower()],
            'execution': [i for i in issues if 'exception' in i.lower() or 'error' in i.lower()],
            'relevance': [i for i in issues if 'irrelevant' in i.lower() or 'unrelated' in i.lower()]
        }
        
        return {
            'total_issues': len(issues),
            'unique_issues': len(issue_counts),
            'most_common_issues': issue_counts.most_common(10),
            'issue_categories': {k: len(v) for k, v in categories.items()},
            'issue_distribution': dict(issue_counts)
        }
    
    def _analyze_strengths(self, strengths: List[str]) -> Dict:
        """Analyze AI strengths"""
        strength_counts = Counter(strengths)
        return {
            'total_strengths': len(strengths),
            'unique_strengths': len(strength_counts),
            'most_common_strengths': strength_counts.most_common(10),
            'strength_distribution': dict(strength_counts)
        }
    
    def _generate_improvement_recommendations(self, results: List[DetailedTestResult],
                                           industry_performance: Dict, issue_analysis: Dict) -> List[str]:
        """Generate specific improvement recommendations"""
        recommendations = []
        
        # Coverage improvements
        if issue_analysis['issue_categories']['coverage'] > 0:
            recommendations.append("Improve coverage by expanding industry-specific templates and rules")
        
        # Quality improvements
        if issue_analysis['issue_categories']['quality'] > 0:
            recommendations.append("Reduce generic responses by adding more specific industry knowledge")
            recommendations.append("Implement better validation to prevent empty field outputs")
        
        # Industry-specific improvements
        poor_performing_industries = [
            industry for industry, perf in industry_performance.items()
            if perf['performance_level'] in ['Poor', 'Fair']
        ]
        
        if poor_performing_industries:
            recommendations.append(f"Focus improvement efforts on: {', '.join(poor_performing_industries[:3])}")
        
        # Performance improvements
        avg_execution_time = statistics.mean([r.execution_time for r in results])
        if avg_execution_time > 5.0:
            recommendations.append("Optimize execution time by improving algorithm efficiency")
        
        return recommendations
    
    def _generate_learning_insights(self, results: List[DetailedTestResult],
                                  industry_performance: Dict) -> Dict:
        """Generate insights for AI learning and improvement"""
        
        # Best performing patterns
        excellent_results = [r for r in results if r.quality_score > 0.8]
        poor_results = [r for r in results if r.quality_score < 0.4]
        
        insights = {
            'best_practices': [],
            'common_failures': [],
            'industry_patterns': {},
            'data_quality_insights': []
        }
        
        # Analyze best practices
        if excellent_results:
            common_industries = Counter([r.industry for r in excellent_results])
            insights['best_practices'].append(f"Strong performance in: {', '.join(common_industries.most_common(3))}")
        
        # Analyze failures
        if poor_results:
            common_industries = Counter([r.industry for r in poor_results])
            insights['common_failures'].append(f"Needs improvement in: {', '.join(common_industries.most_common(3))}")
        
        # Industry patterns
        for industry, perf in industry_performance.items():
            insights['industry_patterns'][industry] = {
                'success_rate': perf['success_rate'],
                'avg_score': perf['avg_score'],
                'recommendation': self._get_industry_recommendation(perf)
            }
        
        return insights
    
    def _get_industry_recommendation(self, performance: Dict) -> str:
        """Get specific recommendation for industry"""
        if performance['avg_score'] < 0.4:
            return "Requires complete industry-specific module development"
        elif performance['avg_score'] < 0.6:
            return "Needs enhancement of existing industry rules and templates"
        elif performance['avg_score'] < 0.8:
            return "Minor improvements needed for consistency"
        else:
            return "Performing well, focus on edge cases"
    
    def _print_comprehensive_summary(self, report: Dict):
        """Print comprehensive test summary"""
        summary = report['summary']
        
        print(f"\n📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        print(f"Total Companies Tested: {summary['total_companies']}")
        print(f"Successful: {summary['successful_companies']}/{summary['total_companies']} ({summary['success_rate']*100:.1f}%)")
        print(f"Total Listings Generated: {summary['total_listings_generated']}")
        print(f"Average Listings per Company: {summary['avg_listings_per_company']:.1f}")
        print(f"Total Execution Time: {summary['total_execution_time']:.1f}s")
        print(f"Average Execution Time: {summary['avg_execution_time']:.2f}s")
        
        print(f"\n📈 PERFORMANCE METRICS")
        print("-" * 40)
        avg_scores = summary['avg_scores']
        print(f"Coverage Score: {avg_scores['coverage']:.3f}")
        print(f"Quality Score: {avg_scores['quality']:.3f}")
        print(f"Relevance Score: {avg_scores['relevance']:.3f}")
        print(f"Specificity Score: {avg_scores['specificity']:.3f}")
        
        print(f"\n🏭 INDUSTRY PERFORMANCE")
        print("-" * 40)
        for industry, perf in report['industry_performance'].items():
            print(f"{industry}: {perf['count']} companies, {perf['success_rate']*100:.1f}% success, {perf['avg_score']:.3f} avg score ({perf['performance_level']})")
        
        print(f"\n⚠️  TOP ISSUES")
        print("-" * 40)
        for issue, count in report['issue_analysis']['most_common_issues'][:5]:
            print(f"{count}x: {issue}")
        
        print(f"\n💪 TOP STRENGTHS")
        print("-" * 40)
        for strength, count in report['strengths_analysis']['most_common_strengths'][:5]:
            print(f"{count}x: {strength}")
        
        print(f"\n🎯 IMPROVEMENT RECOMMENDATIONS")
        print("-" * 40)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print(f"\n📄 Detailed reports saved to:")
        print(f"   - comprehensive_ai_test_report.json")
        print(f"   - ai_improvement_insights.json")

def main():
    """Run the comprehensive test suite"""
    test_suite = ComprehensiveAITestSuite()
    
    # Run tests (you can limit the number for faster testing)
    report = test_suite.run_comprehensive_test_suite(max_companies=20)  # Test first 20 companies
    
    # Save detailed report
    with open('comprehensive_ai_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save improvement insights
    insights = {
        'timestamp': datetime.now().isoformat(),
        'learning_insights': report['learning_insights'],
        'recommendations': report['recommendations'],
        'industry_performance': report['industry_performance']
    }
    
    with open('ai_improvement_insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print(f"\n✅ Test suite completed!")
    print(f"📊 Reports saved for analysis and improvement planning")

if __name__ == "__main__":
    main() 