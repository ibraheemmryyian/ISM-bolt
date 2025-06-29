#!/usr/bin/env python3
"""
AI Testing & Improvement Suite
Comprehensive analysis of onboarding AI performance across multiple companies
"""

import json
import sys
import os
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter
import statistics

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_onboarding_ai import AdvancedOnboardingAI

@dataclass
class TestResult:
    """Detailed test result"""
    company_id: str
    company_name: str
    industry: str
    success: bool
    execution_time: float
    listings_count: int
    input_listings: int
    waste_listings: int
    product_listings: int
    coverage_score: float
    quality_score: float
    issues: List[str]
    strengths: List[str]
    raw_output: List[Dict]

class AITestSuite:
    """Comprehensive test suite for onboarding AI"""
    
    def __init__(self):
        self.ai = AdvancedOnboardingAI()
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
    
    def test_company(self, company: Dict) -> TestResult:
        """Test a single company and return detailed results"""
        start_time = time.time()
        issues = []
        strengths = []
        
        try:
            # Convert company format
            company_data = self._convert_company_format(company)
            
            # Generate listings
            listings = self.ai.generate_advanced_listings(company_data)
            execution_time = time.time() - start_time
            
            # Analyze results
            input_listings = [l for l in listings if l.material_type in ['input', 'requirement', 'raw_material', 'supplies', 'utility', 'equipment']]
            waste_listings = [l for l in listings if l.material_type in ['waste']]
            product_listings = [l for l in listings if l.material_type in ['product', 'output', 'main_product']]
            
            # Check for issues
            if len(listings) == 0:
                issues.append("No listings generated")
            
            if len(input_listings) == 0:
                issues.append("No input/requirement listings")
            
            if len(waste_listings) == 0:
                issues.append("No waste listings")
            
            if len(product_listings) == 0:
                issues.append("No product listings")
            
            # Check for empty fields
            for listing in listings:
                if not listing.description or listing.description.strip() == "":
                    issues.append(f"Empty description for {listing.material_name}")
                if not listing.reasoning or listing.reasoning.strip() == "":
                    issues.append(f"Empty reasoning for {listing.material_name}")
                if not listing.industry_relevance or listing.industry_relevance.strip() == "":
                    issues.append(f"Empty industry relevance for {listing.material_name}")
            
            # Identify strengths
            if len(listings) >= 10:
                strengths.append("Good coverage with many listings")
            
            if len(waste_listings) > 0:
                strengths.append("Successfully identified waste streams")
            
            if len(input_listings) > 0:
                strengths.append("Successfully identified input requirements")
            
            # Check for industry-specific content
            industry_keywords = self._get_industry_keywords(company['industry'])
            industry_specific_count = 0
            for listing in listings:
                if any(keyword.lower() in listing.description.lower() for keyword in industry_keywords):
                    industry_specific_count += 1
            
            if industry_specific_count > 0:
                strengths.append(f"Industry-specific content identified ({industry_specific_count} listings)")
            
            # Calculate scores
            coverage_score = min(len(listings) / 10, 1.0)  # Expect at least 10 listings
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
            
            return TestResult(
                company_id=company['id'],
                company_name=company['name'],
                industry=company['industry'],
                success=len(issues) == 0,
                execution_time=execution_time,
                listings_count=len(listings),
                input_listings=len(input_listings),
                waste_listings=len(waste_listings),
                product_listings=len(product_listings),
                coverage_score=coverage_score,
                quality_score=quality_score,
                issues=issues,
                strengths=strengths,
                raw_output=raw_output
            )
            
        except Exception as e:
            return TestResult(
                company_id=company['id'],
                company_name=company['name'],
                industry=company['industry'],
                success=False,
                execution_time=time.time() - start_time,
                listings_count=0,
                input_listings=0,
                waste_listings=0,
                product_listings=0,
                coverage_score=0.0,
                quality_score=0.0,
                issues=[f"Exception: {str(e)}"],
                strengths=[],
                raw_output=[]
            )
    
    def _get_industry_keywords(self, industry: str) -> List[str]:
        """Get industry-specific keywords for analysis"""
        keywords = {
            "Furniture Production": ["wood", "plywood", "steel", "foam", "upholstery", "furniture"],
            "Electronics Manufacturing": ["silicon", "battery", "copper", "pcb", "electronics"],
            "Hospital": ["medical", "sterile", "patient", "diagnosis", "treatment"],
            "Supermarket": ["produce", "packaging", "retail", "inventory", "food"],
            "Plastic Recycling": ["plastic", "hdp", "ldp", "pellet", "recycling"],
            "Water Treatment": ["water", "coagulant", "filtration", "disinfection", "sludge"],
            "Metal Manufacturing": ["steel", "aluminum", "metal", "furnace", "casting"],
            "Chemical Production": ["chemical", "solvent", "catalyst", "distillation"],
            "Food Production": ["food", "vegetable", "fruit", "canning", "processing"],
            "Textile Manufacturing": ["textile", "cotton", "fabric", "yarn", "dyeing"],
            "Building Materials": ["concrete", "aggregate", "building", "construction"],
            "Auto Parts Manufacturing": ["automotive", "vehicle", "transmission", "wheel"],
            "Transportation": ["transport", "logistics", "shipping", "cargo"],
            "Renewable Energy": ["energy", "biogas", "digestion", "renewable"]
        }
        return keywords.get(industry, [])
    
    def run_test_suite(self, max_companies: int = None) -> Dict:
        """Run comprehensive tests on companies"""
        print("🧪 Running AI Testing & Improvement Suite")
        print("=" * 60)
        
        companies_to_test = self.companies[:max_companies] if max_companies else self.companies
        total_companies = len(companies_to_test)
        
        print(f"📊 Testing {total_companies} companies...")
        print(f"🏭 Industries: {len(set(c['industry'] for c in companies_to_test))}")
        print()
        
        results = []
        industry_stats = defaultdict(lambda: {'count': 0, 'success': 0, 'scores': []})
        total_issues = []
        total_strengths = []
        
        for i, company in enumerate(companies_to_test, 1):
            print(f"📋 Testing {i}/{total_companies}: {company['name']} ({company['industry']})")
            result = self.test_company(company)
            results.append(result)
            
            # Track industry performance
            industry = company['industry']
            industry_stats[industry]['count'] += 1
            industry_stats[industry]['success'] += 1 if result.success else 0
            industry_stats[industry]['scores'].extend([result.coverage_score, result.quality_score])
            
            # Collect issues and strengths
            total_issues.extend(result.issues)
            total_strengths.extend(result.strengths)
            
            # Print summary
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"   {status} - {result.listings_count} listings, {len(result.issues)} issues")
            print(f"   ⏱️  {result.execution_time:.2f}s | 📊 Coverage: {result.coverage_score:.2f} | Quality: {result.quality_score:.2f}")
            
            if result.issues:
                print(f"   ⚠️  Issues: {', '.join(result.issues[:2])}")
            if result.strengths:
                print(f"   💪 Strengths: {', '.join(result.strengths[:2])}")
            print()
        
        # Generate report
        report = self._generate_report(results, industry_stats, total_issues, total_strengths)
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _generate_report(self, results: List[TestResult], industry_stats: Dict, 
                        total_issues: List[str], total_strengths: List[str]) -> Dict:
        """Generate comprehensive analysis report"""
        
        # Calculate overall metrics
        total_companies = len(results)
        successful_companies = sum(1 for r in results if r.success)
        avg_coverage = statistics.mean([r.coverage_score for r in results])
        avg_quality = statistics.mean([r.quality_score for r in results])
        avg_execution_time = statistics.mean([r.execution_time for r in results])
        total_listings = sum(r.listings_count for r in results)
        
        # Industry performance analysis
        industry_performance = {}
        for industry, stats in industry_stats.items():
            success_rate = stats['success'] / stats['count']
            avg_score = statistics.mean(stats['scores']) if stats['scores'] else 0
            industry_performance[industry] = {
                'count': stats['count'],
                'success_rate': success_rate,
                'avg_score': avg_score,
                'performance': 'Excellent' if avg_score >= 0.8 else 'Good' if avg_score >= 0.6 else 'Fair' if avg_score >= 0.4 else 'Poor'
            }
        
        # Issue analysis
        issue_counts = Counter(total_issues)
        strength_counts = Counter(total_strengths)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, industry_performance, issue_counts)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_companies': total_companies,
                'successful_companies': successful_companies,
                'success_rate': successful_companies / total_companies,
                'avg_coverage_score': avg_coverage,
                'avg_quality_score': avg_quality,
                'avg_execution_time': avg_execution_time,
                'total_listings_generated': total_listings,
                'avg_listings_per_company': total_listings / total_companies
            },
            'industry_performance': industry_performance,
            'issue_analysis': {
                'total_issues': len(total_issues),
                'unique_issues': len(issue_counts),
                'most_common_issues': issue_counts.most_common(10),
                'issue_distribution': dict(issue_counts)
            },
            'strengths_analysis': {
                'total_strengths': len(total_strengths),
                'unique_strengths': len(strength_counts),
                'most_common_strengths': strength_counts.most_common(10),
                'strength_distribution': dict(strength_counts)
            },
            'recommendations': recommendations,
            'detailed_results': [
                {
                    'company_id': r.company_id,
                    'company_name': r.company_name,
                    'industry': r.industry,
                    'success': r.success,
                    'listings_count': r.listings_count,
                    'input_listings': r.input_listings,
                    'waste_listings': r.waste_listings,
                    'product_listings': r.product_listings,
                    'coverage_score': r.coverage_score,
                    'quality_score': r.quality_score,
                    'execution_time': r.execution_time,
                    'issues': r.issues,
                    'strengths': r.strengths
                }
                for r in results
            ],
            'raw_outputs': {r.company_name: r.raw_output for r in results}
        }
    
    def _generate_recommendations(self, results: List[TestResult], 
                                industry_performance: Dict, issue_counts: Counter) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Coverage issues
        coverage_issues = sum(1 for issue, count in issue_counts.items() if 'no listings' in issue.lower())
        if coverage_issues > 0:
            recommendations.append("Improve coverage by expanding industry-specific templates")
        
        # Quality issues
        quality_issues = sum(1 for issue, count in issue_counts.items() if 'empty' in issue.lower())
        if quality_issues > 0:
            recommendations.append("Reduce empty fields by improving validation and content generation")
        
        # Industry-specific improvements
        poor_performing = [industry for industry, perf in industry_performance.items() 
                          if perf['performance'] in ['Poor', 'Fair']]
        if poor_performing:
            recommendations.append(f"Focus improvement efforts on: {', '.join(poor_performing[:3])}")
        
        # Performance improvements
        avg_time = statistics.mean([r.execution_time for r in results])
        if avg_time > 5.0:
            recommendations.append("Optimize execution time for better user experience")
        
        return recommendations
    
    def _print_summary(self, report: Dict):
        """Print comprehensive test summary"""
        summary = report['summary']
        
        print(f"\n📊 TEST SUITE SUMMARY")
        print("=" * 60)
        print(f"Total Companies: {summary['total_companies']}")
        print(f"Successful: {summary['successful_companies']}/{summary['total_companies']} ({summary['success_rate']*100:.1f}%)")
        print(f"Average Coverage Score: {summary['avg_coverage_score']:.3f}")
        print(f"Average Quality Score: {summary['avg_quality_score']:.3f}")
        print(f"Average Execution Time: {summary['avg_execution_time']:.2f}s")
        print(f"Total Listings Generated: {summary['total_listings_generated']}")
        print(f"Average Listings per Company: {summary['avg_listings_per_company']:.1f}")
        
        print(f"\n🏭 INDUSTRY PERFORMANCE")
        print("-" * 40)
        for industry, perf in report['industry_performance'].items():
            print(f"{industry}: {perf['count']} companies, {perf['success_rate']*100:.1f}% success, {perf['avg_score']:.3f} avg score ({perf['performance']})")
        
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
        print(f"   - ai_test_report.json")
        print(f"   - ai_improvement_insights.json")

def main():
    """Run the test suite"""
    test_suite = AITestSuite()
    
    # Run tests on all companies (remove max_companies parameter to test all)
    # For faster testing, you can add: max_companies=10
    report = test_suite.run_test_suite()  # Test all companies
    
    # Save detailed report
    with open('ai_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save improvement insights
    insights = {
        'timestamp': datetime.now().isoformat(),
        'recommendations': report['recommendations'],
        'industry_performance': report['industry_performance'],
        'issue_analysis': report['issue_analysis'],
        'strengths_analysis': report['strengths_analysis']
    }
    
    with open('ai_improvement_insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print(f"\n✅ Test suite completed!")
    print(f"📊 Reports saved for analysis and improvement planning")

if __name__ == "__main__":
    main()
