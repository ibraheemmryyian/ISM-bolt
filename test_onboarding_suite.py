#!/usr/bin/env python3
"""
Comprehensive Onboarding AI Test Suite
Tests the AI on multiple companies and provides detailed analysis
"""

import json
import sys
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_onboarding_ai import AdvancedOnboardingAI

@dataclass
class TestResult:
    """Structured test result"""
    company_name: str
    industry: str
    success: bool
    listings_count: int
    input_listings: int
    waste_listings: int
    product_listings: int
    coverage_score: float
    quality_score: float
    issues: List[str]
    raw_output: List[Dict]

class OnboardingAITestSuite:
    """Comprehensive test suite for onboarding AI"""
    
    def __init__(self):
        self.ai = AdvancedOnboardingAI()
        self.test_companies = self._load_test_companies()
        
    def _load_test_companies(self) -> List[Dict]:
        """Load all test companies"""
        return [
            {
                "name": "GreenSteel Dynamics",
                "industry": "Metal Manufacturing",
                "products": "Recycled steel beams, Aluminum extrusions",
                "location": "Essen, Germany",
                "productionVolume": "85,000 metric tons",
                "mainMaterials": "Scrap steel (60%), Aluminum ingots (25%), Ferroalloys (15%)",
                "processDescription": "Electric arc furnace melting → Continuous casting → Hot rolling → Surface treatment"
            },
            {
                "name": "NovaChem Solutions",
                "industry": "Chemical Production",
                "products": "Industrial solvents, Plasticizers, PH adjusters",
                "location": "Antwerp, Belgium",
                "productionVolume": "120,000 liters",
                "mainMaterials": "Waste glycerin, Ethylene oxide, Used catalysts",
                "processDescription": "Catalytic conversion → Distillation → Purification → Quality testing → Drum filling"
            },
            {
                "name": "EcoPack Innovations",
                "industry": "Packaging Manufacturing",
                "products": "Biodegradable food containers, Recycled PET clamshells",
                "location": "Barcelona, Spain",
                "productionVolume": "65 million units",
                "mainMaterials": "Post-consumer PET flakes, Corn starch, Plant-based polymers",
                "processDescription": "Material blending → Injection molding → UV printing → Quality inspection → Palletizing"
            },
            {
                "name": "VoltCycle Systems",
                "industry": "Electronics Recycling",
                "products": "Refurbished lithium batteries, Recovered cobalt",
                "location": "Gothenburg, Sweden",
                "productionVolume": "8,000 kg cobalt, 25,000 battery packs",
                "mainMaterials": "E-waste batteries, Laptop power supplies, EV battery packs",
                "processDescription": "Battery shredding → Hydrometallurgical processing → Metal recovery → Cell reassembly → Performance testing"
            },
            {
                "name": "TerraNourish Organics",
                "industry": "Food Production",
                "products": "Canned legumes, Vegetable broths, Fruit purees",
                "location": "Lyon, France",
                "productionVolume": "40,000 metric tons",
                "mainMaterials": "Organic vegetables (55%), Pulses (30%), Spices (15%)",
                "processDescription": "Washing → Blanching → Canning → Sterilization → Labeling → Case packing"
            },
            {
                "name": "CircuTex Mills",
                "industry": "Textile Manufacturing",
                "products": "Recycled cotton yarns, PET-based fabrics",
                "location": "Milan, Italy",
                "productionVolume": "3,200 metric tons fabric",
                "mainMaterials": "Post-industrial cotton scraps, Recycled PET bottles, Denim waste",
                "processDescription": "Fiber shredding → Carding → Spinning → Dyeing → Weaving → Finishing"
            },
            {
                "name": "ReCon Materials",
                "industry": "Building Materials",
                "products": "Recycled concrete aggregates, Composite decking",
                "location": "Rotterdam, Netherlands",
                "productionVolume": "150,000 cubic meters",
                "mainMaterials": "Demolition concrete (70%), Recycled plastics (25%), Wood chips (5%)",
                "processDescription": "Crushing → Screening → Material blending → Extrusion → Curing → Cutting"
            },
            {
                "name": "AutoCycle Components",
                "industry": "Auto Parts Manufacturing",
                "products": "Recycled aluminum wheels, Remanufactured transmissions",
                "location": "Stuttgart, Germany",
                "productionVolume": "500,000 units",
                "mainMaterials": "End-of-life vehicles, Scrap aluminum, Used transmissions",
                "processDescription": "Dismantling → Cleaning → Machining → Surface coating → Assembly → Testing"
            },
            {
                "name": "LoopLogistics Group",
                "industry": "Transportation",
                "products": "Return trip cargo space, Shared container services",
                "location": "Hamburg, Germany",
                "productionVolume": "12,000 shipments",
                "mainMaterials": "Diesel (90%), Packaging materials (10%)",
                "processDescription": "Route optimization → Container loading → Transportation → Unloading → Reverse logistics"
            },
            {
                "name": "BioWatt Energy",
                "industry": "Renewable Energy",
                "products": "Biogas, Organic fertilizer",
                "location": "Copenhagen, Denmark",
                "productionVolume": "25 GWh electricity",
                "mainMaterials": "Food waste (60%), Agricultural residues (30%), Sewage sludge (10%)",
                "processDescription": "Anaerobic digestion → Biogas purification → CHP generation → Digestate processing → Pelletizing"
            }
        ]
    
    def test_company(self, company_data: Dict) -> TestResult:
        """Test a single company and return detailed results"""
        issues = []
        
        try:
            # Generate listings
            listings = self.ai.generate_advanced_listings(company_data)
            
            # Analyze results
            input_listings = [l for l in listings if l.material_type == 'input']
            waste_listings = [l for l in listings if l.material_type == 'waste']
            product_listings = [l for l in listings if l.material_type == 'product']
            
            # Check for issues
            if len(listings) == 0:
                issues.append("No listings generated")
            
            if len(input_listings) == 0:
                issues.append("No input/requirement listings")
            
            if len(waste_listings) == 0:
                issues.append("No waste listings")
            
            # Check for empty fields
            for listing in listings:
                if not listing.description or listing.description.strip() == "":
                    issues.append(f"Empty description for {listing.material_name}")
                if not listing.reasoning or listing.reasoning.strip() == "":
                    issues.append(f"Empty reasoning for {listing.material_name}")
                if not listing.industry_relevance or listing.industry_relevance.strip() == "":
                    issues.append(f"Empty industry relevance for {listing.material_name}")
            
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
                company_name=company_data['name'],
                industry=company_data['industry'],
                success=len(issues) == 0,
                listings_count=len(listings),
                input_listings=len(input_listings),
                waste_listings=len(waste_listings),
                product_listings=len(product_listings),
                coverage_score=coverage_score,
                quality_score=quality_score,
                issues=issues,
                raw_output=raw_output
            )
            
        except Exception as e:
            return TestResult(
                company_name=company_data['name'],
                industry=company_data['industry'],
                success=False,
                listings_count=0,
                input_listings=0,
                waste_listings=0,
                product_listings=0,
                coverage_score=0.0,
                quality_score=0.0,
                issues=[f"Exception: {str(e)}"],
                raw_output=[]
            )
    
    def run_full_test_suite(self) -> Dict:
        """Run tests on all companies and generate comprehensive report"""
        print("🧪 Running Comprehensive Onboarding AI Test Suite")
        print("=" * 60)
        
        results = []
        industry_coverage = {}
        total_issues = []
        
        for i, company in enumerate(self.test_companies, 1):
            print(f"\n📋 Testing {i}/10: {company['name']} ({company['industry']})")
            result = self.test_company(company)
            results.append(result)
            
            # Track industry coverage
            industry = company['industry']
            if industry not in industry_coverage:
                industry_coverage[industry] = {'count': 0, 'success': 0, 'avg_score': 0}
            industry_coverage[industry]['count'] += 1
            industry_coverage[industry]['success'] += 1 if result.success else 0
            industry_coverage[industry]['avg_score'] += result.quality_score
            
            # Collect issues
            total_issues.extend(result.issues)
            
            # Print summary
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"   {status} - {result.listings_count} listings, {len(result.issues)} issues")
        
        # Calculate overall metrics
        total_companies = len(results)
        successful_companies = sum(1 for r in results if r.success)
        avg_coverage = sum(r.coverage_score for r in results) / total_companies
        avg_quality = sum(r.quality_score for r in results) / total_companies
        total_listings = sum(r.listings_count for r in results)
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_companies': total_companies,
                'successful_companies': successful_companies,
                'success_rate': successful_companies / total_companies,
                'avg_coverage_score': avg_coverage,
                'avg_quality_score': avg_quality,
                'total_listings_generated': total_listings,
                'avg_listings_per_company': total_listings / total_companies
            },
            'industry_coverage': industry_coverage,
            'common_issues': self._analyze_common_issues(total_issues),
            'detailed_results': [
                {
                    'company_name': r.company_name,
                    'industry': r.industry,
                    'success': r.success,
                    'listings_count': r.listings_count,
                    'input_listings': r.input_listings,
                    'waste_listings': r.waste_listings,
                    'product_listings': r.product_listings,
                    'coverage_score': r.coverage_score,
                    'quality_score': r.quality_score,
                    'issues': r.issues
                }
                for r in results
            ],
            'raw_outputs': {r.company_name: r.raw_output for r in results}
        }
        
        # Print summary
        print(f"\n📊 TEST SUITE SUMMARY")
        print("=" * 60)
        print(f"Total Companies: {total_companies}")
        print(f"Successful: {successful_companies}/{total_companies} ({successful_companies/total_companies*100:.1f}%)")
        print(f"Average Coverage Score: {avg_coverage:.2f}")
        print(f"Average Quality Score: {avg_quality:.2f}")
        print(f"Total Listings Generated: {total_listings}")
        print(f"Average Listings per Company: {total_listings/total_companies:.1f}")
        
        print(f"\n🏭 INDUSTRY COVERAGE")
        print("-" * 40)
        for industry, stats in industry_coverage.items():
            success_rate = stats['success'] / stats['count']
            avg_score = stats['avg_score'] / stats['count']
            print(f"{industry}: {stats['success']}/{stats['count']} ({success_rate*100:.1f}%) - Avg Score: {avg_score:.2f}")
        
        if total_issues:
            print(f"\n⚠️  COMMON ISSUES")
            print("-" * 40)
            issue_counts = {}
            for issue in total_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"{count}x: {issue}")
        
        return report
    
    def _analyze_common_issues(self, issues: List[str]) -> Dict[str, int]:
        """Analyze and count common issues"""
        issue_counts = {}
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))

def main():
    """Run the test suite"""
    test_suite = OnboardingAITestSuite()
    report = test_suite.run_full_test_suite()
    
    # Save detailed report
    with open('onboarding_ai_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: onboarding_ai_test_report.json")

if __name__ == "__main__":
    main() 