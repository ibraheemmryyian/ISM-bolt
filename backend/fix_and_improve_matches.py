#!/usr/bin/env python3
"""
Comprehensive Match Quality Fix and Improvement Script
This script:
1. Fixes existing problematic matches (duplicates, generic names, etc.)
2. Generates new high-quality matches using the improved AI engine
3. Validates the results and provides comprehensive reporting
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import os
import sys
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_match_quality_fixer import ComprehensiveMatchQualityFixer
from improved_ai_matching_engine import ImprovedAIMatchingEngine
from match_quality_analyzer import MatchQualityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveMatchFixerAndImprover:
    """Comprehensive fixer and improver for match quality"""
    
    def __init__(self):
        self.logger = logger
        self.fixer = ComprehensiveMatchQualityFixer()
        self.improved_engine = ImprovedAIMatchingEngine()
        self.analyzer = MatchQualityAnalyzer()
        
        # Results tracking
        self.results = {
            'fix_results': {},
            'improvement_results': {},
            'validation_results': {},
            'overall_quality_score': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_comprehensive_fix_and_improvement(self) -> Dict:
        """Run the complete fix and improvement process"""
        self.logger.info("🚀 Starting comprehensive match quality fix and improvement process...")
        
        print("\n" + "="*80)
        print("🔧 COMPREHENSIVE MATCH QUALITY FIX AND IMPROVEMENT")
        print("="*80)
        
        # Step 1: Analyze current quality
        print("\n📊 STEP 1: ANALYZING CURRENT MATCH QUALITY")
        print("-" * 50)
        current_analysis = self.analyzer.generate_quality_report()
        
        # Step 2: Fix existing issues
        print("\n🔧 STEP 2: FIXING EXISTING ISSUES")
        print("-" * 50)
        fix_results = self.fixer.fix_all_issues()
        self.results['fix_results'] = fix_results
        
        if 'error' in fix_results:
            self.logger.error(f"❌ Fix failed: {fix_results['error']}")
            return self.results
        
        # Step 3: Validate fixes
        print("\n🔍 STEP 3: VALIDATING FIXES")
        print("-" * 50)
        fix_validation = self.fixer.validate_fixes()
        self.results['fix_validation'] = fix_validation
        
        # Step 4: Generate improved matches
        print("\n🚀 STEP 4: GENERATING IMPROVED MATCHES")
        print("-" * 50)
        improved_matches = self.improved_engine.generate_all_matches(max_matches_per_material=8)
        self.results['improvement_results'] = {
            'matches_generated': len(improved_matches),
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 5: Validate improved matches
        print("\n🔍 STEP 5: VALIDATING IMPROVED MATCHES")
        print("-" * 50)
        improvement_validation = self.improved_engine.validate_matches(improved_matches)
        self.results['improvement_validation'] = improvement_validation
        
        # Step 6: Final quality analysis
        print("\n📊 STEP 6: FINAL QUALITY ANALYSIS")
        print("-" * 50)
        final_analysis = self.analyzer.generate_quality_report()
        self.results['final_analysis'] = final_analysis
        
        # Step 7: Calculate overall improvement
        print("\n📈 STEP 7: CALCULATING OVERALL IMPROVEMENT")
        print("-" * 50)
        overall_improvement = self._calculate_overall_improvement(
            current_analysis, final_analysis, fix_results, improvement_validation
        )
        self.results['overall_improvement'] = overall_improvement
        
        # Step 8: Generate comprehensive report
        print("\n📋 STEP 8: GENERATING COMPREHENSIVE REPORT")
        print("-" * 50)
        self._generate_comprehensive_report()
        
        return self.results
    
    def _calculate_overall_improvement(self, current_analysis: Dict, final_analysis: Dict, 
                                     fix_results: Dict, improvement_validation: Dict) -> Dict:
        """Calculate overall improvement metrics"""
        
        # Extract quality scores
        current_quality = current_analysis.get('quality_metrics', {}).get('overall_quality', 0)
        final_quality = final_analysis.get('quality_metrics', {}).get('overall_quality', 0)
        
        # Calculate improvements
        quality_improvement = final_quality - current_quality
        quality_improvement_percentage = (quality_improvement / current_quality * 100) if current_quality > 0 else 0
        
        # Extract key metrics
        current_duplicates = current_analysis.get('consistency_analysis', {}).get('duplicate_matches', 0)
        final_duplicates = final_analysis.get('consistency_analysis', {}).get('duplicate_matches', 0)
        
        current_generic_companies = current_analysis.get('hallucination_analysis', {}).get('generic_companies', 0)
        final_generic_companies = final_analysis.get('hallucination_analysis', {}).get('generic_companies', 0)
        
        current_generic_materials = current_analysis.get('hallucination_analysis', {}).get('generic_materials', 0)
        final_generic_materials = final_analysis.get('hallucination_analysis', {}).get('generic_materials', 0)
        
        current_unique_companies = current_analysis.get('consistency_analysis', {}).get('unique_target_companies', 0)
        final_unique_companies = final_analysis.get('consistency_analysis', {}).get('unique_target_companies', 0)
        
        improvement_metrics = {
            'quality_score_improvement': quality_improvement,
            'quality_score_improvement_percentage': quality_improvement_percentage,
            'duplicates_removed': current_duplicates - final_duplicates,
            'generic_companies_removed': current_generic_companies - final_generic_companies,
            'generic_materials_removed': current_generic_materials - final_generic_materials,
            'unique_companies_increase': final_unique_companies - current_unique_companies,
            'fixes_applied': fix_results.get('fixes_applied', []),
            'improvement_validation': improvement_validation
        }
        
        return improvement_metrics
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive final report"""
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE MATCH QUALITY FIX AND IMPROVEMENT COMPLETED")
        print("="*80)
        
        # Summary of improvements
        improvement = self.results.get('overall_improvement', {})
        
        print(f"\n📊 QUALITY IMPROVEMENT SUMMARY:")
        print(f"   • Quality Score Improvement: {improvement.get('quality_score_improvement', 0):.1f} points")
        print(f"   • Quality Score Improvement: {improvement.get('quality_score_improvement_percentage', 0):.1f}%")
        print(f"   • Duplicates Removed: {improvement.get('duplicates_removed', 0):,}")
        print(f"   • Generic Companies Removed: {improvement.get('generic_companies_removed', 0)}")
        print(f"   • Generic Materials Removed: {improvement.get('generic_materials_removed', 0)}")
        print(f"   • Unique Companies Increase: {improvement.get('unique_companies_increase', 0)}")
        
        # Fixes applied
        fixes = improvement.get('fixes_applied', [])
        if fixes:
            print(f"\n🔧 FIXES APPLIED:")
            for fix in fixes:
                print(f"   • {fix}")
        
        # Validation results
        validation = improvement.get('improvement_validation', {})
        if validation:
            print(f"\n✅ VALIDATION RESULTS:")
            print(f"   • Total Matches: {validation.get('total_matches', 0):,}")
            print(f"   • Unique Target Companies: {validation.get('unique_target_companies', 0)}")
            print(f"   • Unique Target Materials: {validation.get('unique_target_materials', 0)}")
            print(f"   • Average Match Score: {validation.get('avg_match_score', 0):.3f}")
            print(f"   • No Duplicates: {validation.get('duplicate_check', False)}")
            print(f"   • No Generic Companies: {validation.get('generic_companies', 0) == 0}")
            print(f"   • No Generic Materials: {validation.get('generic_materials', 0) == 0}")
            print(f"   • Quality Assessment: {validation.get('quality_assessment', 'Unknown')}")
        
        # Score distribution
        score_dist = validation.get('score_distribution', {})
        if score_dist:
            print(f"\n📈 SCORE DISTRIBUTION:")
            print(f"   • Excellent (≥0.9): {score_dist.get('excellent', 0):,}")
            print(f"   • Good (0.7-0.9): {score_dist.get('good', 0):,}")
            print(f"   • Fair (0.5-0.7): {score_dist.get('fair', 0):,}")
            print(f"   • Poor (<0.5): {score_dist.get('poor', 0):,}")
        
        # Final recommendations
        print(f"\n💡 FINAL RECOMMENDATIONS:")
        final_quality = validation.get('overall_quality_score', 0)
        
        if final_quality >= 90:
            print("   • ✅ EXCELLENT: Match quality is now world-class")
            print("   • ✅ No further improvements needed")
        elif final_quality >= 70:
            print("   • ✅ GOOD: Match quality is significantly improved")
            print("   • 🔧 Consider fine-tuning for specific use cases")
        elif final_quality >= 50:
            print("   • ⚠️ FAIR: Match quality has improved but needs more work")
            print("   • 🔧 Consider additional data sources and validation")
        else:
            print("   • ❌ POOR: Match quality still needs significant improvement")
            print("   • 🔧 Review matching algorithms and data quality")
        
        print(f"\n📁 Results saved to: material_matches.csv")
        print(f"📊 Detailed results available in the returned data structure")
        
        print("\n" + "="*80)
    
    def save_detailed_report(self, filename: str = "match_quality_improvement_report.json"):
        """Save detailed report to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            self.logger.info(f"✅ Detailed report saved to {filename}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save detailed report: {e}")
    
    def get_quick_summary(self) -> Dict:
        """Get a quick summary of the results"""
        improvement = self.results.get('overall_improvement', {})
        validation = improvement.get('improvement_validation', {})
        
        return {
            'quality_improvement': improvement.get('quality_score_improvement_percentage', 0),
            'duplicates_removed': improvement.get('duplicates_removed', 0),
            'generic_issues_fixed': improvement.get('generic_companies_removed', 0) + improvement.get('generic_materials_removed', 0),
            'unique_companies': validation.get('unique_target_companies', 0),
            'avg_match_score': validation.get('avg_match_score', 0),
            'quality_assessment': validation.get('quality_assessment', 'Unknown'),
            'overall_quality_score': validation.get('overall_quality_score', 0)
        }

def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Match Quality Fix and Improvement Process")
    print("=" * 80)
    
    # Initialize the comprehensive fixer and improver
    fixer_improver = ComprehensiveMatchFixerAndImprover()
    
    try:
        # Run the complete process
        results = fixer_improver.run_comprehensive_fix_and_improvement()
        
        # Save detailed report
        fixer_improver.save_detailed_report()
        
        # Get quick summary
        summary = fixer_improver.get_quick_summary()
        
        print(f"\n🎯 QUICK SUMMARY:")
        print(f"   • Quality Improvement: {summary['quality_improvement']:.1f}%")
        print(f"   • Duplicates Removed: {summary['duplicates_removed']:,}")
        print(f"   • Generic Issues Fixed: {summary['generic_issues_fixed']}")
        print(f"   • Unique Companies: {summary['unique_companies']}")
        print(f"   • Average Match Score: {summary['avg_match_score']:.3f}")
        print(f"   • Final Quality: {summary['quality_assessment']} ({summary['overall_quality_score']}/100)")
        
        return results, summary
        
    except Exception as e:
        logger.error(f"❌ Process failed: {e}")
        print(f"\n❌ ERROR: {e}")
        return None, None

if __name__ == "__main__":
    results, summary = main()
    
    if results:
        print(f"\n✅ Process completed successfully!")
        print(f"📊 Check material_matches.csv for the improved data")
        print(f"📋 Check match_quality_improvement_report.json for detailed results")
    else:
        print(f"\n❌ Process failed. Check logs for details.") 