#!/usr/bin/env python3
"""
Match Quality Analyzer
Analyzes the quality, accuracy, and potential hallucination in AI-generated matches
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Any
import re

# Suppress warnings
warnings.filterwarnings('ignore')

class MatchQualityAnalyzer:
    """Analyzes the quality and accuracy of AI-generated matches"""
    
    def __init__(self, listings_path: str = "material_listings.csv", 
                 matches_path: str = "material_matches.csv"):
        self.listings_path = listings_path
        self.matches_path = matches_path
        self.listings_df = None
        self.matches_df = None
        self.quality_report = {}
        
    def load_data(self) -> bool:
        """Load and validate data"""
        try:
            print("📊 Loading data for quality analysis...")
            self.listings_df = pd.read_csv(self.listings_path)
            self.matches_df = pd.read_csv(self.matches_path)
            
            print(f"✅ Loaded {len(self.listings_df)} listings and {len(self.matches_df)} matches")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_missing_source_company_id(self) -> Dict[str, Any]:
        """Analyze the critical issue of missing source company IDs"""
        print("\n🔍 ANALYZING MISSING SOURCE COMPANY ID ISSUE:")
        
        # Check for missing source company IDs
        missing_source_ids = self.matches_df['source_company_id'].isna().sum()
        total_matches = len(self.matches_df)
        
        analysis = {
            'total_matches': total_matches,
            'missing_source_ids': missing_source_ids,
            'missing_percentage': (missing_source_ids / total_matches) * 100,
            'has_source_ids': total_matches - missing_source_ids,
            'critical_issue': missing_source_ids > 0
        }
        
        print(f"   • Total matches: {total_matches:,}")
        print(f"   • Missing source company IDs: {missing_source_ids:,} ({analysis['missing_percentage']:.1f}%)")
        print(f"   • Matches with source IDs: {analysis['has_source_ids']:,}")
        
        if missing_source_ids > 0:
            print("   ⚠️  CRITICAL ISSUE: Missing source company IDs make matches unreliable!")
            print("   🔧 This prevents proper validation of match quality")
        
        return analysis
    
    def analyze_match_consistency(self) -> Dict[str, Any]:
        """Analyze consistency patterns in matches"""
        print("\n🔄 ANALYZING MATCH CONSISTENCY:")
        
        # Check for duplicate matches
        duplicate_matches = self.matches_df.duplicated(subset=[
            'source_material_name', 'target_company_name', 'target_material_name'
        ]).sum()
        
        # Check for unrealistic match patterns
        same_company_matches = 0
        if 'source_company_id' in self.matches_df.columns and not self.matches_df['source_company_id'].isna().all():
            same_company_matches = len(self.matches_df[
                self.matches_df['source_company_id'] == self.matches_df['target_company_id']
            ])
        
        # Analyze match score distribution
        score_distribution = {
            'excellent': len(self.matches_df[self.matches_df['match_score'] >= 0.9]),
            'good': len(self.matches_df[(self.matches_df['match_score'] >= 0.7) & (self.matches_df['match_score'] < 0.9)]),
            'fair': len(self.matches_df[(self.matches_df['match_score'] >= 0.5) & (self.matches_df['match_score'] < 0.7)]),
            'poor': len(self.matches_df[self.matches_df['match_score'] < 0.5])
        }
        
        # Check for suspicious patterns
        suspicious_patterns = []
        
        # Check if all matches have same target company
        unique_target_companies = self.matches_df['target_company_name'].nunique()
        if unique_target_companies < 10:
            suspicious_patterns.append(f"Only {unique_target_companies} unique target companies (suspiciously low)")
        
        # Check for repetitive match scores
        score_counts = self.matches_df['match_score'].value_counts()
        repetitive_scores = score_counts[score_counts > 100].sum()
        if repetitive_scores > len(self.matches_df) * 0.5:
            suspicious_patterns.append("High number of repetitive match scores")
        
        # Check for unrealistic value patterns
        value_counts = self.matches_df['potential_value'].value_counts()
        repetitive_values = value_counts[value_counts > 50].sum()
        if repetitive_values > len(self.matches_df) * 0.3:
            suspicious_patterns.append("High number of repetitive potential values")
        
        analysis = {
            'duplicate_matches': duplicate_matches,
            'same_company_matches': same_company_matches,
            'score_distribution': score_distribution,
            'unique_target_companies': unique_target_companies,
            'suspicious_patterns': suspicious_patterns,
            'repetitive_scores': repetitive_scores,
            'repetitive_values': repetitive_values
        }
        
        print(f"   • Duplicate matches: {duplicate_matches:,}")
        print(f"   • Same company matches: {same_company_matches:,}")
        print(f"   • Unique target companies: {unique_target_companies}")
        print(f"   • Repetitive scores: {repetitive_scores:,}")
        print(f"   • Repetitive values: {repetitive_values:,}")
        
        if suspicious_patterns:
            print("   ⚠️  SUSPICIOUS PATTERNS DETECTED:")
            for pattern in suspicious_patterns:
                print(f"      - {pattern}")
        
        return analysis
    
    def analyze_hallucination_indicators(self) -> Dict[str, Any]:
        """Analyze potential hallucination indicators"""
        print("\n🤖 ANALYZING HALLUCINATION INDICATORS:")
        
        hallucination_indicators = []
        
        # Check for generic company names
        generic_companies = []
        for company in self.matches_df['target_company_name'].unique():
            if any(generic in company.lower() for generic in ['match company', 'company', 'target', 'generic']):
                generic_companies.append(company)
        
        if generic_companies:
            hallucination_indicators.append(f"Generic company names found: {len(generic_companies)}")
            print(f"   • Generic companies: {len(generic_companies)}")
            for company in generic_companies[:5]:  # Show first 5
                print(f"      - {company}")
        
        # Check for generic material names
        generic_materials = []
        for material in self.matches_df['target_material_name'].unique():
            if any(generic in material.lower() for generic in ['compatible', 'generic', 'material', 'waste']):
                generic_materials.append(material)
        
        if generic_materials:
            hallucination_indicators.append(f"Generic material names found: {len(generic_materials)}")
            print(f"   • Generic materials: {len(generic_materials)}")
            for material in generic_materials[:5]:  # Show first 5
                print(f"      - {material}")
        
        # Check for unrealistic match scores
        perfect_scores = len(self.matches_df[self.matches_df['match_score'] == 1.0])
        if perfect_scores > 0:
            hallucination_indicators.append(f"Perfect match scores found: {perfect_scores}")
            print(f"   • Perfect scores (1.0): {perfect_scores:,}")
        
        # Check for unrealistic value patterns
        value_std = self.matches_df['potential_value'].std()
        value_mean = self.matches_df['potential_value'].mean()
        if value_std < value_mean * 0.1:  # Very low variance
            hallucination_indicators.append("Unrealistic value variance (too uniform)")
            print(f"   • Value variance: {value_std:.2f} (mean: {value_mean:.2f})")
        
        # Check for repetitive patterns
        unique_combinations = len(self.matches_df.groupby([
            'source_material_name', 'target_company_name', 'target_material_name'
        ]))
        if unique_combinations < len(self.matches_df) * 0.8:
            hallucination_indicators.append("High repetition in match combinations")
            print(f"   • Unique combinations: {unique_combinations:,} / {len(self.matches_df):,}")
        
        analysis = {
            'generic_companies': len(generic_companies),
            'generic_materials': len(generic_materials),
            'perfect_scores': perfect_scores,
            'value_variance': value_std,
            'unique_combinations': unique_combinations,
            'hallucination_indicators': hallucination_indicators
        }
        
        if hallucination_indicators:
            print("   ⚠️  HALLUCINATION INDICATORS DETECTED:")
            for indicator in hallucination_indicators:
                print(f"      - {indicator}")
        
        return analysis
    
    def analyze_input_output_consistency(self) -> Dict[str, Any]:
        """Analyze consistency between input listings and output matches"""
        print("\n📊 ANALYZING INPUT-OUTPUT CONSISTENCY:")
        
        # Check if source materials exist in listings
        source_materials_in_listings = set(self.listings_df['material_name'].unique())
        source_materials_in_matches = set(self.matches_df['source_material_name'].unique())
        
        missing_source_materials = source_materials_in_matches - source_materials_in_listings
        common_materials = source_materials_in_listings & source_materials_in_matches
        
        # Check material type consistency
        material_type_issues = []
        for material in common_materials:
            listing_type = self.listings_df[self.listings_df['material_name'] == material]['material_type'].iloc[0]
            # Check if matches are consistent with material type
            material_matches = self.matches_df[self.matches_df['source_material_name'] == material]
            if len(material_matches) > 0:
                # This would need more sophisticated logic for real validation
                pass
        
        # Check value consistency
        value_consistency_issues = []
        for material in common_materials:
            listing_value = self.listings_df[self.listings_df['material_name'] == material]['potential_value'].iloc[0]
            match_values = self.matches_df[self.matches_df['source_material_name'] == material]['potential_value']
            
            # Check if match values are reasonable compared to listing value
            if len(match_values) > 0:
                avg_match_value = match_values.mean()
                if avg_match_value > listing_value * 10 or avg_match_value < listing_value * 0.1:
                    value_consistency_issues.append(f"{material}: Listing ${listing_value:,.2f} vs Avg Match ${avg_match_value:,.2f}")
        
        analysis = {
            'source_materials_in_listings': len(source_materials_in_listings),
            'source_materials_in_matches': len(source_materials_in_matches),
            'missing_source_materials': len(missing_source_materials),
            'common_materials': len(common_materials),
            'material_type_issues': len(material_type_issues),
            'value_consistency_issues': len(value_consistency_issues),
            'missing_materials_list': list(missing_source_materials)[:10],  # First 10
            'value_issues_list': value_consistency_issues[:10]  # First 10
        }
        
        print(f"   • Source materials in listings: {len(source_materials_in_listings)}")
        print(f"   • Source materials in matches: {len(source_materials_in_matches)}")
        print(f"   • Missing source materials: {len(missing_source_materials)}")
        print(f"   • Common materials: {len(common_materials)}")
        print(f"   • Value consistency issues: {len(value_consistency_issues)}")
        
        if missing_source_materials:
            print("   ⚠️  MATERIALS IN MATCHES NOT FOUND IN LISTINGS:")
            for material in list(missing_source_materials)[:5]:
                print(f"      - {material}")
        
        if value_consistency_issues:
            print("   ⚠️  VALUE CONSISTENCY ISSUES:")
            for issue in value_consistency_issues[:5]:
                print(f"      - {issue}")
        
        return analysis
    
    def calculate_quality_score(self) -> Dict[str, Any]:
        """Calculate overall quality score"""
        print("\n📈 CALCULATING OVERALL QUALITY SCORE:")
        
        # Initialize quality metrics
        quality_metrics = {
            'source_id_completeness': 0,
            'consistency_score': 0,
            'hallucination_score': 0,
            'input_output_consistency': 0,
            'overall_quality': 0
        }
        
        # Source ID completeness (0-100)
        missing_source_analysis = self.analyze_missing_source_company_id()
        if missing_source_analysis['missing_source_ids'] == 0:
            quality_metrics['source_id_completeness'] = 100
        else:
            quality_metrics['source_id_completeness'] = 100 - missing_source_analysis['missing_percentage']
        
        # Consistency score (0-100)
        consistency_analysis = self.analyze_match_consistency()
        base_consistency = 70  # Start with 70%
        
        # Deduct points for issues
        if consistency_analysis['duplicate_matches'] > 0:
            base_consistency -= 10
        if consistency_analysis['suspicious_patterns']:
            base_consistency -= len(consistency_analysis['suspicious_patterns']) * 5
        if consistency_analysis['unique_target_companies'] < 10:
            base_consistency -= 15
        
        quality_metrics['consistency_score'] = max(0, base_consistency)
        
        # Hallucination score (0-100, higher is better)
        hallucination_analysis = self.analyze_hallucination_indicators()
        base_hallucination = 80  # Start with 80%
        
        # Deduct points for hallucination indicators
        if hallucination_analysis['generic_companies'] > 0:
            base_hallucination -= hallucination_analysis['generic_companies'] * 2
        if hallucination_analysis['generic_materials'] > 0:
            base_hallucination -= hallucination_analysis['generic_materials'] * 2
        if hallucination_analysis['perfect_scores'] > 0:
            base_hallucination -= 10
        
        quality_metrics['hallucination_score'] = max(0, base_hallucination)
        
        # Input-output consistency (0-100)
        io_analysis = self.analyze_input_output_consistency()
        base_io = 90  # Start with 90%
        
        if io_analysis['missing_source_materials'] > 0:
            base_io -= (io_analysis['missing_source_materials'] / io_analysis['source_materials_in_matches']) * 50
        if io_analysis['value_consistency_issues'] > 0:
            # Fix: Ensure value_consistency_issues is a list before calling len()
            consistency_issues = io_analysis['value_consistency_issues']
            if isinstance(consistency_issues, list):
                base_io -= len(consistency_issues) * 2
            else:
                base_io -= consistency_issues * 2  # If it's already a count
        
        quality_metrics['input_output_consistency'] = max(0, base_io)
        
        # Overall quality (weighted average)
        weights = {
            'source_id_completeness': 0.3,
            'consistency_score': 0.25,
            'hallucination_score': 0.25,
            'input_output_consistency': 0.2
        }
        
        overall_quality = sum(
            quality_metrics[metric] * weights[metric] 
            for metric in weights.keys()
        )
        quality_metrics['overall_quality'] = overall_quality
        
        # Print quality scores
        print(f"   • Source ID Completeness: {quality_metrics['source_id_completeness']:.1f}/100")
        print(f"   • Consistency Score: {quality_metrics['consistency_score']:.1f}/100")
        print(f"   • Hallucination Score: {quality_metrics['hallucination_score']:.1f}/100")
        print(f"   • Input-Output Consistency: {quality_metrics['input_output_consistency']:.1f}/100")
        print(f"   • Overall Quality Score: {quality_metrics['overall_quality']:.1f}/100")
        
        # Quality assessment
        if quality_metrics['overall_quality'] >= 80:
            assessment = "EXCELLENT"
        elif quality_metrics['overall_quality'] >= 60:
            assessment = "GOOD"
        elif quality_metrics['overall_quality'] >= 40:
            assessment = "FAIR"
        else:
            assessment = "POOR"
        
        print(f"   • Quality Assessment: {assessment}")
        
        quality_metrics['assessment'] = assessment
        return quality_metrics
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE MATCH QUALITY ANALYSIS")
        print("="*80)
        
        if not self.load_data():
            return {"error": "Failed to load data"}
        
        # Run all analyses
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'missing_source_analysis': self.analyze_missing_source_company_id(),
            'consistency_analysis': self.analyze_match_consistency(),
            'hallucination_analysis': self.analyze_hallucination_indicators(),
            'input_output_analysis': self.analyze_input_output_consistency(),
            'quality_metrics': self.calculate_quality_score()
        }
        
        # Print summary
        print("\n" + "="*80)
        print("📋 QUALITY ANALYSIS SUMMARY")
        print("="*80)
        
        quality_score = report['quality_metrics']['overall_quality']
        assessment = report['quality_metrics']['assessment']
        
        print(f"\n🎯 OVERALL QUALITY: {quality_score:.1f}/100 ({assessment})")
        
        # Critical issues
        critical_issues = []
        if report['missing_source_analysis']['critical_issue']:
            critical_issues.append("Missing source company IDs")
        if report['consistency_analysis']['suspicious_patterns']:
            critical_issues.append("Suspicious consistency patterns")
        if report['hallucination_analysis']['hallucination_indicators']:
            critical_issues.append("Potential hallucination indicators")
        if report['input_output_analysis']['missing_source_materials'] > 0:
            critical_issues.append("Materials in matches not found in listings")
        
        if critical_issues:
            print("\n⚠️  CRITICAL ISSUES DETECTED:")
            for issue in critical_issues:
                print(f"   • {issue}")
        else:
            print("\n✅ No critical issues detected")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if report['missing_source_analysis']['critical_issue']:
            print("   • 🔧 FIX: Add source company IDs to all matches")
        if report['consistency_analysis']['duplicate_matches'] > 0:
            print("   • 🔧 FIX: Remove duplicate matches")
        if report['hallucination_analysis']['generic_companies'] > 0:
            print("   • 🔧 FIX: Replace generic company names with real companies")
        if report['input_output_analysis']['missing_source_materials'] > 0:
            print("   • 🔧 FIX: Ensure all source materials exist in listings")
        
        if quality_score < 60:
            print("   • ⚠️  Overall quality needs significant improvement")
        elif quality_score < 80:
            print("   • 🔧 Quality is acceptable but can be improved")
        else:
            print("   • ✅ Quality is excellent")
        
        print("\n" + "="*80)
        return report

def main():
    """Main execution function"""
    analyzer = MatchQualityAnalyzer()
    report = analyzer.generate_quality_report()
    return report

if __name__ == "__main__":
    main() 