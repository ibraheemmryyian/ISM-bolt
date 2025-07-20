#!/usr/bin/env python3
"""
Quick AI Analysis of Material Listings and Matches
Provides immediate insights without additional dependencies
"""

import pandas as pd
import json
from datetime import datetime

def quick_analysis():
    """Quick analysis of the generated CSV files"""
    
    print("🔍 QUICK AI ANALYSIS OF MATERIAL DATA")
    print("=" * 50)
    
    try:
        # Load data
        print("\n📊 Loading data...")
        listings = pd.read_csv('material_listings.csv')
        matches = pd.read_csv('material_matches.csv')
        
        print(f"✅ Loaded {len(listings):,} listings and {len(matches):,} matches")
        
        # Basic statistics
        print("\n📈 BASIC STATISTICS:")
        print(f"   • Total Companies: {listings['company_name'].nunique()}")
        print(f"   • Total Material Types: {listings['material_type'].nunique()}")
        print(f"   • Total Market Value: ${listings['potential_value'].sum():,.2f}")
        print(f"   • Average Match Score: {matches['match_score'].mean():.1%}")
        
        # Top companies by value
        print("\n🏆 TOP 5 COMPANIES BY VALUE:")
        top_companies = listings.groupby('company_name')['potential_value'].sum().sort_values(ascending=False).head(5)
        for i, (company, value) in enumerate(top_companies.items(), 1):
            print(f"   {i}. {company}: ${value:,.2f}")
        
        # Material type distribution
        print("\n🏭 MATERIAL TYPE DISTRIBUTION:")
        material_dist = listings['material_type'].value_counts()
        for material_type, count in material_dist.items():
            percentage = (count / len(listings)) * 100
            print(f"   • {material_type}: {count:,} ({percentage:.1f}%)")
        
        # Match quality analysis
        print("\n🎯 MATCH QUALITY ANALYSIS:")
        excellent = len(matches[matches['match_score'] >= 0.9])
        good = len(matches[(matches['match_score'] >= 0.7) & (matches['match_score'] < 0.9)])
        fair = len(matches[(matches['match_score'] >= 0.5) & (matches['match_score'] < 0.7)])
        poor = len(matches[matches['match_score'] < 0.5])
        
        print(f"   • Excellent (≥90%): {excellent:,}")
        print(f"   • Good (70-89%): {good:,}")
        print(f"   • Fair (50-69%): {fair:,}")
        print(f"   • Poor (<50%): {poor:,}")
        
        # High-value opportunities
        print("\n💰 HIGH-VALUE OPPORTUNITIES:")
        high_value = matches[
            (matches['match_score'] >= 0.7) & 
            (matches['potential_value'] >= 1000)
        ]
        print(f"   • High-quality matches: {len(high_value):,}")
        print(f"   • Total opportunity value: ${high_value['potential_value'].sum():,.2f}")
        
        # Match types
        print("\n🔗 MATCH TYPES:")
        match_types = matches['match_type'].value_counts()
        for match_type, count in match_types.items():
            print(f"   • {match_type}: {count:,}")
        
        # AI insights
        print("\n🤖 AI INSIGHTS:")
        ai_generated = listings['ai_generated'].sum()
        ai_percentage = (ai_generated / len(listings)) * 100
        print(f"   • AI-generated listings: {ai_generated:,} ({ai_percentage:.1f}%)")
        
        # Top matches by score
        print("\n⭐ TOP 5 MATCHES BY SCORE:")
        top_matches = matches.nlargest(5, 'match_score')[
            ['source_material_name', 'target_company_name', 'match_score', 'potential_value']
        ]
        for i, row in top_matches.iterrows():
            print(f"   {i+1}. {row['source_material_name']} → {row['target_company_name']}")
            print(f"      Score: {row['match_score']:.1%}, Value: ${row['potential_value']:,.2f}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if len(high_value) > 0:
            print(f"   • Prioritize {len(high_value)} high-value matches for immediate action")
        
        if matches['match_score'].mean() < 0.7:
            print("   • Consider improving matching algorithms to increase average scores")
        
        waste_materials = listings[listings['material_type'] == 'waste']
        if len(waste_materials) > 0:
            print(f"   • Focus on {len(waste_materials)} waste materials for circular economy opportunities")
        
        print("\n" + "=" * 50)
        print("✅ Quick analysis complete!")
        print("📊 Run 'python ai_material_analysis_engine.py' for detailed analysis")
        print("📈 Run 'run_ai_analysis.bat' for full analysis with visualizations")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Make sure the CSV files are in the current directory")

if __name__ == "__main__":
    quick_analysis() 