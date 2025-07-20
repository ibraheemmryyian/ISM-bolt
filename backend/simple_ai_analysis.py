#!/usr/bin/env python3
"""
Simple AI Analysis of Material Listings and Matches
Comprehensive analysis without complex JSON serialization
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def analyze_materials():
    """Comprehensive AI analysis of material data"""
    
    print("🤖 AI MATERIAL ANALYSIS ENGINE")
    print("=" * 60)
    
    try:
        # Load data
        print("\n📊 Loading data...")
        listings = pd.read_csv('material_listings.csv')
        matches = pd.read_csv('material_matches.csv')
        
        print(f"✅ Loaded {len(listings):,} listings and {len(matches):,} matches")
        
        # Basic statistics
        print("\n📈 MARKET OVERVIEW:")
        total_value = listings['potential_value'].sum()
        avg_match_score = matches['match_score'].mean()
        unique_companies = listings['company_name'].nunique()
        
        print(f"   • Total Market Value: ${total_value:,.2f}")
        print(f"   • Average Match Score: {avg_match_score:.1%}")
        print(f"   • Unique Companies: {unique_companies}")
        print(f"   • Total Material Types: {listings['material_type'].nunique()}")
        
        # Material type analysis
        print("\n🏭 MATERIAL TYPE ANALYSIS:")
        material_dist = listings['material_type'].value_counts()
        for material_type, count in material_dist.items():
            percentage = (count / len(listings)) * 100
            value = listings[listings['material_type'] == material_type]['potential_value'].sum()
            print(f"   • {material_type}: {count:,} listings ({percentage:.1f}%) - ${value:,.2f}")
        
        # Company analysis
        print("\n🏆 TOP 10 COMPANIES BY VALUE:")
        company_values = listings.groupby('company_name')['potential_value'].sum().sort_values(ascending=False)
        for i, (company, value) in enumerate(company_values.head(10).items(), 1):
            print(f"   {i:2d}. {company}: ${value:,.2f}")
        
        # Match quality analysis
        print("\n🎯 MATCH QUALITY ANALYSIS:")
        excellent = len(matches[matches['match_score'] >= 0.9])
        good = len(matches[(matches['match_score'] >= 0.7) & (matches['match_score'] < 0.9)])
        fair = len(matches[(matches['match_score'] >= 0.5) & (matches['match_score'] < 0.7)])
        poor = len(matches[matches['match_score'] < 0.5])
        
        print(f"   • Excellent (≥90%): {excellent:,} matches")
        print(f"   • Good (70-89%): {good:,} matches")
        print(f"   • Fair (50-69%): {fair:,} matches")
        print(f"   • Poor (<50%): {poor:,} matches")
        
        # High-value opportunities
        print("\n💰 HIGH-VALUE OPPORTUNITIES:")
        high_value_matches = matches[
            (matches['match_score'] >= 0.7) & 
            (matches['potential_value'] >= 1000)
        ]
        print(f"   • High-quality matches: {len(high_value_matches):,}")
        print(f"   • Total opportunity value: ${high_value_matches['potential_value'].sum():,.2f}")
        print(f"   • Average opportunity value: ${high_value_matches['potential_value'].mean():,.2f}")
        
        # Match types
        print("\n🔗 MATCH TYPE DISTRIBUTION:")
        match_types = matches['match_type'].value_counts()
        for match_type, count in match_types.items():
            percentage = (count / len(matches)) * 100
            avg_score = matches[matches['match_type'] == match_type]['match_score'].mean()
            print(f"   • {match_type}: {count:,} matches ({percentage:.1f}%) - Avg score: {avg_score:.1%}")
        
        # AI insights
        print("\n🤖 AI-GENERATED INSIGHTS:")
        ai_generated = listings['ai_generated'].sum()
        ai_percentage = (ai_generated / len(listings)) * 100
        print(f"   • AI-generated listings: {ai_generated:,} ({ai_percentage:.1f}%)")
        
        # Top matches
        print("\n⭐ TOP 10 MATCHES BY SCORE:")
        top_matches = matches.nlargest(10, 'match_score')[
            ['source_material_name', 'target_company_name', 'match_score', 'potential_value', 'match_type']
        ]
        for i, row in top_matches.iterrows():
            print(f"   {i+1:2d}. {row['source_material_name'][:30]:<30} → {row['target_company_name'][:25]:<25}")
            print(f"       Score: {row['match_score']:.1%}, Value: ${row['potential_value']:,.2f}, Type: {row['match_type']}")
        
        # Waste analysis
        print("\n♻️ WASTE-TO-RESOURCE OPPORTUNITIES:")
        waste_materials = listings[listings['material_type'] == 'waste']
        if len(waste_materials) > 0:
            waste_value = waste_materials['potential_value'].sum()
            waste_matches = matches[matches['source_material_name'].isin(waste_materials['material_name'])]
            print(f"   • Waste materials: {len(waste_materials):,} listings")
            print(f"   • Waste material value: ${waste_value:,.2f}")
            print(f"   • Waste material matches: {len(waste_matches):,}")
            print(f"   • Average waste match score: {waste_matches['match_score'].mean():.1%}")
        
        # Market concentration
        print("\n📊 MARKET CONCENTRATION:")
        top_10_share = (company_values.head(10).sum() / company_values.sum()) * 100
        print(f"   • Top 10 companies share: {top_10_share:.1f}% of total market value")
        
        # Value density
        print("\n💎 VALUE DENSITY ANALYSIS:")
        value_density = listings['potential_value'] / listings['quantity']
        print(f"   • Average value per unit: ${value_density.mean():,.2f}")
        print(f"   • Highest value density: ${value_density.max():,.2f}")
        print(f"   • Lowest value density: ${value_density.min():,.2f}")
        
        # Recommendations
        print("\n💡 AI RECOMMENDATIONS:")
        
        if len(high_value_matches) > 0:
            print(f"   • 🎯 Prioritize {len(high_value_matches)} high-value matches for immediate action")
        
        if avg_match_score < 0.7:
            print("   • 🔧 Improve matching algorithms to increase average scores")
        
        if len(waste_materials) > 0:
            print(f"   • ♻️ Focus on {len(waste_materials)} waste materials for circular economy")
        
        if top_10_share > 50:
            print("   • ⚠️ High market concentration - consider diversifying opportunities")
        
        # Generate insights
        print("\n🔍 KEY INSIGHTS:")
        insights = []
        
        # Market size
        if total_value > 10000000:
            insights.append("Large market opportunity with significant value potential")
        elif total_value > 1000000:
            insights.append("Medium-sized market with good growth potential")
        else:
            insights.append("Emerging market with room for expansion")
        
        # Match quality
        if avg_match_score >= 0.8:
            insights.append("Excellent match quality indicates strong AI performance")
        elif avg_match_score >= 0.6:
            insights.append("Good match quality with room for improvement")
        else:
            insights.append("Match quality needs improvement for better outcomes")
        
        # Opportunities
        if len(high_value_matches) > 1000:
            insights.append("Abundant high-value opportunities for immediate action")
        elif len(high_value_matches) > 100:
            insights.append("Good number of high-value opportunities available")
        else:
            insights.append("Limited high-value opportunities - focus on quality over quantity")
        
        # Waste potential
        if len(waste_materials) > 100:
            insights.append("Strong potential for waste-to-resource circular economy")
        elif len(waste_materials) > 50:
            insights.append("Moderate potential for waste conversion opportunities")
        else:
            insights.append("Limited waste materials - focus on other material types")
        
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        print("\n" + "=" * 60)
        print("✅ AI Analysis Complete!")
        print("📊 Data analyzed: Material listings and matches")
        print("🎯 Focus areas: High-value opportunities and waste conversion")
        print("=" * 60)
        
        return {
            'total_value': total_value,
            'avg_match_score': avg_match_score,
            'high_value_opportunities': len(high_value_matches),
            'waste_materials': len(waste_materials),
            'insights': insights
        }
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None

if __name__ == "__main__":
    analyze_materials() 