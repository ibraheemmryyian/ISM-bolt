#!/usr/bin/env python3
"""
AI Material Analysis Engine
Comprehensive analysis of material listings and matches for industrial symbiosis insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaterialAnalysisEngine:
    """Advanced AI-powered analysis engine for material listings and matches"""
    
    def __init__(self, listings_path: str = "material_listings.csv", 
                 matches_path: str = "material_matches.csv"):
        self.listings_path = listings_path
        self.matches_path = matches_path
        self.listings_df = None
        self.matches_df = None
        self.analysis_results = {}
        
    def load_data(self) -> bool:
        """Load and validate CSV data"""
        try:
            logger.info("Loading material listings data...")
            self.listings_df = pd.read_csv(self.listings_path)
            logger.info(f"Loaded {len(self.listings_df)} material listings")
            
            logger.info("Loading material matches data...")
            self.matches_df = pd.read_csv(self.matches_path)
            logger.info(f"Loaded {len(self.matches_df)} material matches")
            
            # Convert timestamps
            self.listings_df['generated_at'] = pd.to_datetime(self.listings_df['generated_at'])
            self.matches_df['generated_at'] = pd.to_datetime(self.matches_df['generated_at'])
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def analyze_material_listings(self) -> Dict[str, Any]:
        """Comprehensive analysis of material listings"""
        logger.info("Analyzing material listings...")
        
        analysis = {
            'total_listings': len(self.listings_df),
            'unique_companies': self.listings_df['company_name'].nunique(),
            'material_types': self.listings_df['material_type'].value_counts().to_dict(),
            'quality_distribution': self.listings_df['quality_grade'].value_counts().to_dict(),
            'ai_generated_percentage': (self.listings_df['ai_generated'].sum() / len(self.listings_df)) * 100,
            'total_potential_value': self.listings_df['potential_value'].sum(),
            'avg_potential_value': self.listings_df['potential_value'].mean(),
            'top_companies_by_listings': self.listings_df['company_name'].value_counts().head(10).to_dict(),
            'top_companies_by_value': self.listings_df.groupby('company_name')['potential_value'].sum().sort_values(ascending=False).head(10).to_dict(),
            'quantity_analysis': {
                'total_quantity': self.listings_df['quantity'].sum(),
                'avg_quantity': self.listings_df['quantity'].mean(),
                'quantity_by_type': self.listings_df.groupby('material_type')['quantity'].sum().to_dict()
            }
        }
        
        return analysis
    
    def analyze_material_matches(self) -> Dict[str, Any]:
        """Comprehensive analysis of material matches"""
        logger.info("Analyzing material matches...")
        
        analysis = {
            'total_matches': len(self.matches_df),
            'unique_source_materials': self.matches_df['source_material_name'].nunique(),
            'unique_target_companies': self.matches_df['target_company_name'].nunique(),
            'match_types': self.matches_df['match_type'].value_counts().to_dict(),
            'avg_match_score': self.matches_df['match_score'].mean(),
            'high_quality_matches': len(self.matches_df[self.matches_df['match_score'] >= 0.8]),
            'total_match_value': self.matches_df['potential_value'].sum(),
            'avg_match_value': self.matches_df['potential_value'].mean(),
            'top_match_scores': self.matches_df.nlargest(10, 'match_score')[['source_material_name', 'target_company_name', 'match_score']].to_dict('records'),
            'match_score_distribution': {
                'excellent': len(self.matches_df[self.matches_df['match_score'] >= 0.9]),
                'good': len(self.matches_df[(self.matches_df['match_score'] >= 0.7) & (self.matches_df['match_score'] < 0.9)]),
                'fair': len(self.matches_df[(self.matches_df['match_score'] >= 0.5) & (self.matches_df['match_score'] < 0.7)]),
                'poor': len(self.matches_df[self.matches_df['match_score'] < 0.5])
            }
        }
        
        return analysis
    
    def find_symbiosis_opportunities(self) -> Dict[str, Any]:
        """Identify high-value symbiosis opportunities"""
        logger.info("Identifying symbiosis opportunities...")
        
        # Find high-scoring matches with good value
        high_value_matches = self.matches_df[
            (self.matches_df['match_score'] >= 0.7) & 
            (self.matches_df['potential_value'] >= 1000)
        ].copy()
        
        opportunities = {
            'high_value_opportunities': len(high_value_matches),
            'total_opportunity_value': high_value_matches['potential_value'].sum(),
            'top_opportunities': high_value_matches.nlargest(10, 'potential_value')[
                ['source_material_name', 'target_company_name', 'match_score', 'potential_value']
            ].to_dict('records'),
            'opportunities_by_type': high_value_matches['match_type'].value_counts().to_dict()
        }
        
        return opportunities
    
    def analyze_market_trends(self) -> Dict[str, Any]:
        """Analyze market trends and patterns"""
        logger.info("Analyzing market trends...")
        
        # Material type analysis
        material_trends = self.listings_df.groupby('material_type').agg({
            'potential_value': ['sum', 'mean', 'count'],
            'quantity': ['sum', 'mean']
        }).round(2)
        
        # Convert material trends to JSON-serializable format
        material_trends_dict = {}
        for material_type in material_trends.index:
            material_trends_dict[material_type] = {
                'total_value': float(material_trends.loc[material_type, ('potential_value', 'sum')]),
                'avg_value': float(material_trends.loc[material_type, ('potential_value', 'mean')]),
                'count': int(material_trends.loc[material_type, ('potential_value', 'count')]),
                'total_quantity': float(material_trends.loc[material_type, ('quantity', 'sum')]),
                'avg_quantity': float(material_trends.loc[material_type, ('quantity', 'mean')])
            }
        
        # Company performance analysis
        company_performance = self.listings_df.groupby('company_name').agg({
            'potential_value': ['sum', 'mean'],
            'quantity': ['sum', 'mean'],
            'material_name': 'count'
        }).round(2)
        
        # Convert multi-level columns to JSON-serializable format
        company_perf_dict = {}
        for company in company_performance.index:
            company_perf_dict[company] = {
                'total_value': float(company_performance.loc[company, ('potential_value', 'sum')]),
                'avg_value': float(company_performance.loc[company, ('potential_value', 'mean')]),
                'total_quantity': float(company_performance.loc[company, ('quantity', 'sum')]),
                'avg_quantity': float(company_performance.loc[company, ('quantity', 'mean')]),
                'listing_count': int(company_performance.loc[company, ('material_name', 'count')])
            }
        
        trends = {
            'material_type_performance': material_trends_dict,
            'top_performing_companies': company_perf_dict,
            'value_density': (self.listings_df['potential_value'] / self.listings_df['quantity']).describe().to_dict(),
            'market_concentration': {
                'top_10_companies_share': float((company_performance[('potential_value', 'sum')].nlargest(10).sum() / company_performance[('potential_value', 'sum')].sum()) * 100)
            }
        }
        
        return trends
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate AI-powered recommendations"""
        logger.info("Generating AI recommendations...")
        
        recommendations = {
            'immediate_actions': [],
            'strategic_opportunities': [],
            'risk_mitigation': [],
            'optimization_suggestions': []
        }
        
        # Immediate actions based on high-value matches
        high_matches = self.matches_df[self.matches_df['match_score'] >= 0.8]
        if len(high_matches) > 0:
            recommendations['immediate_actions'].append({
                'action': 'Prioritize high-scoring matches',
                'count': len(high_matches),
                'potential_value': high_matches['potential_value'].sum(),
                'description': f"Focus on {len(high_matches)} matches with scores ≥0.8"
            })
        
        # Strategic opportunities
        waste_materials = self.listings_df[self.listings_df['material_type'] == 'waste']
        if len(waste_materials) > 0:
            recommendations['strategic_opportunities'].append({
                'opportunity': 'Waste-to-Resource conversion',
                'materials_count': len(waste_materials),
                'total_value': waste_materials['potential_value'].sum(),
                'description': 'Convert waste materials into valuable resources'
            })
        
        # Risk mitigation
        low_quality_matches = self.matches_df[self.matches_df['match_score'] < 0.5]
        if len(low_quality_matches) > 0:
            recommendations['risk_mitigation'].append({
                'risk': 'Low-quality matches',
                'count': len(low_quality_matches),
                'action': 'Review and improve matching algorithms'
            })
        
        # Optimization suggestions
        avg_match_score = self.matches_df['match_score'].mean()
        if avg_match_score < 0.7:
            recommendations['optimization_suggestions'].append({
                'suggestion': 'Improve matching algorithm',
                'current_score': avg_match_score,
                'target_score': 0.8,
                'description': 'Enhance AI matching to increase average scores'
            })
        
        return recommendations
    
    def create_visualizations(self, output_dir: str = "analysis_output"):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Material Type Distribution
        plt.figure(figsize=(12, 8))
        material_counts = self.listings_df['material_type'].value_counts()
        plt.pie(material_counts.values, labels=material_counts.index, autopct='%1.1f%%')
        plt.title('Material Type Distribution', fontsize=16, fontweight='bold')
        plt.savefig(f'{output_dir}/material_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Match Score Distribution
        plt.figure(figsize=(12, 6))
        plt.hist(self.matches_df['match_score'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Match Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Match Scores', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/match_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Potential Value by Material Type
        plt.figure(figsize=(14, 8))
        value_by_type = self.listings_df.groupby('material_type')['potential_value'].sum().sort_values(ascending=False)
        plt.bar(range(len(value_by_type)), value_by_type.values)
        plt.xticks(range(len(value_by_type)), value_by_type.index, rotation=45)
        plt.xlabel('Material Type')
        plt.ylabel('Total Potential Value')
        plt.title('Total Potential Value by Material Type', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/value_by_material_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Match Score vs Potential Value
        plt.figure(figsize=(12, 8))
        plt.scatter(self.matches_df['match_score'], self.matches_df['potential_value'], alpha=0.6)
        plt.xlabel('Match Score')
        plt.ylabel('Potential Value')
        plt.title('Match Score vs Potential Value', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/match_score_vs_value.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}/")
    
    def generate_comprehensive_report(self, output_file: str = "ai_analysis_report.json") -> Dict[str, Any]:
        """Generate comprehensive AI analysis report"""
        logger.info("Generating comprehensive AI analysis report...")
        
        if not self.load_data():
            return {"error": "Failed to load data"}
        
        # Run all analyses
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'listings_file': self.listings_path,
                'matches_file': self.matches_path,
                'total_listings': len(self.listings_df),
                'total_matches': len(self.matches_df)
            },
            'material_listings_analysis': self.analyze_material_listings(),
            'material_matches_analysis': self.analyze_material_matches(),
            'symbiosis_opportunities': self.find_symbiosis_opportunities(),
            'market_trends': self.analyze_market_trends(),
            'ai_recommendations': self.generate_recommendations(),
            'key_insights': self.generate_key_insights()
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report saved to {output_file}")
        return report
    
    def generate_key_insights(self) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []
        
        # Market size insights
        total_value = self.listings_df['potential_value'].sum()
        insights.append(f"Total market value: ${total_value:,.2f}")
        
        # Match quality insights
        avg_score = self.matches_df['match_score'].mean()
        insights.append(f"Average match quality: {avg_score:.2%}")
        
        # Opportunity insights
        high_value_opps = len(self.matches_df[
            (self.matches_df['match_score'] >= 0.7) & 
            (self.matches_df['potential_value'] >= 1000)
        ])
        insights.append(f"High-value opportunities identified: {high_value_opps}")
        
        # Company insights
        top_company = self.listings_df.groupby('company_name')['potential_value'].sum().idxmax()
        insights.append(f"Highest value company: {top_company}")
        
        # Material insights
        most_common_material = self.listings_df['material_type'].mode().iloc[0]
        insights.append(f"Most common material type: {most_common_material}")
        
        return insights
    
    def print_summary(self):
        """Print a human-readable summary"""
        print("\n" + "="*80)
        print("🤖 AI MATERIAL ANALYSIS SUMMARY")
        print("="*80)
        
        if not self.load_data():
            print("❌ Failed to load data")
            return
        
        # Basic stats
        print(f"\n📊 DATA OVERVIEW:")
        print(f"   • Material Listings: {len(self.listings_df):,}")
        print(f"   • Material Matches: {len(self.matches_df):,}")
        print(f"   • Unique Companies: {self.listings_df['company_name'].nunique()}")
        
        # Value analysis
        total_value = self.listings_df['potential_value'].sum()
        avg_match_score = self.matches_df['match_score'].mean()
        print(f"\n💰 VALUE ANALYSIS:")
        print(f"   • Total Market Value: ${total_value:,.2f}")
        print(f"   • Average Match Score: {avg_match_score:.1%}")
        
        # Top opportunities
        high_value_matches = self.matches_df[
            (self.matches_df['match_score'] >= 0.7) & 
            (self.matches_df['potential_value'] >= 1000)
        ]
        print(f"\n🎯 TOP OPPORTUNITIES:")
        print(f"   • High-Value Matches: {len(high_value_matches):,}")
        print(f"   • Opportunity Value: ${high_value_matches['potential_value'].sum():,.2f}")
        
        # Material types
        material_dist = self.listings_df['material_type'].value_counts()
        print(f"\n🏭 MATERIAL TYPES:")
        for material_type, count in material_dist.head(5).items():
            print(f"   • {material_type}: {count:,} listings")
        
        print("\n" + "="*80)
        print("📈 Run 'create_visualizations()' to generate charts")
        print("📋 Run 'generate_comprehensive_report()' for detailed analysis")
        print("="*80)

def main():
    """Main execution function"""
    print("🚀 Starting AI Material Analysis Engine...")
    
    # Initialize analysis engine
    analyzer = MaterialAnalysisEngine()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate comprehensive report
    print("\n📊 Generating comprehensive AI analysis...")
    report = analyzer.generate_comprehensive_report()
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    analyzer.create_visualizations()
    
    print("\n✅ Analysis complete! Check the generated files:")
    print("   • ai_analysis_report.json - Detailed analysis report")
    print("   • analysis_output/ - Visualization charts")
    
    return report

if __name__ == "__main__":
    main() 