import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import textstat
from dotenv import load_dotenv  # Add dotenv import

# Load environment variables from .env file if available
load_dotenv()

# Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/embeddings"
CSV_PATH = "listings_output.csv"  # Updated per user request
JSON_PATH = "fixed_realworlddata.json"
OUTPUT_JSON = "ai_quality_benchmark.json"
OUTPUT_MD = "ai_quality_report.md"
OUTPUT_HTML = "ai_quality_report.html"

def get_deepseek_embedding(text):
    """Get text embedding from DeepSeek API"""
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {
        "input": text,
        "model": "text-embedding-001"
    }
    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def load_and_preprocess_data():
    """Load and align datasets using composite key"""
    # Load AI-generated CSV
    ai_df = pd.read_csv(CSV_PATH)
    
    # Load ground truth JSON
    with open(JSON_PATH, 'r') as f:
        real_data = json.load(f)
    real_df = pd.DataFrame(real_data)
    
    # Create composite key from company name and material name
    required_ai_columns = ['company_name', 'material_name']
    required_real_columns = ['company_name', 'material_name']
    
    # Convert column names to lowercase for case-insensitive matching
    ai_df.columns = ai_df.columns.str.lower()
    real_df.columns = real_df.columns.str.lower()
    
    # Rename required columns to lowercase
    required_ai_columns = [col.lower() for col in required_ai_columns]
    required_real_columns = [col.lower() for col in required_real_columns]
    
    if not all(col in ai_df.columns for col in required_ai_columns):
        print("❌ Error: Required columns not found in AI dataset")
        print(f"AI dataset columns: {list(ai_df.columns)}")
        print(f"Required columns: {required_ai_columns}")
        return pd.DataFrame()
        
    if not all(col in real_df.columns for col in required_real_columns):
        print("❌ Error: Required columns not found in real-world dataset")
        print(f"Real-world dataset columns: {list(real_df.columns)}")
        print(f"Required columns: {required_real_columns}")
        return pd.DataFrame()
    
    ai_df['composite_key'] = ai_df['company_name'] + '_' + ai_df['material_name']
    real_df['composite_key'] = real_df['company_name'] + '_' + real_df['material_name']
    
    # Merge datasets using composite key
    try:
        merged_df = pd.merge(
            ai_df, 
            real_df, 
            on='composite_key', 
            suffixes=('_ai', '_real'),
            how='inner'
        )
        
        # Clean up
        merged_df.drop('composite_key', axis=1, inplace=True)
    except Exception as e:
        print(f"❌ Error during merge: {str(e)}")
        return pd.DataFrame()
    return merged_df

def calculate_similarity(ai_text, real_text):
    """Calculate cosine similarity between AI and real text embeddings"""
    # Handle empty text cases
    if not ai_text or not real_text:
        return 0.0
        
    ai_embedding = np.array(get_deepseek_embedding(ai_text)).reshape(1, -1)
    real_embedding = np.array(get_deepseek_embedding(real_text)).reshape(1, -1)
    return cosine_similarity(ai_embedding, real_embedding)[0][0]

def generate_quality_report(df):
    """Generate comprehensive quality report with visualizations"""
    report = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "total_listings": len(df),
            "companies_analyzed": df['company_id'].nunique() if 'company_id' in df.columns else "N/A"
        },
        "summary_metrics": {},
        "text_quality_metrics": {},
        "similarity_distribution": {}
    }
    
    # Extract descriptions from AI data (handling nested structure)
    print("⚙️ Extracting descriptions...")
    df['description_ai'] = df['deepseek_analysis.semantic_analysis'].apply(
        lambda x: x if isinstance(x, str) else ""
    )
    
    # Calculate similarity scores
    print("⚙️ Calculating similarity scores...")
    df['similarity'] = df.apply(
        lambda row: calculate_similarity(row['description_ai'], str(row['description'])), 
        axis=1
    )
    
    # Calculate text quality metrics
    print("⚙️ Calculating text quality metrics...")
    df['ai_readability'] = df['description_ai'].apply(
        lambda x: textstat.flesch_reading_ease(x) if isinstance(x, str) else 0  # type: ignore
    )
    df['real_readability'] = df['description'].apply(
        lambda x: textstat.flesch_reading_ease(str(x)) if pd.notnull(x) else 0  # type: ignore
    )
    
    # Calculate overall metrics
    report["summary_metrics"] = {
        "average_similarity": df['similarity'].mean(),
        "median_similarity": df['similarity'].median(),
        "min_similarity": df['similarity'].min(),
        "max_similarity": df['similarity'].max(),
        "std_dev_similarity": df['similarity'].std(),
        "accuracy_threshold_70": (df['similarity'] > 0.7).mean(),
        "accuracy_threshold_80": (df['similarity'] > 0.8).mean(),
        "top_10%_similarity": df['similarity'].quantile(0.9),
        "bottom_10%_similarity": df['similarity'].quantile(0.1)
    }
    
    report["text_quality_metrics"] = {
        "avg_ai_readability": df['ai_readability'].mean(),
        "avg_real_readability": df['real_readability'].mean(),
        "readability_difference": df['ai_readability'].mean() - df['real_readability'].mean()
    }
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['similarity'], bins=30, kde=True)
    plt.title('Similarity Score Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.savefig('similarity_distribution.png')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['ai_readability'], df['similarity'], alpha=0.5)
    plt.title('Readability vs Similarity')
    plt.xlabel('AI Text Readability (Flesch)')
    plt.ylabel('Similarity Score')
    plt.savefig('readability_vs_similarity.png')
    
    return report, df

def generate_markdown_report(report, sample_df):
    """Generate detailed Markdown report"""
    md_content = f"""
# AI Quality Benchmark Report

**Analysis Date**: {report['metadata']['analysis_date']}  
**Total Listings Analyzed**: {report['metadata']['total_listings']}  
**Companies Represented**: {report['metadata']['companies_analyzed']}  

## Summary Metrics

| Metric | Value |
|--------|-------|
| **Average Similarity** | {report['summary_metrics']['average_similarity']:.2%} |
| **Median Similarity** | {report['summary_metrics']['median_similarity']:.2%} |
| **Min Similarity** | {report['summary_metrics']['min_similarity']:.2%} |
| **Max Similarity** | {report['summary_metrics']['max_similarity']:.2%} |
| **Accuracy @70%** | {report['summary_metrics']['accuracy_threshold_70']:.2%} |
| **Accuracy @80%** | {report['summary_metrics']['accuracy_threshold_80']:.2%} |
| **Top 10% Similarity** | {report['summary_metrics']['top_10%_similarity']:.2%} |
| **Bottom 10% Similarity** | {report['summary_metrics']['bottom_10%_similarity']:.2%} |

## Text Quality Analysis

| Metric | Value |
|--------|-------|
| **AI Text Readability** | {report['text_quality_metrics']['avg_ai_readability']:.1f} (Flesch) |
| **Real Text Readability** | {report['text_quality_metrics']['avg_real_readability']:.1f} (Flesch) |
| **Readability Difference** | {report['text_quality_metrics']['readability_difference']:.1f} |

## Visualizations

![Similarity Distribution](similarity_distribution.png)  
*Distribution of cosine similarity scores between AI-generated and real listings*

![Readability vs Similarity](readability_vs_similarity.png)  
*Relationship between text readability and similarity scores*

## Recommendations

1. **Quality Focus Areas**: 
   - {get_quality_focus(report)}

2. **Content Improvement**:
   - {get_content_improvement(report)}

3. **Benchmarking**:
   - {get_benchmarking_suggestions(report)}
"""

    with open(OUTPUT_MD, 'w') as f:
        f.write(md_content)
        
    # Convert to HTML
    try:
        import markdown
        with open(OUTPUT_MD, 'r') as f:
            html_content = markdown.markdown(f.read())
        with open(OUTPUT_HTML, 'w') as f:
            f.write(f"<html><body>{html_content}</body></html>")
    except ImportError:
        print("⚠️ markdown package not installed, skipping HTML conversion")

def get_quality_focus(report):
    """Generate quality focus recommendations"""
    if report['summary_metrics']['accuracy_threshold_80'] < 0.6:
        return "Prioritize improving listings with similarity scores below 70% as they represent significant quality gaps"
    elif report['summary_metrics']['accuracy_threshold_80'] < 0.8:
        return "Focus on listings in the 70-80% similarity range to bring them up to premium quality standards"
    else:
        return "Maintain high-quality standards while optimizing for cost and efficiency"

def get_content_improvement(report):
    """Generate content improvement recommendations"""
    readability_diff = report['text_quality_metrics']['readability_difference']
    if readability_diff > 10:
        return "Simplify AI text to match real-world readability levels"
    elif readability_diff < -5:
        return "Enhance AI text complexity to match professional standards"
    else:
        return "Maintain current readability levels which align well with real-world content"

def get_benchmarking_suggestions(report):
    """Generate benchmarking suggestions"""
    if report['summary_metrics']['average_similarity'] > 0.85:
        return "AI quality exceeds industry benchmarks - consider cost optimization"
    elif report['summary_metrics']['average_similarity'] > 0.75:
        return "AI quality meets industry standards - focus on consistency"
    else:
        return "Implement quality improvement initiatives to reach industry benchmarks"

def main():
    print("🚀 Starting AI quality benchmark analysis")
    print(f"• Using real-world data: {JSON_PATH}")
    print(f"• Analyzing AI output: {CSV_PATH}")
    
    if not DEEPSEEK_API_KEY:
        print("❌ Error: DEEPSEEK_API_KEY not found in environment variables or .env file")
        print("💡 Create a .env file with DEEPSEEK_API_KEY=your_api_key or set system environment variable")
        return
    
    merged_data = load_and_preprocess_data()
    if merged_data.empty:
        print("❌ Failed to preprocess data. Exiting.")
        return
        
    print(f"✅ Loaded {len(merged_data)} comparable records")
    
    print("⚙️ Calculating quality metrics via DeepSeek API...")
    report, full_df = generate_quality_report(merged_data)
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📝 Generating detailed report...")
    generate_markdown_report(report, full_df.head(20))
    
    print("\n📊 Report Summary:")
    print(f"  - Average similarity: {report['summary_metrics']['average_similarity']:.2%}")
    print(f"  - Accuracy @70%: {report['summary_metrics']['accuracy_threshold_70']:.2%}")
    print(f"  - Accuracy @80%: {report['summary_metrics']['accuracy_threshold_80']:.2%}")
    print(f"  - Readability score: {report['text_quality_metrics']['avg_ai_readability']:.1f}")
    print(f"\n✅ Analysis complete! Generated reports:")
    print(f"  - JSON report: {OUTPUT_JSON}")
    print(f"  - Markdown report: {OUTPUT_MD}")
    if os.path.exists(OUTPUT_HTML):
        print(f"  - HTML report: {OUTPUT_HTML}")
    print("\n💡 Next steps: Review the reports and visualizations for quality insights")

if __name__ == "__main__":
    main()
