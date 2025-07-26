import pandas as pd
import numpy as np
import json
import re
import textstat
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import hashlib
import pickle

# Initialize local embedding model
EMBEDDING_CACHE = {}
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    """Get text embedding using local model with caching"""
    if not text:
        return np.zeros(384)  # Return zero vector for empty text
    
    # Create cache key
    cache_key = hashlib.md5(text.encode()).hexdigest()
    
    # Return cached embedding if available
    if cache_key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[cache_key]
    
    # Generate and cache new embedding
    embedding = EMBEDDING_MODEL.encode(text)
    EMBEDDING_CACHE[cache_key] = embedding
    return embedding

def calculate_semantic_similarity(text1, text2):
    """Calculate cosine similarity between text embeddings"""
    emb1 = get_embedding(text1).reshape(1, -1)
    emb2 = get_embedding(text2).reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]

def analyze_listing_quality(ai_listing, source_data):
    """Analyze quality of a single AI-generated listing"""
    # Accuracy metrics
    accuracy_metrics = {
        "material_name_match": ai_listing["material_name"] == source_data.get("material_name", ""),
        "material_type_match": ai_listing["material_type"] == source_data.get("material_type", ""),
        "semantic_similarity": calculate_semantic_similarity(
            ai_listing["description"], 
            source_data.get("description", "")
        )
    }
    
    # Depth metrics
    ai_desc = ai_listing["description"]
    depth_metrics = {
        "description_length": len(ai_desc),
        "technical_terms": len(re.findall(r'\b(?:alloy|polymer|composite|tensile strength|corrosion resistance)\b', ai_desc, re.IGNORECASE)),
        "readability_score": textstat.flesch_reading_ease(ai_desc),  # type: ignore
        "detail_score": len(re.findall(r'\b(?:specification|grade|composition|property)\b', ai_desc, re.IGNORECASE))
    }
    
    # Production-worthiness metrics
    prod_metrics = {
        "completeness": all(key in ai_listing for key in ["quantity", "unit", "quality_grade"]),
        "precision_score": len(re.findall(r'\b\d+\.?\d*\b', ai_desc)) / max(1, len(ai_desc.split())),
        "professional_tone": len(re.findall(r'\b(?:certified|grade|standard|compliance)\b', ai_desc, re.IGNORECASE))
    }
    
    # Hallucination detection
    hallucination_indicators = [
        "not mentioned in source",
        "no available data",
        "source doesn't specify",
        "unknown"
    ]
    
    hallucination_count = sum(
        indicator in ai_desc.lower() 
        for indicator in hallucination_indicators
    )
    
    return {
        "accuracy": accuracy_metrics,
        "depth": depth_metrics,
        "production_worthiness": prod_metrics,
        "hallucination_indicators": hallucination_count,
        "industry": source_data.get("industry", "unknown")
    }

def generate_quality_report(input_data, ai_listings):
    """Generate comprehensive quality report"""
    report = {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "total_listings": len(ai_listings),
            "companies_analyzed": len(input_data),
            "industries_covered": len(set(item.get("industry", "") for item in input_data))
        },
        "summary_metrics": {},
        "industry_analysis": {},
        "quality_distribution": {},
        "recommendations": {}
    }
    
    # Analyze each listing
    quality_results = []
    for listing in ai_listings:
        company_id = listing.get("company_id")
        source = next((item for item in input_data if item.get("id") == company_id), {})
        quality_results.append(analyze_listing_quality(listing, source))
    
    # Overall metrics
    report["summary_metrics"] = {
        "avg_semantic_similarity": np.mean([r["accuracy"]["semantic_similarity"] for r in quality_results]),
        "avg_description_length": np.mean([r["depth"]["description_length"] for r in quality_results]),
        "avg_technical_terms": np.mean([r["depth"]["technical_terms"] for r in quality_results]),
        "completeness_rate": np.mean([1 if r["production_worthiness"]["completeness"] else 0 for r in quality_results]),
        "hallucination_rate": np.mean([r["hallucination_indicators"] for r in quality_results]),
        "excellent_quality": sum(1 for r in quality_results if r["accuracy"]["semantic_similarity"] > 0.85 and r["depth"]["technical_terms"] > 3)
    }
    
    # Industry-specific analysis
    industry_metrics = {}
    for industry in set(r["industry"] for r in quality_results):
        industry_results = [r for r in quality_results if r["industry"] == industry]
        industry_metrics[industry] = {
            "count": len(industry_results),
            "avg_similarity": np.mean([r["accuracy"]["semantic_similarity"] for r in industry_results]),
            "avg_technical_terms": np.mean([r["depth"]["technical_terms"] for r in industry_results]),
            "hallucination_rate": np.mean([r["hallucination_indicators"] for r in industry_results])
        }
    report["industry_analysis"] = industry_metrics
    
    # Generate visualizations
    plt.figure(figsize=(12, 6))
    sns.histplot([r["accuracy"]["semantic_similarity"] for r in quality_results], bins=20, kde=True)
    plt.title('Semantic Similarity Distribution')
    plt.xlabel('Similarity Score')
    plt.savefig('semantic_similarity_distribution.png')
    
    plt.figure(figsize=(12, 6))
    industries = list(industry_metrics.keys())
    avg_scores = [industry_metrics[i]["avg_similarity"] for i in industries]
    sns.barplot(x=industries, y=avg_scores)
    plt.title('Industry-wise Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Average Semantic Similarity')
    plt.savefig('industry_accuracy_comparison.png')
    
    # Recommendations
    report["recommendations"] = {
        "accuracy_improvement": "Focus on companies in the automotive and electronics industries showing lower accuracy scores" if any(v["avg_similarity"] < 0.7 for v in industry_metrics.values()) else "Accuracy meets standards across all industries",
        "depth_enhancement": "Increase technical specifications for materials in construction and packaging industries" if any(v["avg_technical_terms"] < 2 for k,v in industry_metrics.items() if k in ["construction", "packaging"]) else "Technical depth meets requirements",
        "hallucination_reduction": "Implement stricter fact-checking for pharmaceutical and chemical industry listings" if any(v["hallucination_rate"] > 0.5 for k,v in industry_metrics.items() if k in ["pharmaceutical", "chemical"]) else "Hallucination rates within acceptable limits"
    }
    
    # Save full quality results
    full_report = {
        "metadata": report["metadata"],
        "listings_quality": quality_results,
        "summary": report["summary_metrics"],
        "industry_analysis": report["industry_analysis"],
        "recommendations": report["recommendations"]
    }
    
    with open('ai_quality_full_report.json', 'w') as f:
        json.dump(full_report, f, indent=2)
    
    # Save embedding cache for future use
    with open('embedding_cache.pkl', 'wb') as f:
        pickle.dump(EMBEDDING_CACHE, f)
    
    return full_report

def main():
    print("🚀 Starting Advanced AI Quality Analysis")
    
    # Load input data
    with open('fixed_realworlddata.json', 'r') as f:
        input_data = json.load(f)
    
    # Load AI-generated listings
    ai_listings = pd.read_csv('listings_output.csv').to_dict('records')
    
    # Generate quality report
    report = generate_quality_report(input_data, ai_listings)
    
    print("\n📊 AI Quality Report Summary:")
    print(f"  - Companies analyzed: {report['metadata']['companies_analyzed']}")
    print(f"  - Listings evaluated: {report['metadata']['total_listings']}")
    print(f"  - Semantic similarity: {report['summary']['avg_semantic_similarity']:.2%}")
    print(f"  - Technical depth: {report['summary']['avg_technical_terms']:.1f} terms/listing")
    print(f"  - Production-ready: {report['summary']['excellent_quality']} listings ({report['summary']['excellent_quality']/len(ai_listings):.1%})")
    
    print("\n🔍 Key Recommendations:")
    for key, rec in report['recommendations'].items():
        print(f"  - {rec}")
    
    print("\n✅ Analysis complete! Full report saved to ai_quality_full_report.json")

if __name__ == "__main__":
    main()
