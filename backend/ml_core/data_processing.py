"""
ML Core Data Processing: Data loading and preprocessing utilities
"""
import torch
import pandas as pd
import numpy as np

def preprocess_tabular(df: pd.DataFrame):
    # Fill missing, scale, encode categorical, etc.
    df = df.fillna(0)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]
    return df

def preprocess_graph(graph):
    # Convert to torch-geometric Data object
    # Placeholder: real implementation will depend on graph format
    pass

def preprocess_text(texts):
    # Tokenization, embedding, etc.
    # Placeholder for transformer-based text preprocessing
    pass 