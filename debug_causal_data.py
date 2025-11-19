print("Debug script starting...")
import pandas as pd
import numpy as np
from src.data_loader import MINDDataLoader
from src.mock_data import create_mock_mind_data
import os

def debug_data():
    data_path = 'data'
    # Ensure mock data exists
    create_mock_mind_data(data_path)
    
    loader = MINDDataLoader(data_path)
    loader.load_data()
    loader.preprocess_item_features()
    
    # Generate a smaller sample for quick debugging
    df = loader.generate_causal_dataset(sample_size=1000)
    
    print("\n--- Data Analysis ---")
    print(f"Shape: {df.shape}")
    print("\nColumn Variances:")
    variances = df.var()
    print(variances)
    
    print("\nConstant Columns (Variance = 0):")
    constant_cols = variances[variances == 0].index.tolist()
    print(constant_cols)
    
    print("\nCorrelation Matrix:")
    corr_matrix = df.corr()
    print(corr_matrix)
    
    print("\nChecking for NaNs:")
    print(df.isna().sum())

    # Check for perfect correlation (off-diagonal 1.0 or -1.0)
    print("\nPerfectly Correlated Pairs:")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.99999:
                print(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]}")

if __name__ == "__main__":
    debug_data()
