import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def analyze_csv(file_path: str, nrows: int = 5):
    """Analyze a CSV file and print its structure"""
    try:
        # Read first few rows
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"\nAnalyzing {file_path}:")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nSample data:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def main():
    # Analyze training data
    train_df = analyze_csv("data/consolidated/how2sign_realigned_train.csv")
    
    # Analyze landmark files (reading just a small portion due to size)
    front_landmarks = analyze_csv("data/consolidated/normalized_landmarks_front.csv")
    side_landmarks = analyze_csv("data/consolidated/normalized_landmarks_side.csv")
    
if __name__ == "__main__":
    main() 