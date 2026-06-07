from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

def fetch_and_save_data():
    print("Fetching Gene Expression Cancer RNA-Seq dataset from UCI...")
    # Fetch dataset
    cancer_rna_seq = fetch_ucirepo(id=401)
    
    # Data as pandas dataframes
    X = cancer_rna_seq.data.features
    y = cancer_rna_seq.data.targets
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    print("Saving features to data/features.csv...")
    X.to_csv('data/features.csv', index=False)
    
    print("Saving targets to data/targets.csv...")
    y.to_csv('data/targets.csv', index=False)
    
    print(f"Done! Dataset contains {X.shape[0]} samples and {X.shape[1]} genes.")
    print(f"Target distribution:\n{y.value_counts()}")

if __name__ == "__main__":
    fetch_and_save_data()
