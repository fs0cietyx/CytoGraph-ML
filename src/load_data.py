import pandas as pd
import os

def load_and_explore():
    print("Loading PANCAN dataset...")
    
    data_path = 'data/data.csv'
    labels_path = 'data/labels.csv'
    
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        print("Error: Data files not found in data/ folder.")
        return

    # Load data
    # The first column is the sample ID, we'll set it as index
    X = pd.read_csv(data_path, index_col=0)
    y = pd.read_csv(labels_path, index_col=0)
    
    print(f"Dataset loaded: {X.shape[0]} samples and {X.shape[1]} genes.")
    
    # Basic EDA
    print("\n--- Target Distribution ---")
    print(y['Class'].value_counts())
    
    print("\n--- Feature Preview (First 5 genes) ---")
    print(X.iloc[:, :5].head())
    
    # Check for missing values
    missing_vals = X.isnull().sum().sum()
    print(f"\nTotal missing values in features: {missing_vals}")

if __name__ == "__main__":
    load_and_explore()
