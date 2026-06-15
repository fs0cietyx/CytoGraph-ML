import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utils.logging import logger

def plot_expression_drift(X1: pd.DataFrame, X2: pd.DataFrame, gene: str, label1: str, label2: str, save_path: str):
    """
    Plots the density distributions of a gene's expression values in two datasets
    to visualize any platform or tissue-specific distribution shift.
    """
    logger.info(f"DriftPlots: Saving distribution drift plot for {gene} to {save_path}...")
    plt.figure(figsize=(8, 5))
    
    if gene in X1.columns:
        sns.kdeplot(X1[gene], label=label1, fill=True, color='skyblue', alpha=0.4, lw=2)
    if gene in X2.columns:
        sns.kdeplot(X2[gene], label=label2, fill=True, color='coral', alpha=0.4, lw=2)
        
    plt.title(f"Distribution Drift Analysis for {gene}", fontsize=14, fontweight='bold')
    plt.xlabel("Quantile-Normalized Expression Level", fontsize=12)
    plt.ylabel("Density Estimation", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
