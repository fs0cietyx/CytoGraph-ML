import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utils.logging import logger

def plot_shap_summary(importance_df: pd.DataFrame, save_path: str, top_n: int = 10):
    """
    Plots the top N biomarkers ranked by their mean absolute SHAP value
    and saves the figure to the specified path.
    """
    logger.info(f"SHAPPlots: Plotting top {top_n} biomarkers to {save_path}...")
    plt.figure(figsize=(10, 6))
    
    top_df = importance_df.head(top_n)
    
    sns.barplot(
        x="Mean_Abs_SHAP", 
        y="Gene", 
        data=top_df, 
        palette="crest_r"
    )
    
    plt.title(f"Top {top_n} Biomarkers by Mean Absolute SHAP Value", fontsize=14, fontweight='bold')
    plt.xlabel("Mean Absolute SHAP Value (Impact on Model)", fontsize=12)
    plt.ylabel("Gene Symbol", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
