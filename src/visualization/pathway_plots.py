import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utils.logging import logger

def plot_pathway_contributions(pathway_df: pd.DataFrame, save_path: str):
    """
    Plots biological pathway enrichment scores aggregated from gene SHAP values.
    """
    logger.info(f"PathwayPlots: Saving pathway contribution plot to {save_path}...")
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        x="Pathway_SHAP_Contribution", 
        y="Pathway", 
        data=pathway_df, 
        palette="viridis"
    )
    
    plt.title("Pathway-Level SHAP Contribution Scores", fontsize=14, fontweight='bold')
    plt.xlabel("Pathway SHAP Contribution Score (Aggregated)", fontsize=12)
    plt.ylabel("Biological Pathway", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
