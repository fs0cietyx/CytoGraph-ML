import matplotlib.pyplot as plt
from src.utils.logging import logger

def plot_roc_curve(fpr: list[float], tpr: list[float], auc_score: float, label: str, save_path: str):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and saves the figure.
    """
    logger.info(f"ROCPlots: Saving ROC curve to {save_path}...")
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, color='teal', lw=2.5, label=f'{label} (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
