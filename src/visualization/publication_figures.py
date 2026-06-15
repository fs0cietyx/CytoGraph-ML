import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.utils.logging import logger
from src.utils.io import RESULTS_DIR, FIGURE_DIR

def generate_pipeline_diagram(save_path: str):
    """
    Generates a high-quality vector-like flowchart diagram representing
    the zero-leakage workflow architecture.
    """
    logger.info(f"PublicationFigures: Generating architecture diagram at {save_path}...")
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.axis('off')
    
    # Define boxes and coordinates
    steps = [
        "1. NCBI GEO Data Ingestion",
        "2. Biological Proxy Blacklist Filter",
        "3. Cross-Study Quantile Normalizer",
        "4. Median Imputer",
        "5. Robust Scaler",
        "6. Mutual Information (MI) Selection",
        "7. Random Forest (RF) Ensemble",
        "8. TreeSHAP Explanation Model",
        "9. Multi-Tissue External Validation"
    ]
    
    y_coords = np.linspace(0.9, 0.1, len(steps))
    
    # Plot boxes and text
    for i, (step, y) in enumerate(zip(steps, y_coords)):
        ax.text(
            0.5, y, step,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='azure', edgecolor='teal', lw=1.5),
            fontsize=10, fontweight='bold', color='darkslategray'
        )
        
        # Draw arrow to the next box
        if i < len(steps) - 1:
            ax.annotate(
                '', 
                xy=(0.5, y_coords[i+1] + 0.03), 
                xytext=(0.5, y - 0.03),
                arrowprops=dict(arrowstyle="->", color="teal", lw=2, mutation_scale=15)
            )
            
    plt.title("CytoGraph-ML Framework Architecture", fontsize=14, fontweight='bold', color='teal', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("PublicationFigures: Figure 1 generated.")

def organize_figures():
    """
    Copies generated figures to the main figures/ folder for final publication delivery.
    """
    logger.info("PublicationFigures: Copying analysis plots to papers figures path...")
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # 1. Generate figure 1
    generate_pipeline_diagram(FIGURE_DIR / "figure1_pipeline.png")
    
    # 2. Copy figure 2 (SHAP)
    shap_src = RESULTS_DIR / "shap" / "shap_summary_plot.png"
    if shap_src.exists():
        shutil.copy(shap_src, FIGURE_DIR / "figure2_shap.png")
        logger.info("PublicationFigures: Figure 2 (SHAP) copied.")
        
    # 3. Copy figure 3 (ROC)
    roc_src = RESULTS_DIR / "external" / "roc_curve_gse19804.png"
    if roc_src.exists():
        shutil.copy(roc_src, FIGURE_DIR / "figure3_roc.png")
        logger.info("PublicationFigures: Figure 3 (ROC) copied.")
        
    # 4. Copy figure 4 (Drift)
    drift_src = RESULTS_DIR / "external" / "expression_drift_SLIT3.png"
    if drift_src.exists():
        shutil.copy(drift_src, FIGURE_DIR / "figure4_drift.png")
        logger.info("PublicationFigures: Figure 4 (Drift) copied.")
