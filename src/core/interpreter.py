import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from .config import logger, config

class APEXInterpreter:
    """Advanced Model Interpretability Layer using SHAP values."""
    
    def __init__(self, model, X_train: pd.DataFrame):
        self.model = model
        self.X_train = X_train

    def compute_shap_values(self, X_test: pd.DataFrame):
        """
        Computes SHAP values to explain individual and global gene influence.
        Note: SHAP can be compute-intensive for high-dimensional data.
        """
        logger.info("Computing SHAP values for biological interpretability...")
        
        # We'll use a subset of features for the SHAP summary to ensure performance
        # but the protocol mandates deep insight.
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)
        
        # Save summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plot_path = config.RESULTS_DIR / "shap_summary_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"SHAP Interpretability artifacts saved to {plot_path}")
        return shap_values
