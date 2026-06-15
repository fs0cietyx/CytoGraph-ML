import pandas as pd
import numpy as np
from src.utils.logging import logger

def compute_gene_importance(shap_values: np.ndarray, X_trans: pd.DataFrame) -> pd.DataFrame:
    """
    Computes average absolute SHAP values for each feature to rank gene importance.
    """
    logger.info("GeneImportance: Calculating mean absolute SHAP values...")
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    importance_df = pd.DataFrame({
        "Gene": X_trans.columns,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)
    
    return importance_df
