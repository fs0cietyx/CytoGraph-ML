import numpy as np
import pandas as pd
from src.utils.logging import logger
from src.explainability.shap_runner import SHAPRunner
from src.explainability.gene_importance import compute_gene_importance

def evaluate_shap_stability(pipeline_builder, X: pd.DataFrame, y: pd.Series, n_bootstrap: int = 5, top_n: int = 10, random_state: int = 42) -> dict:
    """
    Evaluates the stability of SHAP feature importance rankings across bootstrap samples
    of the training data using the Jaccard similarity coefficient.
    """
    logger.info(f"SHAPStability: Evaluating stability across {n_bootstrap} bootstrap splits...")
    
    top_genes_sets = []
    
    for i in range(n_bootstrap):
        # Bootstrap sampling
        np.random.seed(random_state + i)
        boot_idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X.iloc[boot_idx]
        y_boot = y.iloc[boot_idx]
        
        # Build and fit new pipeline instance
        pipeline = pipeline_builder()
        pipeline.fit(X_boot, y_boot)
        
        # Calculate SHAP values
        runner = SHAPRunner(pipeline)
        shap_vals, X_trans = runner.compute_shap_values(X_boot)
        
        importance_df = compute_gene_importance(shap_vals, X_trans)
        top_genes = set(importance_df.head(top_n)["Gene"].tolist())
        top_genes_sets.append(top_genes)
        
    # Calculate pairwise Jaccard index
    jaccards = []
    for i in range(len(top_genes_sets)):
        for j in range(i + 1, len(top_genes_sets)):
            intersection = top_genes_sets[i].intersection(top_genes_sets[j])
            union = top_genes_sets[i].union(top_genes_sets[j])
            jaccard = len(intersection) / len(union) if union else 1.0
            jaccards.append(jaccard)
            
    mean_jaccard = np.mean(jaccards) if jaccards else 1.0
    logger.info(f"SHAPStability: Mean Jaccard index (top {top_n} genes) = {mean_jaccard:.4f}")
    
    return {
        "mean_jaccard": mean_jaccard,
        "jaccards": jaccards
    }
