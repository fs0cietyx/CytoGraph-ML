import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.explainability.shap_runner import SHAPRunner
from src.explainability.gene_importance import compute_gene_importance
from src.explainability.pathway_rollup import PathwayRollup

def test_shap_runner_and_rollup():
    """
    Verifies that the SHAP explanation workflow successfully fits,
    computes SHAP values, ranks genes, and aggregates pathway scores.
    """
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(10, 5), columns=["TCF21", "SLIT3", "EPAS1", "LDB2", "CRYAB"])
    y = pd.Series([0, 1] * 5)
    
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=5, random_state=42))
    ])
    pipeline.fit(X, y)
    
    runner = SHAPRunner(pipeline)
    shap_vals, X_trans = runner.compute_shap_values(X)
    
    # Assert dimensions match samples and features
    assert shap_vals.shape == (10, 5)
    
    # Check gene importance ranking
    importance_df = compute_gene_importance(shap_vals, X_trans)
    assert len(importance_df) == 5
    assert "Gene" in importance_df.columns
    assert "Mean_Abs_SHAP" in importance_df.columns
    
    # Check pathway aggregation
    rollup = PathwayRollup()
    pathway_df = rollup.rollup_shap_values(importance_df)
    assert len(pathway_df) > 0
    assert "Pathway" in pathway_df.columns
    assert "Pathway_SHAP_Contribution" in pathway_df.columns
