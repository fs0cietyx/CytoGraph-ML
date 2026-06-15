import os
import sys
import pandas as pd

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import logger, set_seed, load_config, RESULTS_DIR, BioMapper
from src.preprocessing import GEODataLoader, GeneMapper
from src.pipelines import create_lung_pipeline
from src.explainability import SHAPRunner, compute_gene_importance, PathwayRollup
from src.visualization import plot_shap_summary, plot_pathway_contributions

def main():
    set_seed(42)
    logger.info("================== RUNNING SHAP EXPLAINABILITY ==================")
    
    # Load config
    config = load_config("shap")
    
    # Load data
    X_raw, y, _ = GEODataLoader.load_gse10072()
    
    # Map probes
    mapper = GeneMapper(platform_id="GPL96")
    X = mapper.fit_transform(X_raw)
    
    # Build and fit pipeline
    pipeline = create_lung_pipeline(config)
    pipeline.fit(X, y)
    
    # Compute SHAP values
    runner = SHAPRunner(pipeline)
    shap_vals, X_trans = runner.compute_shap_values(X)
    
    # Compute feature importance
    importance_df = compute_gene_importance(shap_vals, X_trans)
    
    os.makedirs(RESULTS_DIR / "shap", exist_ok=True)
    importance_path = RESULTS_DIR / "shap" / "gene_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Gene SHAP importance saved to {importance_path}")
    
    # Save SHAP Summary Plot
    shap_plot_path = RESULTS_DIR / "shap" / "shap_summary_plot.png"
    plot_shap_summary(importance_df, str(shap_plot_path))
    
    # Pathway rollup (pass BioMapper for real metadata search if needed, or defaults to offline fallback)
    rollup = PathwayRollup(gene_metadata_provider=BioMapper)
    pathway_df = rollup.rollup_shap_values(importance_df)
    
    os.makedirs(RESULTS_DIR / "pathway", exist_ok=True)
    pathway_path = RESULTS_DIR / "pathway" / "pathway_shap_rollup.csv"
    pathway_df.to_csv(pathway_path, index=False)
    logger.info(f"Pathway SHAP rollup saved to {pathway_path}")
    
    # Save Pathway Plot
    pathway_plot_path = RESULTS_DIR / "pathway" / "pathway_contributions.png"
    plot_pathway_contributions(pathway_df, str(pathway_plot_path))
    
    logger.info("SHAP analysis and pathway rollup complete.")
    print("Top 5 Biomarkers Identified:")
    print(importance_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
