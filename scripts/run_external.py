import os
import sys
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import logger, set_seed, load_config, RESULTS_DIR
from src.preprocessing import GEODataLoader, GeneMapper
from src.pipelines import create_lung_pipeline
from src.validation import validate_gse19804, validate_gse21510
from src.evaluation import compute_roc_curve
from src.visualization import plot_roc_curve, plot_expression_drift

def main():
    set_seed(42)
    logger.info("================== RUNNING EXTERNAL VALIDATION ==================")
    
    # 1. Load config
    config = load_config("external_validation")
    
    # 2. Ingest datasets
    X_train_raw, y_train, _ = GEODataLoader.load_gse10072()
    X_test_lung_raw, y_test_lung, _ = GEODataLoader.load_gse19804()
    X_test_colorectal_raw, y_test_colorectal, _ = GEODataLoader.load_gse21510()
    
    # 3. Map probes to symbols
    mapper_gpl96 = GeneMapper(platform_id="GPL96")
    mapper_gpl570 = GeneMapper(platform_id="GPL570")
    
    X_train = mapper_gpl96.fit_transform(X_train_raw)
    X_test_lung = mapper_gpl570.fit_transform(X_test_lung_raw)
    X_test_colorectal = mapper_gpl570.fit_transform(X_test_colorectal_raw)
    
    # 4. Build pipeline
    pipeline = create_lung_pipeline(config)
    
    # 5. External validation on GSE19804 (Lung platform shift)
    logger.info("Evaluating on GSE19804 (Lung platform shift)...")
    res_lung = validate_gse19804(pipeline, X_train, y_train, X_test_lung, y_test_lung)
    
    # Get probabilities for ROC
    preprocessor = pipeline[:-1]
    clf = pipeline.named_steps['classifier']
    X_train_aligned = X_train.reindex(columns=list(set(X_train.columns).intersection(set(X_test_lung.columns)))).fillna(0)
    X_test_aligned = X_test_lung.reindex(columns=X_train_aligned.columns).fillna(0)
    
    pipeline.fit(X_train_aligned, y_train)
    X_test_trans = preprocessor.transform(X_test_aligned)
    y_probs_lung = clf.predict_proba(X_test_trans)[:, 1]
    
    # Compute ROC Curve for GSE19804
    roc_res = compute_roc_curve(y_test_lung, y_probs_lung)
    
    # 6. External validation on GSE21510 (Colorectal shift)
    logger.info("Evaluating on GSE21510 (Colorectal cross-tissue shift)...")
    res_colorectal = validate_gse21510(pipeline, X_train, y_train, X_test_colorectal, y_test_colorectal)
    
    # 7. Save metrics
    os.makedirs(RESULTS_DIR / "external", exist_ok=True)
    external_df = pd.DataFrame([
        {
            "Cohort": "GSE19804 (Lung)",
            "Platform": "GPL570",
            "Accuracy": f"{res_lung['accuracy']*100:.2f}%",
            "Recall": f"{res_lung['recall']:.4f}",
            "ROC-AUC": f"{roc_res['auc']:.4f}"
        },
        {
            "Cohort": "GSE21510 (Colorectal)",
            "Platform": "GPL570",
            "Accuracy": f"{res_colorectal['accuracy']*100:.2f}%",
            "Recall": f"{res_colorectal['recall']:.4f}",
            "ROC-AUC": "N/A"
        }
    ])
    external_df.to_csv(RESULTS_DIR / "external" / "external_validation_scores.csv", index=False)
    
    # 8. Save ROC Plot
    plot_roc_curve(
        roc_res["fpr"], 
        roc_res["tpr"], 
        roc_res["auc"], 
        "External GSE19804 (Lung)", 
        str(RESULTS_DIR / "external" / "roc_curve_gse19804.png")
    )
    
    # 9. Jensen-Shannon Distribution Drift Analysis
    logger.info("Running Jensen-Shannon drift analysis for key driver genes...")
    top_genes = ['LDB2', 'SLIT3', 'EPAS1', 'EDNRB', 'KIAA1462']
    drift_results = []
    
    for gene in top_genes:
        if gene in X_train.columns and gene in X_test_lung.columns:
            p_tr = X_train[gene].values
            p_te = X_test_lung[gene].values
            
            # Normalize to [0, 1] for probability estimation
            min_val = min(p_tr.min(), p_te.min())
            max_val = max(p_tr.max(), p_te.max())
            p_tr_norm = (p_tr - min_val) / (max_val - min_val + 1e-9)
            p_te_norm = (p_te - min_val) / (max_val - min_val + 1e-9)
            
            # Histograms
            hist_tr, bin_edges = np.histogram(p_tr_norm, bins=10, density=True)
            hist_te, _ = np.histogram(p_te_norm, bins=bin_edges, density=True)
            
            hist_tr = hist_tr + 1e-9
            hist_te = hist_te + 1e-9
            
            # JS distance
            js_dist = jensenshannon(hist_tr, hist_te)
            
            drift_results.append({
                "Gene": gene,
                "GSE10072 Mean": f"{p_tr.mean():.4f}",
                "GSE19804 Mean": f"{p_te.mean():.4f}",
                "JS Distance": f"{js_dist:.4f}",
                "Drift Status": "Significant" if js_dist > 0.4 else "Moderate"
            })
            
            # Plot drift density for the top gene (e.g. SLIT3)
            if gene == 'SLIT3':
                plot_expression_drift(
                    X_train, X_test_lung, gene, 
                    "GSE10072 (Train)", "GSE19804 (Test)", 
                    str(RESULTS_DIR / "external" / f"expression_drift_{gene}.png")
                )
                
    drift_df = pd.DataFrame(drift_results)
    drift_df.to_csv(RESULTS_DIR / "external" / "distribution_drift.csv", index=False)
    
    logger.info("External validation complete.")
    print(f"Expected External Accuracy (GSE19804): {res_lung['accuracy']*100:.2f}%")
    print(f"Colorectal Transfer Accuracy (GSE21510): {res_colorectal['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
