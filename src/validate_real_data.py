import sys
import os
from core.config import logger, config
from core.geo_engine import GEOIngestor
from core.trainer import GenomicResearchTrainer
from core.bio_mapper import BioMapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit

def run_real_world_validation():
    """
    Brutally Honest Scientific Validation:
    1. Fetches real RNA-seq data from NCBI GEO (GSE10072).
    2. Maps probes to verified Gene Symbols.
    3. Implements Group-Blind Splitting (No Patient Overlap).
    4. Filters out Biological Proxy Genes (Endothelial/Inflammation).
    """
    logger.info("VALIDATION: Starting Scientifically-Hardened Validation (GSE10072)...")

    try:
        # 1. Real Data Acquisition (Now returns Groups/Patient IDs)
        ingestor = GEOIngestor(gse_id="GSE10072")
        X, y, groups = ingestor.fetch_and_map()

        if X is None or y is None:
            logger.error("VALIDATION: Data ingestion failed. Aborting.")
            return

        # 2. Group-Blind Holdout Split
        # Ensures that a patient's data is either ALL in train or ALL in test
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=config.RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        
        logger.info(f"Group-Blind Split: {len(X_train)} Train samples, {len(X_test)} Test samples.")
        logger.info(f"Training on {groups_train.nunique()} unique patients.")

        # 3. Training with Proxy Filtering and MI Selection
        trainer = GenomicResearchTrainer()
        results = trainer.execute_stratified_training(X_train, y_train, groups=groups_train)

        # 4. Holdout Evaluation (Must also apply biological filter to test set)
        X_test_filtered = X_test.drop(columns=[g for g in trainer.biological_blacklist if g in X_test.columns])
        y_pred = trainer.pipeline.predict(X_test_filtered)
        acc = accuracy_score(y_test, y_pred)
        
        logger.info(f"FINAL HONEST ACCURACY: {acc:.4f}")
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

        # 5. Biological Enrichment
        importance_df = trainer.get_feature_importance()
        top_genes = importance_df.head(10)["Gene"].tolist()
        
        logger.info(f"Identified Non-Proxy Biomarkers: {top_genes}")
        bio_report = BioMapper.generate_scientific_report(top_genes)

        report_path = config.RESULTS_DIR / "honest_science_report.md"
        with open(report_path, "w") as f:
            f.write("# Brutally Honest Scientific Validation: GSE10072\n\n")
            f.write("## Methodology\n")
            f.write("- **Group-Blind Validation:** No patient-level data leakage.\n")
            f.write("- **Proxy Filtering:** Removed Endothelial (VWF) and Inflammation markers.\n")
            f.write("- **Model:** Random Forest + Mutual Information (K=250).\n\n")
            f.write(f"**Final Honest Accuracy:** {acc:.4f}\n\n")
            f.write("## Identified Driver Genes\n")
            f.write(bio_report)
        
        logger.info(f"Scientific report generated at {report_path}")

    except Exception as e:
        logger.exception(f"Validation Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure current directory is in path
    sys.path.append(os.path.join(os.getcwd(), "cancer-cell-growth-prediction", "src"))
    run_real_world_validation()
