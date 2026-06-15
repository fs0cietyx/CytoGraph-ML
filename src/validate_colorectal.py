import sys
import os
from core.config import logger, config
from core.geo_engine import GEOIngestor
from core.trainer import GenomicResearchTrainer
from core.bio_mapper import BioMapper
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def run_colorectal_validation():
    """
    Cross-Dataset Generalizability Test:
    Testing the pipeline on GSE21510 (Colorectal Cancer).
    Platform: GPL570.
    """
    logger.info("VALIDATION: Starting Colorectal Cancer Validation (GSE21510)...")

    try:
        # 1. Real Data Acquisition
        # GSE21510: Colorectal Cancer vs Normal
        # Platform: GPL570
        ingestor = GEOIngestor(gse_id="GSE21510")
        X, y, groups = ingestor.fetch_and_map(platform_id="GPL570")

        if X is None or y is None:
            logger.error("VALIDATION: Data ingestion failed. Aborting.")
            return

        logger.info(f"Dataset Shape: {X.shape}")
        
        # 2. Group-Blind Holdout Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=config.RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        
        logger.info(f"Group-Blind Split: {len(X_train)} Train samples, {len(X_test)} Test samples.")

        # 3. Training
        trainer = GenomicResearchTrainer()
        results = trainer.execute_stratified_training(X_train, y_train, groups=groups_train)

        # 4. Evaluation
        X_test_filtered = X_test.drop(columns=[g for g in trainer.biological_blacklist if g in X_test.columns])
        y_pred = trainer.pipeline.predict(X_test_filtered)
        acc = accuracy_score(y_test, y_pred)
        
        logger.info(f"CROSS-DATASET ACCURACY (GSE21510 - Colorectal): {acc:.4f}")
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

        # 5. Biological Enrichment
        importance_df = trainer.get_feature_importance()
        top_genes = importance_df.head(10)["Gene"].tolist()
        
        logger.info(f"Identified Colorectal Biomarkers: {top_genes}")
        bio_report = BioMapper.generate_scientific_report(top_genes)

        report_path = config.RESULTS_DIR / "colorectal_validation_report.md"
        with open(report_path, "w") as f:
            f.write("# Cross-Dataset Validation: GSE21510 (Colorectal Cancer)\n\n")
            f.write(f"**Target:** Colorectal Cancer vs Normal Tissue\n")
            f.write(f"**Honest Accuracy:** {acc:.4f}\n\n")
            f.write("## Identified Colorectal Biomarkers\n")
            f.write(bio_report)
        
        logger.info(f"Validation report generated at {report_path}")

    except Exception as e:
        logger.exception(f"Validation Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.path.append(os.path.join(os.getcwd(), "cancer-cell-growth-prediction", "src"))
    run_colorectal_validation()
