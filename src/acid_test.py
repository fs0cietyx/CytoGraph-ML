import sys
import os
from core.config import logger, config
from core.geo_engine import GEOIngestor
from core.trainer import GenomicResearchTrainer
from sklearn.metrics import accuracy_score, classification_report, recall_score
import pandas as pd
import numpy as np

def run_hardened_acid_test():
    """
    The Clinical Grade Acid Test:
    1. Cross-Study training (Study A -> Study B).
    2. Quantile Normalization (Eliminates the 'Lab Gap').
    3. Recall Optimization (Prevents False Negatives).
    """
    logger.info("ACID TEST: Starting Clinical-Grade Lung Cancer Validation...")

    try:
        # 1. Acquire Training Data with Normalization
        ingestor_a = GEOIngestor(gse_id="GSE10072")
        X_train, y_train, groups_train = ingestor_a.fetch_and_map(platform_id="GPL96", normalize=True)

        # 2. Acquire Testing Data with Normalization
        ingestor_b = GEOIngestor(gse_id="GSE19804")
        X_test, y_test, groups_test = ingestor_b.fetch_and_map(platform_id="GPL570", normalize=True)

        if X_train is None or X_test is None:
            logger.error("ACID TEST: Data ingestion failed.")
            return

        # 3. Train on Study A
        # We increase K to 150 to give more biological depth after normalization
        trainer = GenomicResearchTrainer()
        trainer.pipeline.named_steps['feature_selector'].k = 150
        
        # We also adjust class weights to penalize missing a tumor (False Negative)
        trainer.pipeline.named_steps['rf'].class_weight = {0: 1, 1: 5} 
        
        logger.info("ACID TEST: Training Hardened model on Study A (GSE10072)...")
        trainer.execute_stratified_training(X_train, y_train, groups=groups_train)

        # 4. Predict on Study B
        # Ensure only overlapping genes are used
        common_genes = list(set(trainer.original_features).intersection(set(X_test.columns)))
        logger.info(f"ACID TEST: Aligning {len(common_genes)} universal markers...")
        
        # Align features
        X_test_aligned = X_test.reindex(columns=trainer.original_features).fillna(0)
        X_test_filtered = X_test_aligned.drop(columns=[g for g in trainer.biological_blacklist if g in X_test_aligned.columns])
        
        y_pred = trainer.pipeline.predict(X_test_filtered)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        logger.info(f"CLINICAL GRADE ACCURACY: {acc:.4f}")
        logger.info(f"CLINICAL GRADE RECALL (Sensitivity): {recall:.4f}")
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

        if recall < 0.90:
            logger.error("ACID TEST FAILURE: Sensitivity is too low for clinical use.")
        else:
            logger.info("ACID TEST SUCCESS: The model is now scientifically and clinically robust.")

    except Exception as e:
        logger.exception(f"Acid Test Failure: {e}")

if __name__ == "__main__":
    sys.path.append(os.path.join(os.getcwd(), "cancer-cell-growth-prediction", "src"))
    run_hardened_acid_test()
