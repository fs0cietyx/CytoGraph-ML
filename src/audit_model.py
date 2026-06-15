import numpy as np
import pandas as pd
import joblib
from core.config import config, logger
from core.geo_engine import GEOIngestor
from core.trainer import GenomicResearchTrainer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def run_permutation_audit():
    """
    Brutal Honesty Audit: The Permutation Test.
    If we shuffle the labels and the model still gets high accuracy, the pipeline has a leak.
    A real model should get ~50% (random chance) on shuffled labels.
    """
    logger.info("AUDIT: Initiating Permutation Test (Label Shuffling)...")
    
    # 1. Load Data
    ingestor = GEOIngestor(gse_id="GSE10072")
    X, y, groups = ingestor.fetch_and_map()
    
    # 2. Shuffle the labels
    y_shuffled = np.random.permutation(y.values.ravel())
    
    # 3. Split and Train
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y_shuffled, groups, test_size=0.3, random_state=config.RANDOM_STATE
    )
    
    trainer = GenomicResearchTrainer()
    logger.info("Training on SHUFFLED labels...")
    # Reduce K to speed up the audit
    trainer.pipeline.named_steps['feature_selector'].k = 100
    
    trainer.execute_stratified_training(X_train, y_train, groups=groups_train)
    
    # 4. Evaluate
    # Ensure biological blacklist is dropped from X_test during prediction
    X_test_filtered = X_test.drop(columns=[g for g in trainer.biological_blacklist if g in X_test.columns])
    y_pred = trainer.pipeline.predict(X_test_filtered)
    shuffled_acc = accuracy_score(y_test, y_pred)
    
    logger.info(f"SHUFFLED ACCURACY: {shuffled_acc:.4f}")
    
    if shuffled_acc > 0.65:
        logger.error("AUDIT FAILURE: High accuracy on random labels suggests Data Leakage!")
    else:
        logger.info("AUDIT PASS: Model correctly failed to find patterns in noise.")

if __name__ == "__main__":
    run_permutation_audit()
