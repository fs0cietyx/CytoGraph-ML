import pandas as pd
import joblib
import numpy as np
from core.config import config, logger
from sklearn.metrics import classification_report, confusion_matrix

def batch_stress_test(n_samples: int = 100):
    """
    Executes a high-volume batch test on random samples from the public TCGA dataset.
    """
    logger.info(f"STRESS TEST: Starting batch inference on {n_samples} random samples...")
    
    # 1. Load the production pipeline
    pipeline_path = config.MODEL_DIR / "final_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)
    
    # 2. Load the data
    X = pd.read_csv(config.FEATURES_PATH, index_col=0)
    y = pd.read_csv(config.LABELS_PATH, index_col=0)
    
    # 3. Pick N random samples
    # We'll use a different random seed than the training to ensure "unseen" data points are likely
    test_indices = y.sample(n=min(n_samples, len(y)), random_state=99).index
    
    X_test_samples = X.loc[test_indices]
    y_true_samples = y.loc[test_indices].values.ravel()
    
    # 4. Execute Prediction
    predictions = pipeline.predict(X_test_samples)
    probabilities = pipeline.predict_proba(X_test_samples)
    
    # 5. Report Results
    accuracy = (predictions == y_true_samples).mean()
    
    print("\n" + "="*80)
    print(f"BATCH STRESS TEST RESULTS ({len(test_indices)} SAMPLES)")
    print("-" * 80)
    print(f"Overall Batch Accuracy: {accuracy*100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true_samples, predictions))
    print("="*80 + "\n")
    
    logger.info("STRESS TEST: Batch inference complete.")

if __name__ == "__main__":
    batch_stress_test(100)
