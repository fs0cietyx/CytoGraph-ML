import pandas as pd
import joblib
import numpy as np
from core.config import config, logger

def test_inference():
    """
    Simulates real-world inference using samples from the public TCGA dataset.
    """
    logger.info("TEST: Starting real-world inference verification...")
    
    # 1. Load the production pipeline
    pipeline_path = config.MODEL_DIR / "final_pipeline.joblib"
    if not pipeline_path.exists():
        print(f"Error: {pipeline_path} not found. Please run the training pipeline first.")
        return
    
    pipeline = joblib.load(pipeline_path)
    
    # 2. Load the data
    X = pd.read_csv(config.FEATURES_PATH, index_col=0)
    y = pd.read_csv(config.LABELS_PATH, index_col=0)
    
    # 3. Select 5 diverse samples (one of each cancer type if possible)
    classes = y['Class'].unique()
    test_indices = []
    for cls in classes:
        idx = y[y['Class'] == cls].index[0]
        test_indices.append(idx)
        
    X_test_samples = X.loc[test_indices]
    y_true_samples = y.loc[test_indices]
    
    # 4. Execute Prediction
    predictions = pipeline.predict(X_test_samples)
    probabilities = pipeline.predict_proba(X_test_samples)
    
    # 5. Report Results
    print("\n" + "="*80)
    print(f"{'Sample ID':<12} | {'True Label':<10} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 80)
    
    for i, idx in enumerate(test_indices):
        true_label = y_true_samples.loc[idx, 'Class']
        pred_label = predictions[i]
        conf = np.max(probabilities[i])
        
        print(f"{idx:<12} | {true_label:<10} | {pred_label:<10} | {conf:.4f}")
    
    print("="*80 + "\n")
    logger.info("TEST: Inference verification complete.")

if __name__ == "__main__":
    test_inference()
