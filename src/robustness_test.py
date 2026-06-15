import numpy as np
import pandas as pd
import joblib
from core.config import config, logger
from sklearn.metrics import accuracy_score
from core.data_engine import BioDataLoader
from sklearn.model_selection import train_test_split

def run_robustness_test():
    """
    Evaluates model stability by injecting Gaussian noise into the input features.
    Real-world genomic data is noisy; a robust model should not collapse under minor perturbations.
    """
    logger.info("ROBUSTNESS TEST: Initiating Sensitivity Analysis...")
    
    # 1. Load data and model
    loader = BioDataLoader()
    X, y = loader.load_raw_data()
    
    pipeline_path = config.MODEL_DIR / "final_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)
    
    # Split to get a fresh test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    # 2. Baseline Accuracy
    y_pred_base = pipeline.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred_base)
    logger.info(f"Baseline Accuracy: {base_acc:.4f}")
    
    # 3. Noise Injection Levels
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    results = []
    
    for level in noise_levels:
        # Add noise relative to the standard deviation of each feature
        noise = np.random.normal(0, level, X_test.shape)
        X_noisy = X_test + noise
        
        y_pred_noisy = pipeline.predict(X_noisy)
        noisy_acc = accuracy_score(y_test, y_pred_noisy)
        
        drop = base_acc - noisy_acc
        results.append({"Noise Level": level, "Accuracy": noisy_acc, "Drop": drop})
        logger.info(f"Noise Level {level*100}%: Accuracy {noisy_acc:.4f} (Drop: {drop:.4f})")
    
    # 4. Save results
    results_df = pd.DataFrame(results)
    results_path = config.RESULTS_DIR / "robustness_report.md"
    
    with open(results_path, "w") as f:
        f.write("# Model Robustness & Sensitivity Analysis\n\n")
        f.write("This report evaluates if the model is overfitted to exact feature values or if it captures generalizable patterns.\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n## Conclusion\n")
        if results_df["Drop"].max() < 0.05:
            f.write("The model is **Robust**. Performance remains stable despite significant input noise.\n")
        else:
            f.write("The model shows **Sensitivity** to noise. Further regularization or feature engineering may be required for clinical environments.\n")

    logger.info(f"Robustness report generated at {results_path}")

if __name__ == "__main__":
    run_robustness_test()
