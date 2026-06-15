import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import logger, set_seed, load_config, MODEL_DIR, RESULTS_DIR
from src.preprocessing import GEODataLoader, GeneMapper
from src.pipelines import create_colorectal_pipeline
from src.validation import run_group_kfold, run_holdout_validation

def main():
    set_seed(42)
    logger.info("================== TRAINING COLORECTAL MODEL (GSE21510) ==================")
    
    # Load config
    config = load_config("colorectal")
    
    # Load dataset
    X_raw, y, groups = GEODataLoader.load_gse21510()
    
    # Map probes to symbols
    mapper = GeneMapper(platform_id="GPL570")
    X = mapper.fit_transform(X_raw)
    
    # Build pipeline
    pipeline = create_colorectal_pipeline(config)
    
    # Run Group-Blind CV
    cv_scores = run_group_kfold(pipeline, X, y, groups, n_splits=config["cv"]["folds"])
    
    # Run holdout split validation
    holdout_acc = run_holdout_validation(
        pipeline, X, y, groups, 
        test_size=config["cv"]["test_size"], 
        random_state=config["random_seed"]
    )
    
    # Fit final model
    logger.info("Fitting final colorectal cancer pipeline on complete GSE21510 cohort...")
    pipeline.fit(X, y)
    
    # Save final pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = MODEL_DIR / "rf_colorectal.pkl"
    joblib.dump(pipeline, model_path)
    logger.info(f"Model successfully saved to {model_path}")
    
    # Save results
    os.makedirs(RESULTS_DIR / "cv", exist_ok=True)
    os.makedirs(RESULTS_DIR / "holdout", exist_ok=True)
    
    cv_df = pd.DataFrame({"Fold": range(1, len(cv_scores) + 1), "Accuracy": cv_scores})
    cv_df.to_csv(RESULTS_DIR / "cv" / "colorectal_cv_scores.csv", index=False)
    
    holdout_df = pd.DataFrame({"Metric": ["Holdout Accuracy"], "Value": [holdout_acc]})
    holdout_df.to_csv(RESULTS_DIR / "holdout" / "colorectal_holdout_scores.csv", index=False)
    
    logger.info("Colorectal model training and evaluation complete.")
    print(f"Colorectal CV Accuracy: {np.mean(cv_scores)*100:.2f}%")
    print(f"Colorectal Holdout Accuracy: {holdout_acc*100:.2f}%")

if __name__ == "__main__":
    main()
