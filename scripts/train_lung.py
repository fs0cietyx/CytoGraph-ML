import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import logger, set_seed, load_config, MODEL_DIR, RESULTS_DIR
from src.preprocessing import GEODataLoader, GeneMapper
from src.pipelines import create_lung_pipeline
from src.validation import run_group_kfold, run_holdout_validation
from src.evaluation import calculate_metrics

def main():
    set_seed(42)
    logger.info("================== TRAINING LUNG MODEL (GSE10072) ==================")
    
    # Load config
    config = load_config("lung")
    
    # Load dataset
    X_raw, y, groups = GEODataLoader.load_gse10072()
    
    # Map probes to symbols
    mapper = GeneMapper(platform_id="GPL96")
    X = mapper.fit_transform(X_raw)
    
    # Build pipeline
    pipeline = create_lung_pipeline(config)
    
    # Run Group-Blind CV
    cv_scores = run_group_kfold(pipeline, X, y, groups, n_splits=config["cv"]["folds"])
    
    # Run holdout split validation
    holdout_acc = run_holdout_validation(
        pipeline, X, y, groups, 
        test_size=config["cv"]["test_size"], 
        random_state=config["random_seed"]
    )
    
    # Fit final model on all data
    logger.info("Fitting final lung cancer pipeline on complete GSE10072 cohort...")
    pipeline.fit(X, y)
    
    # Save final pipeline and feature selector
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = MODEL_DIR / "rf_lung.pkl"
    joblib.dump(pipeline, model_path)
    logger.info(f"Model successfully saved to {model_path}")
    
    selector_path = MODEL_DIR / "feature_selector.pkl"
    joblib.dump(pipeline.named_steps["feature_selector"], selector_path)
    logger.info(f"Feature selector successfully saved to {selector_path}")
    
    # Save results
    os.makedirs(RESULTS_DIR / "cv", exist_ok=True)
    os.makedirs(RESULTS_DIR / "holdout", exist_ok=True)
    
    cv_df = pd.DataFrame({"Fold": range(1, len(cv_scores) + 1), "Accuracy": cv_scores})
    cv_df.to_csv(RESULTS_DIR / "cv" / "lung_cv_scores.csv", index=False)
    
    holdout_df = pd.DataFrame({"Metric": ["Holdout Accuracy"], "Value": [holdout_acc]})
    holdout_df.to_csv(RESULTS_DIR / "holdout" / "lung_holdout_scores.csv", index=False)
    
    logger.info("Lung model training and evaluation complete.")
    print(f"Expected CV Accuracy: {np.mean(cv_scores)*100:.2f}%")
    print(f"Holdout Accuracy: {holdout_acc*100:.2f}%")

if __name__ == "__main__":
    main()
