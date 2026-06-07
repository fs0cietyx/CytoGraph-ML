import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import joblib
from typing import Dict, Any
from .config import logger, config

class APEXTrainer:
    """Enterprise ML Trainer with Stratified Validation, Anti-Leakage Pipelines & Deterministic Locking."""
    
    def __init__(self):
        self.config = config
        
        # DEFENSIVE PATCH: Impenetrable Pipeline
        # Ensures imputation, scaling, and feature selection are fitted ONLY on training folds.
        self.pipeline = Pipeline([
            ('variance_filter', VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)),
            ('imputer', SimpleImputer(strategy='median')), # Handles sparse/NaN adversarial inputs
            ('scaler', RobustScaler()), # Handles extreme biological outliers
            ('rf', RandomForestClassifier(
                n_estimators=config.RF_ESTIMATORS,
                max_features=config.RF_MAX_FEATURES,
                n_jobs=config.RF_N_JOBS,
                random_state=config.RANDOM_STATE
            ))
        ])

    def execute_stratified_training(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Executes N-Fold Stratified Cross-Validation and final pipeline fitting.
        """
        logger.info(f"Initiating Stratified {config.N_FOLD}-Fold Cross-Validation...")
        
        skf = StratifiedKFold(n_splits=config.N_FOLD, shuffle=True, random_state=config.RANDOM_STATE)
        
        # Cross-validation is now leakage-free because the pipeline fits per fold
        scores = cross_val_score(self.pipeline, X, y, cv=skf, scoring='accuracy')
        
        logger.info(f"CV Accuracy Scores: {scores}")
        logger.info(f"Mean CV Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        # Final Fit on entire training set
        logger.info("Fitting final production pipeline...")
        self.pipeline.fit(X, y)
        
        # Save original feature names to the pipeline for later mapping
        self.original_features = X.columns
        
        # Serialization
        model_path = config.MODEL_DIR / "final_pipeline.joblib"
        joblib.dump(self.pipeline, model_path)
        
        # Persist base features for API schema validation
        feature_list_path = config.MODEL_DIR / "base_features.joblib"
        joblib.dump(self.original_features.tolist(), feature_list_path)
        
        logger.info(f"Production pipeline serialized to {model_path}")
        
        return {
            "mean_cv_accuracy": np.mean(scores),
            "std_cv_accuracy": np.std(scores),
            "model_path": model_path
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Extracts Gini importance mapping back through the VarianceThreshold mask."""
        rf_model = self.pipeline.named_steps['rf']
        var_filter = self.pipeline.named_steps['variance_filter']
        
        # Get the mask of features that survived the variance threshold
        surviving_features_mask = var_filter.get_support()
        surviving_feature_names = self.original_features[surviving_features_mask]
        
        importances = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            "Gene": surviving_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        return importance_df
