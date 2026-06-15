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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import joblib
from typing import Dict, Any
from .config import logger, config

class GenomicResearchTrainer:
    """
    High-Integrity Research Trainer.
    Implements Group-Blind Validation and Biological Proxy Filtering.
    """
    
    def __init__(self):
        self.config = config
        
        # PROXY FILTER: Blacklist of non-specific biological markers
        # These genes are 'smoke' (inflammation/blood vessels) that mask real oncogenesis.
        self.biological_blacklist = [
            'VWF', 'PECAM1', 'CD34', 'ENG', 'CDH5', # Endothelial/Blood Vessel markers
            'IL6', 'TNF', 'CRP', 'CCL2', 'IL8',     # General Inflammation
            'ACTB', 'GAPDH', 'B2M', 'ALB'          # Housekeeping/Stroma
        ]
        
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('feature_selector', SelectKBest(score_func=mutual_info_classif, k=250)),
            ('rf', RandomForestClassifier(
                n_estimators=config.RF_ESTIMATORS,
                max_features=config.RF_MAX_FEATURES,
                n_jobs=config.RF_N_JOBS,
                random_state=config.RANDOM_STATE
            ))
        ])

    def execute_stratified_training(self, X: pd.DataFrame, y: pd.Series, groups=None) -> Dict[str, Any]:
        """
        Executes Group-Blind N-Fold Cross-Validation.
        Ensures patients are NEVER split across train/test boundaries.
        """
        # 1. Apply Biological Filter
        logger.info(f"Applying Biological Filter: Removing {len(self.biological_blacklist)} proxy markers...")
        X_filtered = X.drop(columns=[g for g in self.biological_blacklist if g in X.columns])
        
        # 2. Group-Blind Validation
        logger.info(f"Initiating Group-Blind {config.N_FOLD}-Fold Cross-Validation...")
        gkf = GroupKFold(n_splits=config.N_FOLD)
        
        y_numeric = np.array(y).astype(int)
        
        # Perform CV with Group awareness
        scores = cross_val_score(
            self.pipeline, X_filtered, y_numeric, 
            cv=gkf, groups=groups, scoring='accuracy'
        )
        
        logger.info(f"Honest CV Accuracy Scores: {scores}")
        logger.info(f"Mean Honest Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        # 3. Final Fit
        logger.info("Fitting final scientifically-hardened pipeline...")
        self.pipeline.fit(X_filtered, y_numeric)
        
        self.original_features = X_filtered.columns
        
        # Serialization
        model_path = config.MODEL_DIR / "final_pipeline.joblib"
        joblib.dump(self.pipeline, model_path)
        
        return {
            "mean_cv_accuracy": np.mean(scores),
            "model_path": model_path
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Extracts Gini importance mapping back through the SelectKBest mask."""
        rf_model = self.pipeline.named_steps['rf']
        selector = self.pipeline.named_steps['feature_selector']
        
        surviving_features_mask = selector.get_support()
        surviving_feature_names = self.original_features[surviving_features_mask]
        
        importances = rf_model.feature_importances_
        
        importance_df = pd.DataFrame({
            "Gene": surviving_feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        return importance_df
