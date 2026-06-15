import pandas as pd
import joblib
from typing import Tuple, List
from .config import logger, config

class BioDataLoader:
    """Clinical Data Ingestion Layer for Biological Datasets."""
    
    def __init__(self):
        self.features_path = config.FEATURES_PATH
        self.labels_path = config.LABELS_PATH

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads and validates raw genomic expression data.
        Returns: Tuple of (Features, Labels)
        """
        logger.info(f"Ingesting features from {self.features_path}")
        try:
            X = pd.read_csv(self.features_path, index_col=0)
            y = pd.read_csv(self.labels_path, index_col=0)
            
            # Data Integrity Check
            if X.shape[0] != y.shape[0]:
                raise ValueError("Row mismatch between features and labels.")
            
            # Ensure y is a Series
            y_series = y.iloc[:, 0]
            
            logger.info(f"Successfully loaded {X.shape[0]} samples with {X.shape[1]} features.")
            return X, y_series
            
        except FileNotFoundError as e:
            logger.error(f"Critical Data Failure: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected Data Error: {e}")
            raise

class BioPreprocessor:
    """Hardened Bioinformatic Preprocessing Engine."""
    
    def __init__(self):
        self.random_state = config.RANDOM_STATE
        self.variance_threshold = config.VARIANCE_THRESHOLD

    def clean_and_subset(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Implements variance-based feature selection to remove low-information genes.
        Saves selected feature names for API inference alignment.
        """
        logger.info("Executing variance thresholding...")
        initial_count = X.shape[1]
        
        # Variance filter
        variances = X.var()
        keep_cols = variances[variances > self.variance_threshold].index
        X_filtered = X[keep_cols]
        
        # Persist selected features for inference consistency
        feature_list_path = config.MODEL_DIR / "selected_features.joblib"
        joblib.dump(keep_cols.tolist(), feature_list_path)
        
        dropped = initial_count - X_filtered.shape[1]
        logger.info(f"Feature Selection: Removed {dropped} low-variance genes. {X_filtered.shape[1]} remaining.")
        logger.info(f"Selected features persisted to {feature_list_path}")
        
        return X_filtered
