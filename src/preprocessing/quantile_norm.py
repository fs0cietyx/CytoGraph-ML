import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logging import logger

class QuantileNormalizer(BaseEstimator, TransformerMixin):
    """
    Leakage-free Quantile Normalizer.
    Fits a reference distribution on the training set (sorted mean values)
    and projects validation/testing datasets onto this reference.
    """
    def __init__(self):
        self.reference_distribution_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        logger.info("QuantileNormalizer: Fitting reference distribution on training data...")
        # Sort values of each sample (along axis 1, columns)
        sorted_X = np.sort(X.values, axis=1)
        # Take the mean of sorted values across all samples (along axis 0)
        self.reference_distribution_ = np.mean(sorted_X, axis=0)
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.reference_distribution_ is None:
            # Fallback to local quantile normalization if not fitted (e.g. static/standalone run)
            logger.warning("QuantileNormalizer: Not fitted! Performing sample-wise self-normalization.")
            sorted_X = np.sort(X.values, axis=1)
            ref = np.mean(sorted_X, axis=0)
        else:
            ref = self.reference_distribution_
            
        logger.info("QuantileNormalizer: Normalizing expression values...")
        X_arr = X.values
        normalized_X = np.empty_like(X_arr)
        
        # Project each sample onto the reference distribution based on rank
        for i in range(X_arr.shape[0]):
            row = X_arr[i]
            # rank returns 1-indexed ranks, subtract 1 to get 0-indexed indices
            ranks = pd.Series(row).rank(method='first').values.astype(int) - 1
            # Bound ranks to reference distribution size just in case
            ranks = np.clip(ranks, 0, len(ref) - 1)
            normalized_X[i] = ref[ranks]
            
        return pd.DataFrame(normalized_X, index=X.index, columns=X.columns)
