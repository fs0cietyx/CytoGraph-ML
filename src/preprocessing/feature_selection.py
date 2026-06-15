import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from src.utils.logging import logger

class MIScoreCallable:
    """Module-level callable to make the mutual information score function picklable."""
    def __init__(self, random_state):
        self.random_state = random_state
        
    def __call__(self, X, y):
        return mutual_info_classif(X, y, random_state=self.random_state)

class MutualInformationSelector(BaseEstimator, TransformerMixin):
    """
    Selects top K features using Mutual Information.
    Preserves Pandas DataFrame structure (indices and selected columns).
    """
    def __init__(self, k: int = 250, random_state: int = 42):
        self.k = k
        self.random_state = random_state
        self.selector = SelectKBest(score_func=MIScoreCallable(self.random_state), k=self.k)
        self.selected_features_ = None
        
    def fit(self, X: pd.DataFrame, y):
        logger.info(f"MutualInformationSelector: Computing Mutual Information to select top K={self.k} features...")
        self.selector.fit(X, y)
        support = self.selector.get_support()
        self.selected_features_ = X.columns[support]
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise ValueError("MutualInformationSelector must be fitted before transforming.")
        logger.info("MutualInformationSelector: Subsetting selected features...")
        # Handle cases where some features are not present in X (e.g. cross-platform validation)
        # Use reindex to keep the same columns, filling with 0 if missing
        X_selected = X.reindex(columns=self.selected_features_).fillna(0)
        return X_selected
