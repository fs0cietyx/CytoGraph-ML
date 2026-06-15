import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.constants import BIOLOGICAL_BLACKLIST
from src.utils.logging import logger

class BlacklistFilter(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that filters out biological proxy genes
    (housekeeping, endothelial, inflammatory genes) from expression data.
    """
    def __init__(self, blacklist=None):
        self.blacklist = blacklist if blacklist is not None else BIOLOGICAL_BLACKLIST
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"BlacklistFilter: Removing proxy markers (n={len(self.blacklist)})...")
        cols_to_drop = [col for col in self.blacklist if col in X.columns]
        X_filtered = X.drop(columns=cols_to_drop)
        logger.info(f"BlacklistFilter: Dropped {len(cols_to_drop)} columns. Remaining: {X_filtered.shape[1]}")
        return X_filtered
