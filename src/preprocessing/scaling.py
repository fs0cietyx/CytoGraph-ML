import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from src.utils.logging import logger

class GenomicScaler(BaseEstimator, TransformerMixin):
    """
    Scales features robustly using median and interquartile range.
    Preserves Pandas DataFrame structure (indices and columns).
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.columns = None
        
    def fit(self, X: pd.DataFrame, y=None):
        self.columns = X.columns
        self.scaler.fit(X)
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("GenomicScaler: Scaling features robustly...")
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, index=X.index, columns=self.columns)
