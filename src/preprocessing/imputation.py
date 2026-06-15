import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from src.utils.logging import logger

class MedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values with the median value of each feature on the training set.
    Preserves Pandas DataFrame structure (indices and columns).
    """
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.columns = None
        
    def fit(self, X: pd.DataFrame, y=None):
        self.columns = X.columns
        self.imputer.fit(X)
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("MedianImputer: Imputing missing values...")
        X_imputed = self.imputer.transform(X)
        return pd.DataFrame(X_imputed, index=X.index, columns=self.columns)
