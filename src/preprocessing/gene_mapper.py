import pandas as pd
import GEOparse
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logging import logger
from src.utils.io import METADATA_DIR

class GeneMapper(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that maps Probe IDs to Gene Symbols
    using NCBI platform metadata.
    """
    def __init__(self, platform_id: str = "GPL96"):
        self.platform_id = platform_id
        self.probe_to_symbol = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        logger.info(f"GeneMapper: Loading platform annotations for {self.platform_id}...")
        gpl = GEOparse.get_GEO(geo=self.platform_id, destdir=METADATA_DIR, silent=True)
        symbol_col = 'Gene Symbol' if 'Gene Symbol' in gpl.table.columns else 'Symbol'
        self.probe_to_symbol = gpl.table.set_index('ID')[symbol_col].to_dict()
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("GeneMapper: Mapping Probe IDs to Gene Symbols...")
        X_mapped = X.rename(columns=self.probe_to_symbol)
        
        # Drop duplicates and invalid columns
        X_mapped = X_mapped.loc[:, ~X_mapped.columns.duplicated()]
        X_mapped = X_mapped.loc[:, ~X_mapped.columns.str.contains('---|^$', na=True)]
        X_mapped = X_mapped.loc[:, X_mapped.columns.notna()]
        
        logger.info(f"GeneMapper: Reduced columns from {X.shape[1]} to {X_mapped.shape[1]}")
        return X_mapped
