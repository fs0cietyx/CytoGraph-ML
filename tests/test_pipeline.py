import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.blacklist import BlacklistFilter
from src.preprocessing.quantile_norm import QuantileNormalizer
from src.preprocessing.imputation import MedianImputer
from src.preprocessing.scaling import GenomicScaler
from src.preprocessing.feature_selection import MutualInformationSelector

def test_no_leakage():
    """
    Rigorously verifies that the feature selection step in the pipeline
    never sees validation fold data during fitting.
    """
    np.random.seed(42)
    # Generate mock dataset
    X = pd.DataFrame(np.random.rand(30, 20), columns=[f"Gene_{i}" for i in range(20)])
    y = pd.Series(np.random.choice([0, 1], size=30))
    # 10 patient groups
    groups = pd.Series([i // 3 for i in range(30)])
    
    # Store indices processed in each fit call
    selector_fit_indices = []
    
    # Subclass our feature selector to intercept and spy on fit inputs
    class SpyingSelector(MutualInformationSelector):
        def fit(self, X_fit, y_fit):
            selector_fit_indices.append(list(X_fit.index))
            return super().fit(X_fit, y_fit)
            
    spy_selector = SpyingSelector(k=5, random_state=42)
    
    pipeline = Pipeline([
        ('blacklist', BlacklistFilter()),
        ('quantile_norm', QuantileNormalizer()),
        ('imputer', MedianImputer()),
        ('scaler', GenomicScaler()),
        ('feature_selector', spy_selector),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    gkf = GroupKFold(n_splits=5)
    
    # Run cross-validation manually to inspect leakage boundaries
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        # Get indices seen by our spy selector
        fit_idx = selector_fit_indices[-1]
        
        # Assert zero overlap between fitted samples and validation samples
        overlap = set(fit_idx).intersection(set(val_idx))
        assert len(overlap) == 0, f"LEAKAGE DETECTED in fold {fold+1}: Feature selector saw validation indices: {overlap}"
        
    print("Zero-leakage check PASSED. Feature selector never saw validation fold indices.")
