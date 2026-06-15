import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.validation.external import validate_external

def test_external_validation_alignment():
    """
    Verifies that validate_external correctly aligns overlapping features
    between different dataset schemas and calculates metrics without crashing.
    """
    np.random.seed(42)
    # Train dataset has features Gene_0 to Gene_5
    X_train = pd.DataFrame(np.random.rand(10, 6), columns=[f"Gene_{i}" for i in range(6)])
    y_train = pd.Series(np.random.choice([0, 1], size=10))
    
    # Test dataset has features Gene_3 to Gene_8 (overlap: Gene_3, Gene_4, Gene_5)
    X_test = pd.DataFrame(np.random.rand(5, 6), columns=[f"Gene_{i}" for i in range(3, 9)])
    y_test = pd.Series(np.random.choice([0, 1], size=5))
    
    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=5, random_state=42))
    ])
    
    res = validate_external(pipeline, X_train, y_train, X_test, y_test, "TestCohort")
    
    assert "accuracy" in res
    assert "recall" in res
    assert len(res["predictions"]) == 5
    assert 0.0 <= res["accuracy"] <= 1.0
