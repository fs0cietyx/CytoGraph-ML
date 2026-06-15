import numpy as np
import pandas as pd
from src.preprocessing.feature_selection import MutualInformationSelector

def test_mutual_information_selector():
    """
    Verifies that the MutualInformationSelector selects the top K features
    and preserves dataframe column names.
    """
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(20, 10), columns=[f"Gene_{i}" for i in range(10)])
    y = pd.Series(np.random.choice([0, 1], size=20))
    
    # Inject strong target-correlation into Gene_0
    X["Gene_0"] = y.values.astype(float) * 5.0 + np.random.normal(0, 0.1, size=20)
    
    selector = MutualInformationSelector(k=3, random_state=42)
    selector.fit(X, y)
    
    assert len(selector.selected_features_) == 3
    # Verify Gene_0 is selected
    assert "Gene_0" in selector.selected_features_
    
    X_trans = selector.transform(X)
    assert X_trans.shape == (20, 3)
    assert list(X_trans.columns) == list(selector.selected_features_)
