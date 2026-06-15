import numpy as np
import pandas as pd
from src.preprocessing.quantile_norm import QuantileNormalizer

def test_quantile_normalizer():
    """
    Verifies that the QuantileNormalizer correctly aligns distributions
    without leakage by projecting test data onto training reference distributions.
    """
    np.random.seed(42)
    # Generate mock training data (mean = 5) and testing data (mean = 10)
    X_train = pd.DataFrame(np.random.normal(5.0, 1.0, size=(10, 5)))
    X_test = pd.DataFrame(np.random.normal(10.0, 2.0, size=(5, 5)))
    
    qn = QuantileNormalizer()
    qn.fit(X_train)
    
    # Assert fitted reference distribution has correct size
    assert qn.reference_distribution_ is not None
    assert len(qn.reference_distribution_) == 5
    
    # Transform test set
    X_test_norm = qn.transform(X_test)
    
    # Verify shape is preserved
    assert X_test_norm.shape == X_test.shape
    
    # Verify sorted test values are mathematically mapped to reference train distributions
    for i in range(X_test_norm.shape[0]):
        sorted_row = np.sort(X_test_norm.values[i])
        np.testing.assert_array_almost_equal(sorted_row, qn.reference_distribution_)
