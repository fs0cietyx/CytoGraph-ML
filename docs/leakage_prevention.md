# Leakage Prevention & High-Integrity Validation

Data leakage is one of the most common reasons genomic machine learning models fail to translate to clinical settings. CytoGraph-ML implements three strict defense boundaries to ensure scientific integrity.

## 1. Group-Blind Cross-Validation

Standard cross-validation splits samples randomly. However, when patients contribute multiple tissue samples (e.g., matching tumor and normal tissue), random splits leak patient-specific genetic signatures (the "twin-study leak"). 

Our framework uses `GroupKFold` and `GroupShuffleSplit`, grouping by patient ID. This guarantees that all samples from a single patient are strictly isolated to either the training set or the validation/test set.

## 2. Fit-Transform Preprocessing Separation

To prevent feature selection and normalization leakage:
- Preprocessing steps (imputer, scaler, selector, normalizer) are fit **only** on the training folds.
- The fitted parameters (median values, scaling ranges, selected feature masks, and reference quantile distributions) are then applied to the validation fold during `transform`.

This is programmatically proven in our test suite:
```python
def test_no_leakage():
    # ...
    # Asserts that validation indices are never passed to selector.fit()
```

## 3. Permutation Audits

Our validation suite includes a label-shuffling audit:
- Shuffles target labels ($y$) to destroy genuine biological signals.
- Runs the training pipeline on this random noise.
- Verifies that model accuracy drops to random chance ($\approx 50\%$). A high accuracy on shuffled labels instantly triggers an audit warning.
