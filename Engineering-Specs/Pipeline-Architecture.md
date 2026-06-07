# 🏗️ APEX Pipeline Architecture

The **APEX Protocol** utilizes a strictly isolated `sklearn.pipeline.Pipeline` to ensure statistical integrity and zero data leakage.

## 🧱 Components
1. **Imputer**: `SimpleImputer(strategy='median')` - Handles missing data in biological samples.
2. **Scaler**: `RobustScaler()` - Mitigates the impact of physically impossible biological outliers.
3. **Feature Filter**: `VarianceThreshold(0.01)` - Eradicates low-information "noise" genes.
4. **Classifier**: `RandomForestClassifier` - 100 Estimators, deterministic random seed (42).

## 🛡️ Stability Features
- Fits transformations ONLY on training folds during [[Audit-Logs/Stability-Audit|Cross-Validation]].
- Vectorized batch inference support.

---
**Backlinks:** [[00-Index-MOC]] | [[decisions]]
