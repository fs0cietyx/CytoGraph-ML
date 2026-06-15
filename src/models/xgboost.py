from xgboost import XGBClassifier

def get_xgboost(n_estimators: int = 100, random_state: int = 42, n_jobs: int = -1):
    """Instantiates and returns an XGBoost Classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=n_jobs
    )
