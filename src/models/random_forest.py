from sklearn.ensemble import RandomForestClassifier

def get_random_forest(n_estimators: int = 100, class_weight = None, random_state: int = 42, n_jobs: int = -1):
    """Instantiates and returns a Random Forest Classifier."""
    if class_weight is None:
        class_weight = {0: 1, 1: 5}
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
