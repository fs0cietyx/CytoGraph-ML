from sklearn.linear_model import LogisticRegression

def get_logistic_regression(penalty: str = 'l2', C: float = 1.0, random_state: int = 42, max_iter: int = 1000):
    """Instantiates and returns a Logistic Regression Classifier."""
    if penalty == 'elasticnet':
        return LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            C=C,
            random_state=random_state,
            max_iter=max_iter
        )
    return LogisticRegression(
        penalty=penalty,
        C=C,
        random_state=random_state,
        max_iter=max_iter
    )
