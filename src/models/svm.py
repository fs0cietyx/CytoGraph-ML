from sklearn.svm import SVC

def get_svm(kernel: str = 'linear', probability: bool = True, C: float = 1.0, random_state: int = 42):
    """Instantiates and returns a Support Vector Classifier."""
    return SVC(
        kernel=kernel,
        probability=probability,
        C=C,
        random_state=random_state
    )
