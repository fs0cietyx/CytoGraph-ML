import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from src.utils.logging import logger

def run_holdout_validation(pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, test_size: float = 0.3, random_state: int = 42) -> float:
    """
    Splits the dataset using GroupShuffleSplit, trains the model on the train split,
    and evaluates it on the independent holdout split.
    """
    logger.info(f"Holdout: Running holdout validation (split={test_size})...")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, holdout_idx = next(gss.split(X, y, groups))
    
    X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
    y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_holdout)
    acc = accuracy_score(y_holdout, preds)
    
    logger.info(f"Holdout: Validation completed. Accuracy: {acc*100:.2f}%")
    return acc
