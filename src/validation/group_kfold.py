import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from src.utils.logging import logger

def run_group_kfold(pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5) -> list[float]:
    """
    Executes group-blind validation using GroupKFold to ensure patients are
    never split across train/test folds.
    """
    logger.info(f"GroupKFold: Running {n_splits}-fold cross-validation...")
    gkf = GroupKFold(n_splits=n_splits)
    
    scores = []
    y_arr = y.values
    groups_arr = groups.values
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_arr, groups_arr)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y_arr[train_idx], y_arr[val_idx]
        
        # Clone or re-fit pipeline on train fold
        pipeline.fit(X_train_fold, y_train_fold)
        
        # Evaluate on validation fold
        preds = pipeline.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, preds)
        scores.append(acc)
        logger.info(f"GroupKFold: Fold {fold + 1}/{n_splits} - Accuracy: {acc*100:.2f}%")
        
    mean_acc = np.mean(scores)
    logger.info(f"GroupKFold: Completed. Mean Accuracy = {mean_acc*100:.2f}% (+/- {np.std(scores)*100:.2f}%)")
    return scores
