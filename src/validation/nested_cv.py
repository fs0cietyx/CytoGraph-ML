import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from src.utils.logging import logger

def run_nested_cv(pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series, param_grid: dict, outer_splits: int = 5, inner_splits: int = 3) -> dict:
    """
    Runs nested cross-validation with group-blind folds.
    Tuning hyper-parameters in the inner loop and evaluating model generalizability in the outer loop.
    """
    logger.info(f"NestedCV: Running outer={outer_splits}-fold and inner={inner_splits}-fold Group-Blind Nested CV...")
    outer_cv = GroupKFold(n_splits=outer_splits)
    inner_cv = GroupKFold(n_splits=inner_splits)
    
    nested_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y, groups)):
        X_train_out, X_val_out = X.iloc[train_idx], X.iloc[val_idx]
        y_train_out, y_val_out = y.iloc[train_idx], y.iloc[val_idx]
        groups_out = groups.iloc[train_idx]
        
        # Set up grid search for inner loop
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit inner CV using group constraints
        grid_search.fit(X_train_out, y_train_out, groups=groups_out)
        
        # Predict on outer loop validation fold
        best_model = grid_search.best_estimator_
        preds = best_model.predict(X_val_out)
        acc = accuracy_score(y_val_out, preds)
        nested_scores.append(acc)
        
        logger.info(f"NestedCV: Outer Fold {fold + 1}/{outer_splits} - Best params: {grid_search.best_params_} - Accuracy: {acc*100:.2f}%")
        
    mean_nested = np.mean(nested_scores)
    logger.info(f"NestedCV: Completed. Nested CV Accuracy: {mean_nested*100:.2f}% (+/- {np.std(nested_scores)*100:.2f}%)")
    return {
        "nested_scores": nested_scores,
        "mean_nested_accuracy": mean_nested
    }
