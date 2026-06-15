import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from src.utils.logging import logger

def validate_external(pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, cohort_name: str) -> dict:
    """
    Evaluates the pipeline on an independent external dataset.
    Aligns features between train and test datasets, re-fits the pipeline, and returns performance metrics.
    """
    logger.info(f"External: Validating model on external cohort {cohort_name}...")
    
    # Find overlapping features between training dataset and external dataset
    common_genes = list(set(X_train.columns).intersection(set(X_test.columns)))
    logger.info(f"External: Aligned {len(common_genes)} common genes between training and external cohort.")
    
    X_train_aligned = X_train.reindex(columns=common_genes).fillna(0)
    X_test_aligned = X_test.reindex(columns=common_genes).fillna(0)
    
    # Fit the pipeline on the aligned training dataset
    pipeline.fit(X_train_aligned, y_train)
    
    # Predict and evaluate on the external cohort
    preds = pipeline.predict(X_test_aligned)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds, zero_division=0)
    
    logger.info(f"External ({cohort_name}) -> Accuracy: {acc*100:.2f}%, Recall: {rec:.2f}")
    return {
        "accuracy": acc,
        "recall": rec,
        "predictions": preds
    }

def validate_gse19804(pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluates GSE19804 (External Lung Cohort)."""
    return validate_external(pipeline, X_train, y_train, X_test, y_test, "GSE19804")

def validate_gse21510(pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluates GSE21510 (External Colorectal Shift Cohort)."""
    return validate_external(pipeline, X_train, y_train, X_test, y_test, "GSE21510")
