from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from src.utils.logging import logger

def calculate_metrics(y_true, y_pred, y_probs=None) -> dict[str, float]:
    """
    Computes key performance metrics: Accuracy, Recall (Sensitivity), Precision, F1-Score, and ROC-AUC.
    """
    logger.info("Metrics: Calculating classification performance...")
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    if y_probs is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_probs))
        except Exception as e:
            logger.warning(f"Metrics: Could not calculate AUC: {e}")
            metrics["auc"] = 0.5
            
    return metrics
