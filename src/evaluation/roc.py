from sklearn.metrics import roc_curve, auc
from src.utils.logging import logger

def compute_roc_curve(y_true, y_probs) -> dict:
    """
    Computes false positive rates, true positive rates, and AUC for plotting ROC curves.
    """
    logger.info("ROC: Computing ROC curve coordinates...")
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        roc_auc = float(auc(fpr, tpr))
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": roc_auc
        }
    except Exception as e:
        logger.error(f"ROC: Curve computation failed: {e}")
        return {
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "auc": 0.5
        }
