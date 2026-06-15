from sklearn.metrics import confusion_matrix
from src.utils.logging import logger

def compute_confusion_matrix(y_true, y_pred) -> dict[str, int]:
    """
    Computes confusion matrix elements (TN, FP, FN, TP) for model validation.
    """
    logger.info("Confusion: Calculating confusion matrix...")
    try:
        cm = confusion_matrix(y_true, y_pred)
        # Handle cases where classes might be missing (e.g. all 0 or all 1 predictions)
        if cm.shape == (1, 1):
            val = int(cm[0, 0])
            # Check what class is predicted
            if list(set(y_true))[0] == 0:
                tn, fp, fn, tp = val, 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, val
        else:
            tn, fp, fn, tp = cm.ravel()
        return {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    except Exception as e:
        logger.error(f"Confusion: Calculation failed: {e}")
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
