from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from src.utils.logging import logger

def compute_calibration(y_true, y_probs, n_bins: int = 5) -> dict:
    """
    Computes calibration curve bins and the Brier score loss to evaluate
    probability prediction accuracy.
    """
    logger.info("Calibration: Calculating calibration curve and Brier score...")
    
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)
        brier = float(brier_score_loss(y_true, y_probs))
        return {
            "brier_score": brier,
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        }
    except Exception as e:
        logger.error(f"Calibration: Calculation failed: {e}")
        return {
            "brier_score": 1.0,
            "prob_true": [],
            "prob_pred": []
        }
