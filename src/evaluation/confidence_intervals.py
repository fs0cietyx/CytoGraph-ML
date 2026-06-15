import numpy as np
from src.utils.logging import logger

def compute_confidence_intervals(scores: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """
    Computes a confidence interval for a list of evaluation scores
    using the standard error of the mean (SEM).
    """
    logger.info("ConfidenceIntervals: Computing metrics confidence intervals...")
    n = len(scores)
    if n <= 1:
        val = np.mean(scores) if scores else 0.0
        return float(val), float(val)
        
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))
    
    # Using 1.96 as standard multiplier for 95% confidence
    sem = std / np.sqrt(n)
    margin = 1.96 * sem
    
    return float(mean - margin), float(mean + margin)
