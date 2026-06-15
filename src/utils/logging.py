import logging
import sys
from pathlib import Path

def setup_logger(name="CytoGraph-ML"):
    """Sets up a standardized logger for all framework modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Stream Handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(log_format)
        logger.addHandler(sh)
        
        # File Handler (saved to root directory)
        project_root = Path(__file__).resolve().parent.parent.parent
        fh = logging.FileHandler(project_root / "pipeline.log")
        fh.setFormatter(log_format)
        logger.addHandler(fh)
        
    return logger

logger = setup_logger()
