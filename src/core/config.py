import logging
import sys
from pathlib import Path
from pydantic import BaseModel, Field

class PipelineConfig(BaseModel):
    """Configuration schema for the Cancer Growth Prediction Pipeline."""
    
    # Paths
    BASE_DIR: Path = Path("/Users/mainakbiswas/Documents/AI_Vault/cancer-cell-growth-prediction")
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"
    RESULTS_DIR: Path = BASE_DIR / "results"
    
    # Data Files
    FEATURES_PATH: Path = DATA_DIR / "data.csv"
    LABELS_PATH: Path = DATA_DIR / "labels.csv"
    
    # Statistical Hyperparameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    N_FOLD: int = 5
    
    # Model Hyperparameters
    RF_ESTIMATORS: int = 100
    RF_MAX_FEATURES: str = "sqrt"
    RF_N_JOBS: int = -1
    
    # Feature Selection
    VARIANCE_THRESHOLD: float = 0.01

def setup_logging():
    """Configures the enterprise logging suite."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log")
        ]
    )
    return logging.getLogger("APEX-Pipeline")

# Initialize global config and logger
config = PipelineConfig()
logger = setup_logging()
