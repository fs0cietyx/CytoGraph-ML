import yaml
from pathlib import Path
from typing import Any, Dict

# Standard Directory Structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
CONFIG_DIR = BASE_DIR / "configs"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"
FIGURE_DIR = BASE_DIR / "figures"

def load_config(config_name: str) -> Dict[str, Any]:
    """Loads a YAML configuration file from the configs directory."""
    # Handle optional extension
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
